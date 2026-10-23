"""Tests for Phase 7.1 — Migration & Duplicate Data Resolution."""

import importlib
from unittest.mock import MagicMock, patch

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql, sqlite

from app.models import User, WellnessTask
from tests.conftest import create_test_student

phase7_migration = importlib.import_module(
    "migrations.versions.5e6f7a8b9c0d_phase7_1_concurrency_unique_index"
)


def test_pre_existing_duplicate_active_task_resolution(app, db):
    """Verify duplicate active tasks in existing DB are safely resolved by marking excess duplicates completed."""
    with app.app_context():
        # Drop index temporarily to simulate pre-migration table state
        db.session.execute(sa.text("DROP INDEX IF EXISTS uix_task_user_activity_active"))
        db.session.commit()

        user = create_test_student(db, username="mig_dup_user", email="migdup@test.com")

        # Manually create 3 active tasks for same user & activity (pre-migration state)
        t1 = WellnessTask(user_id=user.id, activity_id='tension_release', title='Stretch 1', completed=False)
        t2 = WellnessTask(user_id=user.id, activity_id='tension_release', title='Stretch 2', completed=False)
        t3 = WellnessTask(user_id=user.id, activity_id='tension_release', title='Stretch 3', completed=False)
        db.session.add_all([t1, t2, t3])
        db.session.commit()

        # Execute migration duplicate-resolution SQL query using dialect-aware generator
        connection = db.session.connection()
        dialect_name = getattr(connection.dialect, 'name', 'sqlite')
        resolution_sql = phase7_migration.get_duplicate_resolution_sql(dialect_name)
        connection.execute(resolution_sql)
        db.session.commit()

        # Re-create partial unique index using dialect-aware where clause
        is_pg = dialect_name in ('postgresql', 'postgres')
        where_clause = 'completed = false AND activity_id IS NOT NULL' if is_pg else 'completed = 0 AND activity_id IS NOT NULL'
        db.session.execute(sa.text(f"""
            CREATE UNIQUE INDEX uix_task_user_activity_active 
            ON wellness_tasks (user_id, activity_id) 
            WHERE {where_clause}
        """))
        db.session.commit()

        # Verify results:
        # t1 (oldest / lowest id) remains completed = False
        # t2, t3 (newer duplicate active tasks) are updated to completed = True
        db.session.refresh(t1)
        db.session.refresh(t2)
        db.session.refresh(t3)

        assert t1.completed is False
        assert t2.completed is True
        assert t3.completed is True

        # Total tasks preserved = 3
        all_tasks = WellnessTask.query.filter_by(user_id=user.id, activity_id='tension_release').all()
        assert len(all_tasks) == 3


def test_partial_unique_index_enforces_active_uniqueness(app, db):
    """Verify partial unique index rejects duplicate active task insertion at DB layer."""
    with app.app_context():
        user = create_test_student(db, username="index_test_user", email="indextest@test.com")

        # First active task succeeds
        t1 = WellnessTask(user_id=user.id, activity_id='screen_break', title='Screen Break 1', completed=False)
        db.session.add(t1)
        db.session.commit()

        # Second active task for same user & activity fails at DB layer
        t2 = WellnessTask(user_id=user.id, activity_id='screen_break', title='Screen Break 2', completed=False)
        db.session.add(t2)

        with pytest.raises(sa.exc.IntegrityError):
            db.session.commit()

        db.session.rollback()


def test_migration_sql_postgres_and_sqlite_dialects():
    """Verify get_duplicate_resolution_sql produces dialect-safe SQL for Postgres and SQLite."""
    # Test PostgreSQL dialect variants
    for pg_dialect in ('postgresql', 'postgres', 'PostgreSQL', 'POSTGRES'):
        sql_clause = phase7_migration.get_duplicate_resolution_sql(pg_dialect)
        sql_str = str(sql_clause)

        # Must use native boolean literals
        assert 'SET completed = true' in sql_str
        assert 't1.completed = false' in sql_str
        assert 't2.completed = false' in sql_str

        # Must NOT contain integer boolean comparisons that fail on PostgreSQL
        assert 'completed = 0' not in sql_str
        assert 'completed = 1' not in sql_str

        # Must compile under PostgreSQL dialect
        compiled = str(sql_clause.compile(dialect=postgresql.dialect()))
        assert 'SET completed = true' in compiled

    # Test SQLite dialect
    for sqlite_dialect in ('sqlite', 'SQLite'):
        sql_clause = phase7_migration.get_duplicate_resolution_sql(sqlite_dialect)
        sql_str = str(sql_clause)

        # Must use integer booleans for SQLite compatibility
        assert 'SET completed = 1' in sql_str
        assert 't1.completed = 0' in sql_str
        assert 't2.completed = 0' in sql_str

        # Must NOT use boolean literals in SQLite mode
        assert 'completed = false' not in sql_str
        assert 'completed = true' not in sql_str

        # Must compile under SQLite dialect
        compiled = str(sql_clause.compile(dialect=sqlite.dialect()))
        assert 'SET completed = 1' in compiled


def test_migration_upgrade_uses_dialect_from_bind():
    """Verify upgrade() inspects connection dialect and executes matching SQL."""
    # PostgreSQL connection mock
    mock_pg_conn = MagicMock()
    mock_pg_conn.dialect.name = 'postgresql'

    with patch.object(phase7_migration.op, 'get_bind', return_value=mock_pg_conn), \
         patch.object(phase7_migration.op, 'batch_alter_table'):
        phase7_migration.upgrade()

    assert mock_pg_conn.execute.call_count == 1
    executed_sql = str(mock_pg_conn.execute.call_args[0][0])
    assert 'SET completed = true' in executed_sql
    assert 't1.completed = false' in executed_sql
    assert 'completed = 0' not in executed_sql

    # SQLite connection mock
    mock_sqlite_conn = MagicMock()
    mock_sqlite_conn.dialect.name = 'sqlite'

    with patch.object(phase7_migration.op, 'get_bind', return_value=mock_sqlite_conn), \
         patch.object(phase7_migration.op, 'batch_alter_table'):
        phase7_migration.upgrade()

    assert mock_sqlite_conn.execute.call_count == 1
    executed_sql = str(mock_sqlite_conn.execute.call_args[0][0])
    assert 'SET completed = 1' in executed_sql
    assert 't1.completed = 0' in executed_sql
    assert 'completed = false' not in executed_sql


def test_migration_chain_integrity():
    """Verify that migration revision IDs and lineage chain are preserved."""
    mig_init = importlib.import_module("migrations.versions.3c4d4685691f_initial_schema_users_preferences_stress_")
    mig_phase7 = importlib.import_module("migrations.versions.4d5e6f7a8b9c_phase7_add_activity_id_to_wellness_tasks")
    mig_phase7_1 = phase7_migration
    mig_phase9_1 = importlib.import_module("migrations.versions.6f7a8b9c0d1e_phase9_1_user_preferences_timezone")

    assert mig_init.revision == '3c4d4685691f'
    assert mig_init.down_revision is None

    assert mig_phase7.revision == '4d5e6f7a8b9c'
    assert mig_phase7.down_revision == '3c4d4685691f'

    assert mig_phase7_1.revision == '5e6f7a8b9c0d'
    assert mig_phase7_1.down_revision == '4d5e6f7a8b9c'

    assert mig_phase9_1.revision == '6f7a8b9c0d1e'
    assert mig_phase9_1.down_revision == '5e6f7a8b9c0d'


def test_subsequent_migration_has_no_incompatible_sql():
    """Verify the next migration (6f7a8b9c0d1e) does not contain dialect-incompatible SQL."""
    mig_phase9_1 = importlib.import_module("migrations.versions.6f7a8b9c0d1e_phase9_1_user_preferences_timezone")
    import inspect
    source = inspect.getsource(mig_phase9_1)

    # Ensure no raw integer boolean comparisons or raw SQL statements exist
    assert '= 0' not in source
    assert '= 1' not in source
    assert 'execute(' not in source

