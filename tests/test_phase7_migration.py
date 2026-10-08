"""Tests for Phase 7.1 — Migration & Duplicate Data Resolution."""

import sqlalchemy as sa
import pytest
from app.models import User, WellnessTask
from tests.conftest import create_test_student


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

        # Execute migration duplicate-resolution SQL query
        connection = db.session.connection()
        connection.execute(sa.text("""
            UPDATE wellness_tasks
            SET completed = 1, completed_at = created_at
            WHERE id IN (
                SELECT t1.id
                FROM wellness_tasks t1
                JOIN wellness_tasks t2 ON t1.user_id = t2.user_id 
                                     AND t1.activity_id = t2.activity_id 
                                     AND t1.completed = 0 
                                     AND t2.completed = 0
                                     AND t1.id > t2.id
                WHERE t1.activity_id IS NOT NULL
            )
        """))
        db.session.commit()

        # Re-create partial unique index
        db.session.execute(sa.text("""
            CREATE UNIQUE INDEX uix_task_user_activity_active 
            ON wellness_tasks (user_id, activity_id) 
            WHERE completed = 0 AND activity_id IS NOT NULL
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
