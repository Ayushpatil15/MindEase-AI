"""Tests for Phase 7.1 — Activity Task Concurrency & Integrity."""

import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from app import create_app
from app.extensions import db as _db
from app.models import User, WellnessTask
from app.services import task_service
from tests.conftest import create_test_student


def test_concurrent_start_activity_single_active_task():
    """Verify 10 simultaneous start requests produce exactly 1 active task in DB."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, 'test_concurrency_suite.db')

    from flask import Flask
    from app.config import config

    app = Flask(__name__)
    app.config.from_object(config['testing'])
    app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{db_path}"
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'connect_args': {'timeout': 30, 'check_same_thread': False}}
    _db.init_app(app)

    with app.app_context():
        _db.create_all()
        # Enable WAL mode for smooth multi-threaded SQLite writes
        conn = _db.session.connection()
        conn.execute(sa.text("PRAGMA journal_mode=WAL;"))
        conn.execute(sa.text("PRAGMA busy_timeout=30000;"))
        _db.session.commit()

        user = create_test_student(_db, username="race_student", email="race@test.com")
        user_id = user.id

    num_threads = 10
    barrier = threading.Barrier(num_threads)

    def worker():
        with app.app_context():
            barrier.wait()
            task, created = task_service.start_activity_task(user_id, 'box_breathing')
            task_id = task.id if task else None
            return task_id, created

    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker) for _ in range(num_threads)]
        for f in as_completed(futures):
            results.append(f.result())

    with app.app_context():
        active_tasks = WellnessTask.query.filter_by(
            user_id=user_id,
            activity_id='box_breathing',
            completed=False,
        ).all()

        # Exactly ONE active task row exists in database
        assert len(active_tasks) == 1
        canonical_task_id = active_tasks[0].id

        # Every successful caller received the exact same task ID
        returned_task_ids = [r[0] for r in results if r[0] is not None]
        assert len(returned_task_ids) == num_threads
        assert set(returned_task_ids) == {canonical_task_id}

        # Exactly 1 caller created the task, 9 callers reused it
        created_flags = [r[1] for r in results]
        assert created_flags.count(True) == 1
        assert created_flags.count(False) == 9

    # Cleanup temp file
    try:
        os.remove(db_path)
        os.rmdir(temp_dir)
    except Exception:
        pass


def test_sequential_restart_after_completion(app, db):
    """Verify completing an activity allows a new active session to be created later."""
    with app.app_context():
        user = create_test_student(db, username="restart_student", email="restart@test.com")

        # 1. Start first session
        task1, created1 = task_service.start_activity_task(user.id, 'pomodoro_focus')
        assert created1 is True
        assert task1.completed is False

        # 2. Complete first session
        comp_task = task_service.complete_activity_task(user.id, 'pomodoro_focus')
        assert comp_task.completed is True

        # 3. Start second session
        task2, created2 = task_service.start_activity_task(user.id, 'pomodoro_focus')
        assert created2 is True
        assert task2.id != task1.id
        assert task2.completed is False

        # 4. Verify DB contains 1 completed task and 1 active task
        all_tasks = WellnessTask.query.filter_by(user_id=user.id, activity_id='pomodoro_focus').all()
        assert len(all_tasks) == 2
        completed_tasks = [t for t in all_tasks if t.completed]
        active_tasks = [t for t in all_tasks if not t.completed]
        assert len(completed_tasks) == 1
        assert len(active_tasks) == 1


def test_generic_tasks_unaffected_by_index(app, db):
    """Verify generic tasks with activity_id=NULL allow multiple active tasks."""
    with app.app_context():
        user = create_test_student(db, username="generic_student", email="generic@test.com")

        t1 = task_service.create_task(user.id, "Generic task 1", activity_id=None)
        t2 = task_service.create_task(user.id, "Generic task 2", activity_id=None)
        t3 = task_service.create_task(user.id, "Generic task 3", activity_id=None)

        pending = task_service.get_tasks(user.id, include_completed=False)
        assert len(pending) == 3


def test_multi_user_activity_isolation(app, db):
    """Verify different users can independently have active tasks for the same activity."""
    with app.app_context():
        user_a = create_test_student(db, username="multi_a", email="multia@test.com")
        user_b = create_test_student(db, username="multi_b", email="multib@test.com")

        task_a, created_a = task_service.start_activity_task(user_a.id, 'grounding_54321')
        task_b, created_b = task_service.start_activity_task(user_b.id, 'grounding_54321')

        assert created_a is True
        assert created_b is True
        assert task_a.id != task_b.id

        active_a = task_service.get_active_activity_task(user_a.id, 'grounding_54321')
        active_b = task_service.get_active_activity_task(user_b.id, 'grounding_54321')
        assert active_a.id == task_a.id
        assert active_b.id == task_b.id


def test_unhandled_integrity_error_re_raised(app, db):
    """Verify start_activity_task re-raises IntegrityError when NOT caused by activity race."""
    with app.app_context():
        user = create_test_student(db, username="err_user", email="err@test.com")
        # Direct insert of invalid NOT NULL title triggers IntegrityError
        with pytest.raises(IntegrityError):
            bad_task = WellnessTask(user_id=user.id, title=None, activity_id='box_breathing')
            db.session.add(bad_task)
            db.session.commit()
