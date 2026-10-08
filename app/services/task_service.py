"""Wellness Task Service — personal to-do CRUD.

All operations are owner-scoped (user_id from session, never from URL).

Full implementation in Phase 6.
"""

from datetime import datetime

from app.extensions import db
from app.models import WellnessTask


def create_task(user_id, title, description=None, activity_id=None):
    """Create a new wellness task for the user.

    Args:
        user_id: Owner's user ID.
        title: Task title (max 200 chars).
        description: Optional description.
        activity_id: Optional linked activity identifier slug.

    Returns:
        The created WellnessTask.
    """
    task = WellnessTask(
        user_id=user_id,
        title=title,
        description=description,
        activity_id=activity_id,
    )
    db.session.add(task)
    db.session.commit()
    return task


def get_active_activity_task(user_id, activity_id):
    """Read-only query for an active (uncompleted) task for an activity.

    Args:
        user_id: Owner's user ID.
        activity_id: Stable activity identifier slug.

    Returns:
        WellnessTask | None: Active task if found, else None.
    """
    return WellnessTask.query.filter_by(
        user_id=user_id,
        activity_id=activity_id,
        completed=False,
    ).first()


def start_activity_task(user_id, activity_id):
    """Start an activity task: retrieve active task or create a new active task.
    Concurrency-safe via database partial unique index.

    Args:
        user_id: Owner's user ID.
        activity_id: Stable activity identifier slug.

    Returns:
        tuple[WellnessTask | None, bool]: (task, created_flag)
    """
    from sqlalchemy.exc import IntegrityError
    from app.services.activity_service import get_activity_by_id

    activity = get_activity_by_id(activity_id)
    if not activity:
        return None, False

    existing = get_active_activity_task(user_id, activity_id)
    if existing:
        return existing, False

    try:
        task = create_task(
            user_id=user_id,
            title=activity['title'],
            description=activity['short_description'],
            activity_id=activity_id,
        )
        return task, True
    except IntegrityError as exc:
        db.session.rollback()
        # Race recovery: check if a concurrent thread successfully inserted the active task
        active_task = get_active_activity_task(user_id, activity_id)
        if active_task:
            return active_task, False
        # Unrelated IntegrityError (e.g. FK constraint failure) -> re-raise
        raise exc


def complete_activity_task(user_id, activity_id):
    """Mark the active task associated with an activity as complete.

    Args:
        user_id: Owner's user ID.
        activity_id: Stable activity identifier slug.

    Returns:
        WellnessTask | None: Updated task if active task existed, or None if no active task.
    """
    task = get_active_activity_task(user_id, activity_id)
    if not task:
        return None

    task.mark_complete()
    db.session.commit()
    return task


def complete_task(task_id, user_id):
    """Mark a task as completed. Verifies ownership.

    Args:
        task_id: The task to complete.
        user_id: The requesting user's ID (for IDOR protection).

    Returns:
        The updated WellnessTask, or None if not found / not owned.
    """
    task = WellnessTask.query.filter_by(id=task_id, user_id=user_id).first()
    if task is None:
        return None
    task.mark_complete()
    db.session.commit()
    return task


def delete_task(task_id, user_id):
    """Delete a task. Verifies ownership.

    Returns:
        True if deleted, False if not found / not owned.
    """
    task = WellnessTask.query.filter_by(id=task_id, user_id=user_id).first()
    if task is None:
        return False
    db.session.delete(task)
    db.session.commit()
    return True


def get_tasks(user_id, include_completed=False):
    """Return tasks for a user.

    Args:
        user_id: Owner's user ID.
        include_completed: If False, returns only pending tasks.

    Returns:
        List of WellnessTask objects.
    """
    query = WellnessTask.query.filter_by(user_id=user_id)
    if not include_completed:
        query = query.filter_by(completed=False)
    return query.order_by(WellnessTask.created_at.desc()).all()


def get_completed_count_last_7_days(user_id):
    """Count tasks completed in the last 7 days (for wellness score)."""
    from datetime import timedelta, timezone
    seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
    return (
        WellnessTask.query
        .filter(
            WellnessTask.user_id == user_id,
            WellnessTask.completed.is_(True),
            WellnessTask.completed_at >= seven_days_ago,
        )
        .count()
    )

