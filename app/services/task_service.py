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
    if not existing:
        from app.services.activity_service import RETIRED_ACTIVITIES_MAP
        if activity_id in RETIRED_ACTIVITIES_MAP:
            existing = get_active_activity_task(user_id, RETIRED_ACTIVITIES_MAP[activity_id])
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
        from app.services.activity_service import RETIRED_ACTIVITIES_MAP
        if activity_id in RETIRED_ACTIVITIES_MAP:
            task = get_active_activity_task(user_id, RETIRED_ACTIVITIES_MAP[activity_id])
        else:
            for retired_id, target_id in RETIRED_ACTIVITIES_MAP.items():
                if target_id == activity_id:
                    task = get_active_activity_task(user_id, retired_id)
                    if task:
                        break
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


_RECENT_CHUNKING_BATCHES = {}


def clear_chunking_duplicate_cache():
    """Clear in-memory cache of recent chunking batches (primarily for testing)."""
    _RECENT_CHUNKING_BATCHES.clear()


def create_chunked_tasks(user_id, big_task, micro_steps):
    """Atomically create 3 micro-steps as WellnessTasks for the user.

    Validates inputs, checks for crisis language at request time without persisting
    it or setting a trigger flag, guards against duplicate batches within ~30s, and
    commits all 3 tasks in a single database transaction.

    Args:
        user_id (int): User ID of the student.
        big_task (str | None): Optional overarching task title.
        micro_steps (list[str]): List of exactly 3 micro-step titles.

    Returns:
        tuple[dict, int]: (response_dict, http_status_code)
    """
    import time
    from app.services.safety_service import text_has_crisis_language, get_safety_support_card_data

    if not isinstance(micro_steps, (list, tuple)) or len(micro_steps) != 3:
        return {'status': 'error', 'message': 'Exactly 3 micro-steps are required.'}, 400

    cleaned_steps = []
    for i, step in enumerate(micro_steps, start=1):
        if not isinstance(step, str) or not step.strip():
            return {'status': 'error', 'message': f'Step {i} cannot be empty.'}, 400
        cleaned = step.strip()
        if len(cleaned) > 200:
            return {'status': 'error', 'message': f'Step {i} exceeds the 200 character limit.'}, 400
        cleaned_steps.append(cleaned)

    cleaned_big_task = (big_task or '').strip()
    if len(cleaned_big_task) > 200:
        return {'status': 'error', 'message': 'Big task title exceeds the 200 character limit.'}, 400

    # Crisis language check: run at request time across submitted text
    # Never log or persist the typed text separately and never store a trigger flag
    combined_text = f"{cleaned_big_task} " + " ".join(cleaned_steps)
    if text_has_crisis_language(combined_text):
        return {
            'status': 'crisis_detected',
            'crisis_detected': True,
            'support_card': get_safety_support_card_data(),
            'message': 'Support and care resources are available.',
        }, 200

    # Duplicate guard: reject identical repeat batch from same user within 30s
    batch_key = tuple(s.lower() for s in cleaned_steps)
    now = time.time()
    last_submission = _RECENT_CHUNKING_BATCHES.get(user_id)
    if last_submission:
        last_time, last_key = last_submission
        if (now - last_time < 30.0) and (last_key == batch_key):
            return {
                'status': 'error',
                'message': 'Duplicate batch detected. Please wait 30 seconds before submitting identical steps.',
            }, 409

    # Atomic transaction: create all 3 tasks in a single transaction
    desc = f"Part of: {cleaned_big_task}" if cleaned_big_task else "Task chunking micro-step"
    created_tasks = []
    try:
        for step_title in cleaned_steps:
            task = WellnessTask(
                user_id=user_id,
                title=step_title,
                description=desc,
                activity_id=None,
            )
            db.session.add(task)
            created_tasks.append(task)
        db.session.commit()
        _RECENT_CHUNKING_BATCHES[user_id] = (now, batch_key)
        return {
            'status': 'success',
            'message': 'Created 3 micro-tasks successfully.',
            'tasks': [
                {'id': t.id, 'title': t.title, 'description': t.description}
                for t in created_tasks
            ],
        }, 201
    except Exception:
        db.session.rollback()
        return {
            'status': 'error',
            'message': 'Failed to create micro-tasks due to a server error.',
        }, 500


