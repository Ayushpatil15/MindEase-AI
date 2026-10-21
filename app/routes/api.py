"""JSON API routes — AJAX endpoints for dynamic features.

Full implementation across Phases 2–7.
"""

from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user
from app.utils.decorators import student_required
from app.services.dashboard_service import get_student_dashboard_data
from app.services.history_service import get_wellness_history_data
from app.services.daily_plan_service import get_daily_plan
from app.services import task_service

api_bp = Blueprint('api', __name__, url_prefix='/api')


@api_bp.route('/health')
def health_check():
    """Simple health-check endpoint for verifying the app is running."""
    return jsonify({'status': 'ok', 'message': 'MindEase AI is running.'}), 200


@api_bp.route('/dashboard')
@login_required
@student_required
def get_dashboard_data():
    """Return normalized dashboard payload for current authenticated student."""
    data = get_student_dashboard_data(current_user.id)
    return jsonify({'status': 'success', 'dashboard': data}), 200


@api_bp.route('/tasks', methods=['POST'])
@login_required
@student_required
def create_task_endpoint():
    """Create a new wellness task for the current student."""
    json_data = request.get_json(silent=True) or {}
    title = json_data.get('title', '').strip()
    description = json_data.get('description', '').strip() or None

    if not title:
        return jsonify({'status': 'error', 'message': 'Task title is required.'}), 400

    task = task_service.create_task(current_user.id, title, description)
    return jsonify({
        'status': 'success',
        'task': {
            'id': task.id,
            'title': task.title,
            'description': task.description,
            'created_at': task.created_at.strftime('%Y-%m-%d %H:%M UTC'),
        }
    }), 201


@api_bp.route('/tasks/<int:task_id>/complete', methods=['POST'])
@login_required
@student_required
def complete_task_endpoint(task_id):
    """Mark a task as completed for the current student (IDOR protected)."""
    task = task_service.complete_task(task_id, current_user.id)
    if not task:
        return jsonify({'status': 'error', 'message': 'Task not found or access denied.'}), 404

    return jsonify({
        'status': 'success',
        'task': {
            'id': task.id,
            'title': task.title,
            'completed': task.completed,
            'completed_at': task.completed_at.strftime('%Y-%m-%d %H:%M UTC') if task.completed_at else None,
        }
    }), 200


@api_bp.route('/activities')
@login_required
@student_required
def get_activities_list_endpoint():
    """Return JSON list of available activity definitions."""
    from app.services.activity_service import get_all_activities
    return jsonify({
        'status': 'success',
        'activities': get_all_activities(),
    }), 200


@api_bp.route('/activities/<activity_id>')
@login_required
@student_required
def get_activity_detail_endpoint(activity_id):
    """Return JSON detail for a specific activity."""
    from app.services.activity_service import get_activity_by_id
    activity = get_activity_by_id(activity_id)
    if not activity:
        return jsonify({'status': 'error', 'message': 'Activity not found.'}), 404

    return jsonify({
        'status': 'success',
        'activity': activity,
    }), 200


@api_bp.route('/activities/<activity_id>/start', methods=['POST'])
@login_required
@student_required
def start_activity_endpoint(activity_id):
    """Explicitly start an activity, creating or retrieving active WellnessTask (CSRF protected)."""
    task, created = task_service.start_activity_task(current_user.id, activity_id)
    if not task:
        return jsonify({'status': 'error', 'message': 'Activity not found or invalid.'}), 404

    status_code = 201 if created else 200
    return jsonify({
        'status': 'success',
        'created': created,
        'message': f"Started activity '{activity_id}'.",
        'task': {
            'id': task.id,
            'activity_id': task.activity_id,
            'title': task.title,
            'completed': task.completed,
            'created_at': task.created_at.strftime('%Y-%m-%d %H:%M UTC'),
        }
    }), status_code


@api_bp.route('/activities/<activity_id>/complete', methods=['POST'])
@login_required
@student_required
def complete_activity_endpoint(activity_id):
    """Mark the active task associated with an activity as completed (CSRF protected)."""
    task = task_service.complete_activity_task(current_user.id, activity_id)
    if not task:
        return jsonify({'status': 'error', 'message': 'No active task found for this activity. Please start the activity first.'}), 400

    return jsonify({
        'status': 'success',
        'message': f"Completed activity '{activity_id}'.",
        'task': {
            'id': task.id,
            'activity_id': task.activity_id,
            'title': task.title,
            'completed': task.completed,
            'completed_at': task.completed_at.strftime('%Y-%m-%d %H:%M UTC') if task.completed_at else None,
        }
    }), 200


@api_bp.route('/history')
@login_required
@student_required
def get_history_endpoint():
    """Return windowed historical data and deterministic insights for current student."""
    timeframe = request.args.get('range', '7d')
    history_data = get_wellness_history_data(current_user.id, timeframe=timeframe)
    return jsonify({
        'status': 'success',
        'history': history_data,
    }), 200


@api_bp.route('/daily-plan')
@login_required
@student_required
def get_daily_plan_endpoint():
    """Return normalized daily plan payload for current authenticated student."""
    plan_data = get_daily_plan(current_user.id)
    return jsonify({
        'status': 'success',
        'daily_plan': plan_data,
    }), 200


@api_bp.route('/activities/task_chunking/chunk', methods=['POST'])
@login_required
@student_required
def chunk_task_endpoint():
    """Create 3 WellnessTasks from task chunking activity (CSRF protected).

    Atomically creates 3 micro-steps, runs crisis screening, and enforces duplicate guard.
    """
    payload = request.get_json(silent=True) or {}
    big_task = payload.get('big_task', '')
    micro_steps = payload.get('micro_steps', [])

    response_data, status_code = task_service.create_chunked_tasks(
        user_id=current_user.id,
        big_task=big_task,
        micro_steps=micro_steps,
    )
    return jsonify(response_data), status_code






