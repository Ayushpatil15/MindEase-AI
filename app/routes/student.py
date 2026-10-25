import json
import os
from flask import Blueprint, render_template, abort, request, current_app
from flask_login import login_required, current_user
from app.utils.decorators import student_required
from app.services.dashboard_service import get_student_dashboard_data
from app.services.activity_service import get_all_activities, get_activity_by_id
from app.services.history_service import get_wellness_history_data
from app.services import task_service

student_bp = Blueprint('student', __name__)


@student_bp.route('/')
@login_required
@student_required
def dashboard():
    """Student dashboard — Phase 6 Holistic Wellness Dashboard."""
    data = get_student_dashboard_data(current_user.id)
    return render_template('student/dashboard.html', data=data, wellness=data['wellness_score'])


@student_bp.route('/history')
@login_required
@student_required
def history():
    """Student Wellness History & Insights view."""
    timeframe = request.args.get('range', '7d')
    history_data = get_wellness_history_data(current_user.id, timeframe=timeframe)
    return render_template(
        'student/history.html',
        history=history_data,
        history_json=history_data,
    )





@student_bp.route('/activities')
@login_required
@student_required
def activities():
    """Guided Wellness Activities catalog view."""
    all_activities = get_all_activities()
    return render_template('student/activities.html', activities=all_activities)


@student_bp.route('/activities/<activity_id>')
@login_required
@student_required
def activity_detail(activity_id):
    """Dedicated interactive guided wellness activity detail view (read-only, no task creation)."""
    activity = get_activity_by_id(activity_id)
    if not activity:
        abort(404)

    # Read-only check for existing active task (does NOT mutate DB)
    active_task = task_service.get_active_activity_task(current_user.id, activity_id)
    return render_template('student/activity_detail.html', activity=activity, active_task=active_task)


@student_bp.route('/support')
@login_required
@student_required
def support_center():
    """Support Center page displaying institution-configured support resources."""
    resource_path = current_app.config.get('SUPPORT_RESOURCES_PATH') or os.path.join(
        current_app.root_path, 'content', 'support_resources.json'
    )
    resources = []
    if os.path.exists(resource_path):
        with open(resource_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            resources = data.get('resources', [])

    counseling = [
        r for r in resources
        if r.get('category') in ('counseling', 'wellness') and r.get('is_configured') is True
    ]
    professional = [
        r for r in resources
        if r.get('category') == 'professional' and r.get('is_configured') is True
    ]
    crisis = [
        r for r in resources
        if r.get('category') == 'crisis' or r.get('is_emergency')
    ]

    return render_template(
        'student/support.html',
        counseling_resources=counseling,
        professional_resources=professional,
        crisis_resources=crisis,
    )


@student_bp.route('/checkin')
@login_required
@student_required
def checkin_hub():
    """Unified Check-In Hub landing page for Mood, Stress, and Facial Expression."""
    from app.services import stress_service
    from app.services.safety_service import evaluate_safety, get_safety_support_card_data

    latest_stress = stress_service.get_latest(current_user.id)
    safety_card = None
    if evaluate_safety(current_user.id):
        safety_card = get_safety_support_card_data()

    return render_template(
        'checkin/hub.html',
        latest_stress=latest_stress,
        safety_card=safety_card,
    )
