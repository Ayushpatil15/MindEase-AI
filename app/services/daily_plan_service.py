"""Daily Wellness Plan & Check-In Flow Service.

Generates a dynamic daily plan for a student based strictly on current persisted state
(today's check-ins, active activity tasks, latest assessments, mood, facial scans, and onboarding goals).
Zero new database tables; zero AI/LLM dependencies.
"""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from app.models import MoodEntry, StressAssessment, EmotionScan, WellnessTask, UserPreferences
from app.services import stress_service, mood_service, emotion_service, task_service
from app.services.activity_service import get_activity_by_id
from app.utils.validators import validate_timezone


def get_user_today_utc_range(user_tz_str='UTC'):
    """Compute local calendar start/end of day converted to UTC for DB querying.

    Args:
        user_tz_str (str): Student's IANA timezone string (e.g., 'Asia/Kolkata').

    Returns:
        tuple[datetime, datetime, datetime, datetime]: (now_utc, start_of_day_utc, end_of_day_utc, now_local)
    """
    valid_tz_name = validate_timezone(user_tz_str)
    user_tz = ZoneInfo(valid_tz_name)
    now_utc = datetime.now(timezone.utc)

    # Current local time in user's timezone
    now_local = now_utc.astimezone(user_tz)
    local_date = now_local.date()

    # Local start (00:00:00) and end (23:59:59.999999) of day
    local_start = datetime.combine(local_date, datetime.min.time(), tzinfo=user_tz)
    local_end = datetime.combine(local_date, datetime.max.time(), tzinfo=user_tz)

    # Convert local day boundaries back to UTC timestamps for DB filtering
    start_of_day_utc = local_start.astimezone(timezone.utc)
    end_of_day_utc = local_end.astimezone(timezone.utc)

    return now_utc, start_of_day_utc, end_of_day_utc, now_local


def get_today_utc_range():
    """Backward compatible wrapper returning UTC calendar range."""
    now_utc, start_of_day, end_of_day, _ = get_user_today_utc_range('UTC')
    return now_utc, start_of_day, end_of_day


def build_daily_plan_context(user_id):
    """Fetch read-only records required for daily plan calculation if not provided."""
    return {
        'latest_stress': stress_service.get_latest(user_id),
        'latest_mood': mood_service.get_latest(user_id),
        'latest_emotion': emotion_service.get_latest(user_id),
        'preferences': UserPreferences.query.filter_by(user_id=user_id).first(),
        'pending_tasks': task_service.get_tasks(user_id, include_completed=False),
    }


def get_daily_plan(user_id, context=None):
    """Generate normalized daily plan dictionary for a student user.

    Args:
        user_id (int): Owner student user ID.
        context (dict, optional): Pre-fetched context to avoid duplicate DB queries.

    Returns:
        dict: Normalized daily plan data payload.
    """
    if context is None:
        context = build_daily_plan_context(user_id)

    pref = context.get('preferences')
    user_tz_str = pref.timezone if (pref and pref.timezone) else 'UTC'

    now_utc, start_of_day, end_of_day, now_local = get_user_today_utc_range(user_tz_str)

    # 1. Check-Ins Completed Today (using user's local calendar day boundaries in UTC)
    mood_today = MoodEntry.query.filter(
        MoodEntry.user_id == user_id,
        MoodEntry.created_at >= start_of_day,
        MoodEntry.created_at <= end_of_day
    ).first()

    stress_today = StressAssessment.query.filter(
        StressAssessment.user_id == user_id,
        StressAssessment.created_at >= start_of_day,
        StressAssessment.created_at <= end_of_day
    ).first()

    emotion_today = EmotionScan.query.filter(
        EmotionScan.user_id == user_id,
        EmotionScan.created_at >= start_of_day,
        EmotionScan.created_at <= end_of_day
    ).first()

    mood_completed = mood_today is not None
    stress_completed = stress_today is not None
    emotion_completed = emotion_today is not None
    all_completed = mood_completed and stress_completed and emotion_completed

    checkins = {
        'mood_completed': mood_completed,
        'stress_completed': stress_completed,
        'emotion_completed': emotion_completed,
        'all_completed': all_completed,
    }

    # 2. Extract State Objects from Context
    latest_stress = context.get('latest_stress')
    latest_mood = context.get('latest_mood')
    latest_emotion = context.get('latest_emotion')
    goals = pref.wellness_goals if (pref and pref.wellness_goals) else []
    preferred_time = pref.preferred_checkin_time if pref else None

    # Check for existing active activity task (Priority 1)
    pending_tasks = context.get('pending_tasks')
    if pending_tasks is None:
        pending_tasks = task_service.get_tasks(user_id, include_completed=False)
    active_activity_task = next((t for t in pending_tasks if t.activity_id is not None), None)

    # 3. Deterministic Focus Resolution Cascade
    focus = None

    # Priority 1: Resume Active Activity Task
    if active_activity_task:
        act = get_activity_by_id(active_activity_task.activity_id)
        if act:
            focus = {
                'title': act['title'],
                'category': act['category'],
                'description': act['short_description'],
                'reason': 'Continue your active wellness activity session.',
                'activity_id': act['id'],
                'is_active_task': True,
                'task_id': active_activity_task.id,
            }

    # Priority 2: High Stress
    if not focus and latest_stress and latest_stress.level == 'High':
        act = get_activity_by_id('box_breathing')
        if act:
            focus = {
                'title': act['title'],
                'category': act['category'],
                'description': act['short_description'],
                'reason': 'Recommended because your latest stress assessment indicated high stress level.',
                'activity_id': act['id'],
                'is_active_task': False,
            }

    # Priority 3: Low Mood
    if not focus and latest_mood and latest_mood.valence <= -1:
        act = get_activity_by_id('self_compassion')
        if act:
            focus = {
                'title': act['title'],
                'category': act['category'],
                'description': act['short_description'],
                'reason': 'Recommended because your recent mood check-in registered a lower mood.',
                'activity_id': act['id'],
                'is_active_task': False,
            }

    # Priority 4: Goal-Based Selections
    if not focus and 'study_productivity' in goals:
        act = get_activity_by_id('pomodoro_focus')
        if act:
            focus = {
                'title': act['title'],
                'category': act['category'],
                'description': act['short_description'],
                'reason': 'Suggested based on your onboarding wellness goal: Study Productivity.',
                'activity_id': act['id'],
                'is_active_task': False,
            }

    if not focus and 'sleep' in goals:
        act = get_activity_by_id('screen_break')
        if act:
            focus = {
                'title': act['title'],
                'category': act['category'],
                'description': act['short_description'],
                'reason': 'Suggested based on your onboarding wellness goal: Sleep.',
                'activity_id': act['id'],
                'is_active_task': False,
            }

    if not focus and ('healthier_routines' in goals or 'stress' in goals):
        act = get_activity_by_id('movement_break')
        if act:
            focus = {
                'title': act['title'],
                'category': act['category'],
                'description': act['short_description'],
                'reason': 'Suggested based on your onboarding wellness goal: Healthier Routines.',
                'activity_id': act['id'],
                'is_active_task': False,
            }

    # Priority 5: Default Fallback Focus
    if not focus:
        act = get_activity_by_id('grounding_54321')
        focus = {
            'title': act['title'],
            'category': act['category'],
            'description': act['short_description'],
            'reason': 'Suggested to help establish your daily wellness routine.',
            'activity_id': act['id'],
            'is_active_task': False,
        }

    # 4. Next Actions
    next_actions = []
    if not mood_completed:
        next_actions.append({
            'type': 'checkin',
            'title': 'Log Mood Check-In',
            'action_url': '/mood/',
        })
    if not stress_completed:
        next_actions.append({
            'type': 'checkin',
            'title': 'Take Stress Assessment',
            'action_url': '/stress/',
        })
    if not emotion_completed:
        next_actions.append({
            'type': 'checkin',
            'title': 'Facial Expression Check-In',
            'action_url': '/emotion/',
        })

    # 5. Preferred Check-In Time Reminder Window Check (evaluated in student's local timezone)
    is_reminder_time = False
    reminder_message = None
    if preferred_time:
        try:
            pref_hour = int(preferred_time.split(':')[0])
            current_hour = now_local.hour
            # Active if within 1 hour before or after preferred hour in local time
            if abs(current_hour - pref_hour) <= 1:
                is_reminder_time = True
                reminder_message = f"It's around your preferred check-in time ({preferred_time})!"
        except Exception:
            pass

    reminder = {
        'preferred_time': preferred_time,
        'is_reminder_time': is_reminder_time,
        'message': reminder_message,
    }

    # 6. Progress Metrics
    completed_today_count = WellnessTask.query.filter(
        WellnessTask.user_id == user_id,
        WellnessTask.completed.is_(True),
        WellnessTask.completed_at >= start_of_day,
        WellnessTask.completed_at <= end_of_day
    ).count()

    progress = {
        'completed_today_count': completed_today_count,
        'active_tasks_count': len(pending_tasks),
    }

    return {
        'date': now_local.strftime('%Y-%m-%d'),
        'checkins': checkins,
        'focus': focus,
        'next_actions': next_actions,
        'reminder': reminder,
        'progress': progress,
    }

