"""Dashboard Aggregation Service.

Centralized service that aggregates all student wellness signals into a single
normalized payload for the student dashboard and JSON API.
"""

from app.extensions import db
from app.models.user import User
from app.models.user_preferences import UserPreferences
from app.services import (
    wellness_calculator,
    stress_service,
    mood_service,
    emotion_service,
    recommendation_service,
    task_service,
    daily_plan_service,
)


def get_student_dashboard_data(user_id):
    """Aggregate all wellness components for a student into a unified dashboard dictionary.

    Args:
        user_id (int): The ID of the student.

    Returns:
        dict: Normalized dashboard data containing user info, wellness score,
              latest stress assessment, latest mood & 7-day trend, latest emotion scan,
              recommendations, active tasks, preferences, and recent activity.
    """
    user = db.session.get(User, user_id)
    if not user:
        return None

    # 1. Wellness Score
    wellness_score = wellness_calculator.calculate_wellness_score(user_id)

    # 2. Latest Stress Assessment
    latest_stress_obj = stress_service.get_latest(user_id)
    latest_stress = None
    if latest_stress_obj:
        latest_stress = {
            'id': latest_stress_obj.id,
            'score': latest_stress_obj.score,
            'level': latest_stress_obj.level,
            'created_at': latest_stress_obj.created_at.strftime('%Y-%m-%d %H:%M UTC'),
        }

    # 3. Latest Mood Entry & 7-Day Trend
    latest_mood_obj = mood_service.get_latest(user_id)
    latest_mood = None
    if latest_mood_obj:
        latest_mood = {
            'id': latest_mood_obj.id,
            'mood': latest_mood_obj.mood,
            'valence': latest_mood_obj.valence,
            'created_at': latest_mood_obj.created_at.strftime('%Y-%m-%d %H:%M UTC'),
        }
    mood_trend_7d = mood_service.get_7day_trend(user_id)

    # 4. Latest Emotion Scan
    latest_emotion_obj = emotion_service.get_latest(user_id)
    latest_emotion = None
    if latest_emotion_obj:
        latest_emotion = {
            'id': latest_emotion_obj.id,
            'emotion': latest_emotion_obj.emotion,
            'confidence': round(latest_emotion_obj.confidence, 4),
            'created_at': latest_emotion_obj.created_at.strftime('%Y-%m-%d %H:%M UTC'),
        }

    # 5. Recommendations (Fetch recent; if empty, generate deterministic set)
    recs_objs = recommendation_service.get_recent(user_id, limit=5)
    if not recs_objs:
        recs_objs = recommendation_service.generate_recommendations(user_id)

    recommendations = [
        {
            'id': r.id,
            'category': r.category,
            'title': r.title,
            'description': r.description,
            'reason': r.reason,
            'activity_id': recommendation_service.get_activity_id_for_recommendation(r),
            'created_at': r.created_at.strftime('%Y-%m-%d %H:%M UTC'),
        }
        for r in recs_objs
    ]

    # 6. Tasks (Pending tasks + 7-day completed count)
    pending_task_objs = task_service.get_tasks(user_id, include_completed=False)
    pending_tasks = [
        {
            'id': t.id,
            'title': t.title,
            'description': t.description,
            'activity_id': t.activity_id,
            'created_at': t.created_at.strftime('%Y-%m-%d %H:%M UTC'),
        }
        for t in pending_task_objs
    ]
    completed_7d_count = task_service.get_completed_count_last_7_days(user_id)

    # 7. User Preferences & Goals
    pref = UserPreferences.query.filter_by(user_id=user_id).first()
    preferences = {
        'goals': pref.wellness_goals if (pref and pref.wellness_goals) else [],
        'preferred_checkin_time': pref.preferred_checkin_time if pref else None,
    }

    # 8. Recent Activity Feed (Chronological events across stress, mood, emotion)
    activity_items = []
    if latest_stress_obj:
        activity_items.append({
            'type': 'stress',
            'title': 'Stress Assessment Completed',
            'summary': f'Score: {latest_stress_obj.score}/44 ({latest_stress_obj.level} Stress)',
            'created_at_dt': latest_stress_obj.created_at,
            'created_at': latest_stress_obj.created_at.strftime('%Y-%m-%d %H:%M UTC'),
        })
    if latest_mood_obj:
        activity_items.append({
            'type': 'mood',
            'title': 'Mood Check-In Logged',
            'summary': f'Mood: {latest_mood_obj.mood} (Valence: {latest_mood_obj.valence:+d})',
            'created_at_dt': latest_mood_obj.created_at,
            'created_at': latest_mood_obj.created_at.strftime('%Y-%m-%d %H:%M UTC'),
        })
    if latest_emotion_obj:
        activity_items.append({
            'type': 'emotion',
            'title': 'Facial Expression Check-In',
            'summary': f'Expression: {latest_emotion_obj.emotion} ({round(latest_emotion_obj.confidence * 100)}% confidence)',
            'created_at_dt': latest_emotion_obj.created_at,
            'created_at': latest_emotion_obj.created_at.strftime('%Y-%m-%d %H:%M UTC'),
        })

    # Sort activity feed newest first
    activity_items.sort(key=lambda x: x['created_at_dt'], reverse=True)
    recent_activity = [
        {
            'type': item['type'],
            'title': item['title'],
            'summary': item['summary'],
            'created_at': item['created_at'],
        }
        for item in activity_items[:5]
    ]

    # 9. Daily Wellness Plan (Pass pre-fetched context to eliminate duplicate DB queries)
    context = {
        'latest_stress': latest_stress_obj,
        'latest_mood': latest_mood_obj,
        'latest_emotion': latest_emotion_obj,
        'preferences': pref,
        'pending_tasks': pending_task_objs,
    }
    daily_plan = daily_plan_service.get_daily_plan(user_id, context=context)


    return {
        'user': {
            'username': user.username,
            'email': user.email,
        },
        'wellness_score': wellness_score,
        'latest_stress': latest_stress,
        'latest_mood': latest_mood,
        'mood_trend_7d': mood_trend_7d,
        'latest_emotion': latest_emotion,
        'recommendations': recommendations,
        'tasks': {
            'pending': pending_tasks,
            'completed_7d_count': completed_7d_count,
        },
        'preferences': preferences,
        'recent_activity': recent_activity,
        'daily_plan': daily_plan,
    }
