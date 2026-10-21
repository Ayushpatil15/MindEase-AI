"""Recommendation Service — rule-based engine.

Generates personalized recommendations based on actual user state.
All rules are deterministic, explainable, and testable.
"""

from app.extensions import db
from app.models import Recommendation


# Deterministic mapping from recommendation title to activity slug
RECOMMENDATION_ACTIVITY_MAP = {
    'Box Breathing': 'box_breathing',
    'Box Breathing & Grounding Exercise': 'box_breathing',
    'Self-Compassion & Reflection': 'self_compassion',
    'Physical Stretch & Tension Release': 'tension_release',
    'Structured Academic Break': 'task_chunking',
    'Consistent Evening Wind-Down': 'screen_break',
    'Pomodoro Focus Sessions': 'pomodoro_focus',
    'Hydration & Movement Reset': 'movement_break',
    'Welcome to MindEase AI': 'grounding_54321',
}


def get_activity_id_for_recommendation(rec_or_title):
    """Resolve an activity_id slug for a recommendation or title, validated against the registry.

    Args:
        rec_or_title (Recommendation | str): Recommendation object or title string.

    Returns:
        str: Valid activity identifier slug (defaults to 'grounding_54321').
    """
    from app.services.activity_service import get_activity_by_id

    title = getattr(rec_or_title, 'title', rec_or_title)
    act_id = RECOMMENDATION_ACTIVITY_MAP.get(title, 'grounding_54321')
    if get_activity_by_id(act_id):
        return act_id
    return 'grounding_54321'


def generate_recommendations(user_id):
    """Generate personalized recommendations based on current user state.

    Evaluates a deterministic rule set across stress, mood, emotion, and preferences.

    Returns:
        list[Recommendation]: Newly generated recommendations.
    """
    from app.services import stress_service, mood_service
    from app.services.safety_service import evaluate_safety
    from app.models.user_preferences import UserPreferences

    latest_stress = stress_service.get_latest(user_id)
    latest_mood = mood_service.get_latest(user_id)
    pref = UserPreferences.query.filter_by(user_id=user_id).first()
    goals = pref.wellness_goals if (pref and pref.wellness_goals) else []

    candidates = []

    # Safety Rule: Put CLINICAL_SUPPORT first when any safety trigger is active
    if evaluate_safety(user_id, latest_mood=latest_mood, latest_stress=latest_stress):
        candidates.append({
            'category': 'CLINICAL_SUPPORT',
            'title': 'Campus & Confidential Support Services',
            'description': 'Connect with campus counseling, Tele-MANAS (14416), or emergency services (112) for immediate, compassionate support.',
            'reason': 'Recommended to provide prompt access to verified support and counseling resources.',
            'activity_id': 'grounding_54321',
        })

    # Rule 1: High Stress
    if latest_stress and latest_stress.level == 'High':
        candidates.append({
            'category': 'MINDFULNESS',
            'title': 'Box Breathing',
            'description': 'Practice box breathing (inhale 4, hold 4, exhale 4, hold 4) for 5 minutes to lower physiological arousal.',
            'reason': 'Triggered because your latest stress assessment indicated high stress level.',
            'activity_id': 'box_breathing',
        })

    # Rule 2: Low Mood
    if latest_mood and latest_mood.valence <= -1:
        candidates.append({
            'category': 'MINDFULNESS',
            'title': 'Self-Compassion & Reflection',
            'description': 'Take a 5-minute pause to write down how you feel without judgment.',
            'reason': 'Triggered because your recent mood check-in registered a lower mood.',
            'activity_id': 'self_compassion',
        })

    # Rule 3: Moderate Stress
    if latest_stress and latest_stress.level == 'Moderate':
        candidates.append({
            'category': 'ACADEMIC',
            'title': 'Structured Academic Break',
            'description': 'Break study blocks into 25-minute focus periods with mandatory 5-minute rests.',
            'reason': 'Triggered because your latest stress assessment indicated moderate stress level.',
            'activity_id': 'task_chunking',
        })

    # Rule 5: Goal-Based Candidates
    if 'sleep' in goals:
        candidates.append({
            'category': 'PHYSICAL',
            'title': 'Consistent Evening Wind-Down',
            'description': 'Turn off digital screens 30 minutes before sleep to improve recovery.',
            'reason': 'Suggested based on your onboarding wellness goal: Sleep.',
            'activity_id': 'screen_break',
        })

    if 'study_productivity' in goals:
        candidates.append({
            'category': 'ACADEMIC',
            'title': 'Pomodoro Focus Sessions',
            'description': 'Set clear single-task goals for each 25-minute study period.',
            'reason': 'Suggested based on your onboarding wellness goal: Study Productivity.',
            'activity_id': 'pomodoro_focus',
        })

    if 'healthier_routines' in goals or 'stress' in goals:
        candidates.append({
            'category': 'PHYSICAL',
            'title': 'Hydration & Movement Reset',
            'description': 'Drink a glass of water and take a brief 5-minute walk.',
            'reason': 'Suggested based on your onboarding wellness goal: Healthier Routines.',
            'activity_id': 'movement_break',
        })

    # Fallback / New User Default Candidate
    if not candidates:
        candidates.append({
            'category': 'MINDFULNESS',
            'title': 'Welcome to MindEase AI',
            'description': 'Complete your first mood or stress check-in to unlock personalized insights.',
            'reason': 'Suggested to help build your initial wellness baseline.',
            'activity_id': 'grounding_54321',
        })

    # Deduplicate candidates by title and limit to 5
    seen_titles = set()
    unique_candidates = []
    for c in candidates:
        if c['title'] not in seen_titles:
            seen_titles.add(c['title'])
            unique_candidates.append(c)
            if len(unique_candidates) == 5:
                break

    # Persist newly generated recommendations to DB
    recs = []
    for c in unique_candidates:
        rec = Recommendation(
            user_id=user_id,
            category=c['category'],
            title=c['title'],
            description=c['description'],
            reason=c['reason'],
        )
        db.session.add(rec)
        recs.append(rec)

    db.session.commit()
    return recs


def get_recent(user_id, limit=5):
    """Return the most recent recommendations for a user."""
    return (
        Recommendation.query
        .filter_by(user_id=user_id)
        .order_by(Recommendation.created_at.desc())
        .limit(limit)
        .all()
    )


def refresh_recommendations(user_id):
    """Delete existing recommendations for the user and regenerate them from current state.

    Args:
        user_id (int): User identifier.

    Returns:
        list[Recommendation]: Newly generated recommendations.
    """
    Recommendation.query.filter_by(user_id=user_id).delete()
    db.session.commit()
    return generate_recommendations(user_id)


