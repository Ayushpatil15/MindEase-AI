"""Wellness Calculator Service.

Authoritative implementation of the MindEase AI Wellness Score.
"""

from datetime import datetime, timedelta, timezone
from app.extensions import db
from app.models.user import User
from app.models.stress import StressAssessment
from app.models.mood import MoodEntry


def calculate_wellness_score(user_id):
    """Calculate the MVP Wellness Score for a student.

    Formula:
    S = ((44 - latest_stress) / 44) * 100
    M = ((average_valence_7d + 2) / 4) * 100
    C = (active_days_7d / elapsed_days) * 100

    Weights:
    - Both S and M: S=40%, M=40%, C=20%
    - S only: S=80%, C=20%
    - M only: M=80%, C=20%
    - Neither: None (insufficient data)

    Returns:
        dict: {'status': 'available'|'insufficient_data', 'score': int|None}
    """
    user = db.session.get(User, user_id)
    if not user:
        return {'status': 'insufficient_data', 'score': None}

    now = datetime.now(timezone.utc)
    seven_days_ago = now - timedelta(days=7)

    # 1. Fetch Stress (Latest ever)
    latest_stress = (
        StressAssessment.query
        .filter_by(user_id=user_id)
        .order_by(StressAssessment.created_at.desc())
        .first()
    )

    # 2. Fetch Mood (Last 7 days)
    recent_moods = (
        MoodEntry.query
        .filter(MoodEntry.user_id == user_id, MoodEntry.created_at >= seven_days_ago)
        .all()
    )

    # If neither exists, insufficient data
    if not latest_stress and not recent_moods:
        return {'status': 'insufficient_data', 'score': None}

    # 3. Fetch recent stress for Consistency check (Last 7 days)
    recent_stress = (
        StressAssessment.query
        .filter(StressAssessment.user_id == user_id, StressAssessment.created_at >= seven_days_ago)
        .all()
    )

    # --- Normalizations ---

    # S: Stress Normalization
    S = None
    if latest_stress:
        S = ((44 - latest_stress.score) / 44) * 100.0

    # M: Mood Normalization
    M = None
    if recent_moods:
        avg_valence = sum(m.valence for m in recent_moods) / len(recent_moods)
        M = ((avg_valence + 2) / 4) * 100.0

    # C: Consistency Normalization
    # Unique active calendar days (UTC) in last 7 days
    active_dates = set()
    for m in recent_moods:
        active_dates.add(m.created_at.date())
    for s in recent_stress:
        active_dates.add(s.created_at.date())

    unique_active_days = len(active_dates)

    # Elapsed days since registration
    delta_days = (now.date() - user.created_at.date()).days
    calendar_days_since_reg = max(0, delta_days)
    elapsed_days = max(1, min(7, calendar_days_since_reg + 1))

    C = (unique_active_days / elapsed_days) * 100.0
    # Explicitly clamp C
    C = max(0.0, min(100.0, C))

    # --- Weighting ---
    final_score = None

    if S is not None and M is not None:
        final_score = (S * 0.40) + (M * 0.40) + (C * 0.20)
    elif S is not None and M is None:
        final_score = (S * 0.80) + (C * 0.20)
    elif M is not None and S is None:
        final_score = (M * 0.80) + (C * 0.20)

    # Bounds and Rounding
    final_score = max(0.0, min(100.0, final_score))
    final_score = int(round(final_score))

    return {
        'status': 'available',
        'score': final_score
    }
