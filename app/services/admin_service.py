"""Admin Analytics Service.

Provides aggregated, anonymized analytics for the admin dashboard.
All queries use GROUP BY / COUNT / AVG — never individual records.

Privacy rules:
- Journal notes and reflections are NEVER included.
- Individual student records are NEVER returned.
- Small-group threshold (MIN_AGGREGATION_GROUP_SIZE) applied where relevant.

Full implementation in Phase 7.
"""

from flask import current_app

from app.extensions import db
from app.models import User, StressAssessment, MoodEntry, EmotionScan


def get_student_count():
    """Count total registered students (not admins)."""
    return User.query.filter_by(role='student').count()


def get_total_checkins():
    """Count total check-ins across all types."""
    stress = StressAssessment.query.count()
    mood = MoodEntry.query.count()
    emotion = EmotionScan.query.count()
    return {'stress': stress, 'mood': mood, 'emotion': emotion, 'total': stress + mood + emotion}


def get_stress_distribution():
    """Return aggregate stress level distribution.

    Returns dict: {"Low": n, "Moderate": n, "High": n}
    Only includes levels with >= MIN_AGGREGATION_GROUP_SIZE entries.

    Full implementation in Phase 7.
    """
    return {}


def get_mood_trends():
    """Return aggregate mood distribution over time.

    Full implementation in Phase 7.
    """
    return {}


def get_emotion_distribution():
    """Return aggregate emotion class distribution.

    Full implementation in Phase 7.
    """
    return {}


def get_engagement_stats():
    """Return engagement metrics (active students, check-in frequency).

    Full implementation in Phase 7.
    """
    return {}
