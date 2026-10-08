"""Stress Assessment Service.

Implements the LOCKED scoring specification:
- Exactly 11 questions, each 0–4.
- Reverse-scored items: Q4, Q5, Q7, Q8 (0-indexed: 3, 4, 6, 7).
- Reverse formula: 4 - raw_value.
- Total range: 0–44.
- Thresholds: 0–14 Low, 15–28 Moderate, 29–44 High.
- Adapted from PSS-10 concepts with academic workload addition.

Full implementation in Phase 2.
"""

from app.extensions import db
from app.models import StressAssessment

# 0-indexed positions of reverse-scored items (Q4, Q5, Q7, Q8 → indices 3, 4, 6, 7)
REVERSE_SCORED_INDICES = {3, 4, 6, 7}

# Threshold boundaries
THRESHOLD_LOW_MAX = 14
THRESHOLD_MODERATE_MAX = 28
THRESHOLD_HIGH_MAX = 44


def score_assessment(responses):
    """Score an 11-question stress assessment.

    Args:
        responses: List of exactly 11 integers, each in [0, 4].

    Returns:
        dict with keys: score (int), level (str).

    Raises:
        ValueError: If responses are invalid.
    """
    if len(responses) != 11:
        raise ValueError('Exactly 11 responses required.')
    if any(r < 0 or r > 4 for r in responses):
        raise ValueError('Each response must be between 0 and 4.')

    total = 0
    for i, raw in enumerate(responses):
        if i in REVERSE_SCORED_INDICES:
            total += 4 - raw
        else:
            total += raw

    if total <= THRESHOLD_LOW_MAX:
        level = 'Low'
    elif total <= THRESHOLD_MODERATE_MAX:
        level = 'Moderate'
    else:
        level = 'High'

    return {'score': total, 'level': level}


def save_assessment(user_id, responses, reflection=None):
    """Score and persist a stress assessment.

    Full implementation in Phase 2.
    """
    result = score_assessment(responses)

    assessment = StressAssessment(
        user_id=user_id,
        responses=responses,
        score=result['score'],
        level=result['level'],
        reflection=reflection,
    )
    db.session.add(assessment)
    db.session.commit()

    return assessment


def get_latest(user_id):
    """Return the most recent stress assessment for a user, or None."""
    return (
        StressAssessment.query
        .filter_by(user_id=user_id)
        .order_by(StressAssessment.created_at.desc())
        .first()
    )


def get_history(user_id, limit=20):
    """Return stress assessment history for a user, newest first."""
    return (
        StressAssessment.query
        .filter_by(user_id=user_id)
        .order_by(StressAssessment.created_at.desc())
        .limit(limit)
        .all()
    )
