"""Mood Check-In Service.

Implements the LOCKED mood specification:
- 5 moods: Very low, Low, Okay, Good, Very good.
- Valence mapping centralized in models/mood.py.
- 7-day trend uses ONLY actual data points (no gap-filling).
- Journal notes are private — never included in admin queries.

Full implementation in Phase 3.
"""

from datetime import datetime, timedelta, timezone

from app.extensions import db
from app.models import MoodEntry, MOOD_VALENCE_MAP, VALID_MOODS


def record_checkin(user_id, mood, journal_note=None):
    """Record a mood check-in.

    Args:
        user_id: The student's user ID.
        mood: One of VALID_MOODS.
        journal_note: Optional private journal text.

    Returns:
        The created MoodEntry.

    Raises:
        ValueError: If mood is not in VALID_MOODS.
    """
    if mood not in VALID_MOODS:
        raise ValueError(f'Invalid mood. Must be one of: {", ".join(VALID_MOODS)}')

    valence = MOOD_VALENCE_MAP[mood]

    entry = MoodEntry(
        user_id=user_id,
        mood=mood,
        valence=valence,
        journal_note=journal_note,
    )
    db.session.add(entry)
    db.session.commit()

    return entry


def get_7day_trend(user_id):
    """Return actual mood entries for the last 7 days.

    Returns ONLY real data points — no gap-filling for missing days.
    If only 3 days have data, returns only those 3 entries.

    Returns:
        List of dicts with keys: date, mood, valence (chronological order).
    """
    seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)

    entries = (
        MoodEntry.query
        .filter(
            MoodEntry.user_id == user_id,
            MoodEntry.created_at >= seven_days_ago,
        )
        .order_by(MoodEntry.created_at.asc())
        .all()
    )

    return [
        {
            'date': entry.created_at.strftime('%Y-%m-%d'),
            'mood': entry.mood,
            'valence': entry.valence,
        }
        for entry in entries
    ]


def get_latest(user_id):
    """Return the most recent mood entry for a user, or None."""
    return (
        MoodEntry.query
        .filter_by(user_id=user_id)
        .order_by(MoodEntry.created_at.desc())
        .first()
    )


def get_history(user_id, limit=30):
    """Return mood entry history for a user, newest first.

    Note: journal_note is included here because this is the student's
    own private history view. Admin queries must NEVER include journal_note.
    """
    return (
        MoodEntry.query
        .filter_by(user_id=user_id)
        .order_by(MoodEntry.created_at.desc())
        .limit(limit)
        .all()
    )
