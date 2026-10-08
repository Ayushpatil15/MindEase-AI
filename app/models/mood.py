"""Mood Entry model.

Stores daily mood check-ins with valence and optional private journal notes.
The mood and valence mapping is locked — see mood_service.py for the canonical map.
"""

from datetime import datetime, timezone

from app.extensions import db


# Canonical mood → valence mapping (centralized, documented)
MOOD_VALENCE_MAP = {
    'Very good': 2,
    'Good': 1,
    'Okay': 0,
    'Low': -1,
    'Very low': -2,
}

VALID_MOODS = list(MOOD_VALENCE_MAP.keys())


class MoodEntry(db.Model):
    """A single mood check-in for a student."""

    __tablename__ = 'mood_entries'
    __table_args__ = (
        db.Index('ix_mood_user_date', 'user_id', 'created_at'),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
    )

    # One of the 5 valid moods
    mood = db.Column(db.String(20), nullable=False)

    # Numeric valence (-2 to +2), stored for efficient trend queries
    valence = db.Column(db.Integer, nullable=False)

    # Optional private journal note — NEVER exposed to admin analytics
    journal_note = db.Column(db.Text, nullable=True)

    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f'<MoodEntry user={self.user_id} mood={self.mood} valence={self.valence}>'
