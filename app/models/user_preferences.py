"""User Preferences model.

Stores optional onboarding data (wellness goals, preferred check-in time).
Uses JSON for goals so new wellness dimensions can be added without schema
changes — the onboarding UI just adds options to the list.

Initial goals: stress, study_productivity, mood, sleep, healthier_routines.
Future goals (not yet implemented): physical_activity, nutrition, habits, etc.
"""

from datetime import datetime, timezone

from app.extensions import db


# Canonical initial wellness goals available during onboarding
WELLNESS_GOALS = [
    'stress',
    'study_productivity',
    'mood',
    'sleep',
    'healthier_routines',
]


class UserPreferences(db.Model):
    """Optional preferences collected during onboarding."""

    __tablename__ = 'user_preferences'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        unique=True,
        index=True,
    )

    # Wellness goals selected during onboarding — JSON list of goal keys
    # e.g. ["stress", "mood", "sleep"]
    # Nullable: user may skip onboarding entirely
    wellness_goals = db.Column(db.JSON, nullable=True)

    # Optional preferred check-in/reminder time — "HH:MM" (24h format)
    # Nullable: user may not set a preference
    preferred_checkin_time = db.Column(db.String(5), nullable=True)

    # Optional student IANA timezone string — e.g. "Asia/Kolkata", "America/New_York", "UTC"
    # Nullable: existing users default safely to "UTC"
    timezone = db.Column(db.String(50), nullable=True, default='UTC')

    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime(timezone=True), nullable=True, onupdate=lambda: datetime.now(timezone.utc))

    # Relationship
    user = db.relationship('User', backref=db.backref('preferences', uselist=False))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self):
        goals = self.wellness_goals or []
        return f'<UserPreferences user={self.user_id} goals={goals}>'
