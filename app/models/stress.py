"""Stress Assessment model.

Stores the 11-question PSS-derived assessment results.
Scoring specification is locked — see stress_service.py for the scoring logic.
"""

from datetime import datetime, timezone

from app.extensions import db


class StressAssessment(db.Model):
    """A completed stress assessment for a student."""

    __tablename__ = 'stress_assessments'
    __table_args__ = (
        db.Index('ix_stress_user_date', 'user_id', 'created_at'),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
    )

    # Raw responses: list of exactly 11 integers, each 0–4
    responses = db.Column(db.JSON, nullable=False)

    # Computed score (0–44) and categorical level
    score = db.Column(db.Integer, nullable=False)
    level = db.Column(db.String(20), nullable=False)  # "Low", "Moderate", "High"

    # Optional private reflection text — NEVER exposed to admin analytics
    reflection = db.Column(db.Text, nullable=True)

    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f'<StressAssessment user={self.user_id} score={self.score} level={self.level}>'
