"""Wellness Task model.

Personal to-do items for wellness activities.
"""

from datetime import datetime, timezone

from app.extensions import db


class WellnessTask(db.Model):
    """A personal wellness task owned by a student."""

    __tablename__ = 'wellness_tasks'
    __table_args__ = (
        db.Index('ix_task_user_completed', 'user_id', 'completed'),
        db.Index('ix_task_user_activity', 'user_id', 'activity_id'),
        db.Index(
            'uix_task_user_activity_active',
            'user_id',
            'activity_id',
            unique=True,
            sqlite_where=db.text('completed = 0 AND activity_id IS NOT NULL'),
            postgresql_where=db.text('completed = false AND activity_id IS NOT NULL'),
        ),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
    )

    activity_id = db.Column(db.String(50), nullable=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    completed = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    completed_at = db.Column(db.DateTime, nullable=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def mark_complete(self):
        """Mark the task as completed with a timestamp."""
        self.completed = True
        self.completed_at = datetime.now(timezone.utc)

    def __repr__(self):
        status = 'done' if self.completed else 'pending'
        return f'<WellnessTask user={self.user_id} title={self.title!r} activity={self.activity_id!r} {status}>'

