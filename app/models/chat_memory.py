"""Chat Memory model — Persistent non-sensitive student preferences and facts.

Enforces strict user isolation, compact text boundaries, and category classification.
Sensitive mental-health disclosures, assessments, and crisis notes are NEVER stored here.
"""

from datetime import datetime, timezone
from app.extensions import db


class ChatMemory(db.Model):
    """Stores non-sensitive facts, interests, or study preferences for personalized chat."""

    __tablename__ = 'chat_memories'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
    )
    memory_text = db.Column(db.String(256), nullable=False)
    category = db.Column(db.String(50), nullable=False, default='general')
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationship back to User
    user = db.relationship('User', backref=db.backref('chat_memories', lazy='dynamic', cascade='all, delete-orphan'))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self):
        return f'<ChatMemory user={self.user_id} category={self.category}>'
