"""Emotion Scan model.

Stores facial expression analysis results. No raw images are persisted.
The 5-class specification is locked — see ml/predictor.py for inference.
"""

from datetime import datetime, timezone

from app.extensions import db


# Locked 5-class set mapping (no Disgust, no Surprise)
# Explicit and centralized mapping so training and inference cannot accidentally disagree.
EMOTION_CLASS_MAPPING = {
    0: 'Angry',
    1: 'Fear',
    2: 'Happy',
    3: 'Sad',
    4: 'Neutral'
}

EMOTION_CLASSES = list(EMOTION_CLASS_MAPPING.values())


class EmotionScan(db.Model):
    """Result of a facial expression check-in (CNN inference only)."""

    __tablename__ = 'emotion_scans'
    __table_args__ = (
        db.Index('ix_emotion_user_date', 'user_id', 'created_at'),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
    )

    # Predicted emotion class — one of EMOTION_CLASSES
    emotion = db.Column(db.String(20), nullable=False)

    # Softmax confidence for the predicted class (0.0–1.0)
    confidence = db.Column(db.Float, nullable=False)

    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f'<EmotionScan user={self.user_id} emotion={self.emotion} confidence={self.confidence:.2f}>'
