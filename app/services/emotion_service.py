"""Emotion Service — Persistence layer for Facial Expression Inference."""

from app.extensions import db
from app.models.emotion import EmotionScan, EMOTION_CLASSES


def save_emotion_scan(user_id, emotion, confidence):
    """Securely save a facial expression estimate.
    
    Args:
        user_id (int): The explicit user ID to bind the record to.
        emotion (str): The exact 5-class string.
        confidence (float): The softmax confidence (0.0 to 1.0).
        
    Returns:
        EmotionScan: The persisted record.
        
    Raises:
        ValueError: If constraints are violated.
    """
    if emotion not in EMOTION_CLASSES:
        raise ValueError(f"Invalid emotion class: {emotion}")
        
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(f"Invalid confidence range: {confidence}")
        
    scan = EmotionScan(
        user_id=user_id,
        emotion=emotion,
        confidence=confidence
    )
    
    db.session.add(scan)
    db.session.commit()
    
    return scan


def get_latest(user_id):
    """Return the most recent EmotionScan for a user, or None."""
    return (
        EmotionScan.query
        .filter_by(user_id=user_id)
        .order_by(EmotionScan.created_at.desc())
        .first()
    )


def get_history(user_id, limit=20):
    """Return emotion scan history for a user, newest first."""
    return (
        EmotionScan.query
        .filter_by(user_id=user_id)
        .order_by(EmotionScan.created_at.desc())
        .limit(limit)
        .all()
    )

