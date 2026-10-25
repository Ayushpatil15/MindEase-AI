"""Models package — re-exports all models for convenient access.

Usage:
    from app.models import User, StressAssessment, MoodEntry, ...
"""

from app.models.user import User
from app.models.user_preferences import UserPreferences, WELLNESS_GOALS
from app.models.stress import StressAssessment
from app.models.mood import MoodEntry, MOOD_VALENCE_MAP, VALID_MOODS
from app.models.emotion import EmotionScan, EMOTION_CLASSES
from app.models.recommendation import Recommendation, RECOMMENDATION_CATEGORIES
from app.models.task import WellnessTask
from app.models.chat_memory import ChatMemory

__all__ = [
    'User',
    'UserPreferences',
    'WELLNESS_GOALS',
    'StressAssessment',
    'MoodEntry',
    'MOOD_VALENCE_MAP',
    'VALID_MOODS',
    'EmotionScan',
    'EMOTION_CLASSES',
    'Recommendation',
    'RECOMMENDATION_CATEGORIES',
    'WellnessTask',
    'ChatMemory',
]
