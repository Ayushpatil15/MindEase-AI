"""Tests for the Phase 4 Wellness Score Calculator.

Covers exact formula integration, normalizations, missing data 
redistributions, and bounds.
"""

from datetime import datetime, timedelta, timezone
import pytest

from app.extensions import db
from app.models.stress import StressAssessment
from app.models.mood import MoodEntry
from app.services.wellness_calculator import calculate_wellness_score
from tests.conftest import create_test_student


class TestWellnessCalculatorNormalization:
    """Test exact mathematical formulas."""
    
    def test_insufficient_data(self, app, db):
        with app.app_context():
            user = create_test_student(db, email='w_empty@test.com', password='pass')
            res = calculate_wellness_score(user.id)
            assert res['status'] == 'insufficient_data'
            assert res['score'] is None

    def test_stress_only_normalization_and_weighting(self, app, db):
        """
        Stress = 44 (worst) -> 0 normalized. S=0, M=None.
        Consistency = 1/1 days -> 100 normalized.
        Weights: S=80%, C=20%
        Final: (0 * 0.8) + (100 * 0.2) = 20
        """
        with app.app_context():
            user = create_test_student(db, email='w_s_only_bad@test.com', password='pass')
            user.created_at = datetime.now(timezone.utc)
            db.session.commit()
            
            s = StressAssessment(user_id=user.id, score=44, level='High', responses=[0]*11)
            db.session.add(s)
            db.session.commit()
            
            res = calculate_wellness_score(user.id)
            assert res['status'] == 'available'
            assert res['score'] == 20

            # Test best stress
            s.score = 0
            db.session.commit()
            res2 = calculate_wellness_score(user.id)
            assert res2['score'] == 100

    def test_mood_only_normalization_and_weighting(self, app, db):
        """
        Mood = Very low (-2) -> 0 normalized. S=None, M=0.
        Consistency = 1/1 days -> 100 normalized.
        Weights: M=80%, C=20%
        Final: (0 * 0.8) + (100 * 0.2) = 20
        """
        with app.app_context():
            user = create_test_student(db, email='w_m_only_bad@test.com', password='pass')
            
            m = MoodEntry(user_id=user.id, mood='Very low', valence=-2)
            db.session.add(m)
            db.session.commit()
            
            res = calculate_wellness_score(user.id)
            assert res['score'] == 20
            
            m.mood = 'Very good'
            m.valence = 2
            db.session.commit()
            res2 = calculate_wellness_score(user.id)
            assert res2['score'] == 100

    def test_both_available(self, app, db):
        """
        S = 22 -> 50%
        M = 0 -> 50%
        C = 1/1 -> 100%
        Weights: S=40%, M=40%, C=20%
        Final: (50 * 0.4) + (50 * 0.4) + (100 * 0.2) = 20 + 20 + 20 = 60
        """
        with app.app_context():
            user = create_test_student(db, email='w_both@test.com', password='pass')
            
            s = StressAssessment(user_id=user.id, score=22, level='Moderate', responses=[0]*11)
            m = MoodEntry(user_id=user.id, mood='Okay', valence=0)
            db.session.add_all([s, m])
            db.session.commit()
            
            res = calculate_wellness_score(user.id)
            assert res['score'] == 60


class TestWellnessConsistency:
    def test_consistency_denominator_capping(self, app, db):
        with app.app_context():
            user = create_test_student(db, email='w_cons@test.com', password='pass')
            now = datetime.now(timezone.utc)
            
            user.created_at = now - timedelta(days=10)
            
            s = StressAssessment(user_id=user.id, score=22, level='Moderate', created_at=now, responses=[0]*11)
            db.session.add(s)
            db.session.commit()
            
            res = calculate_wellness_score(user.id)
            assert res['score'] == 43
            
            user.created_at = now - timedelta(days=1)
            db.session.commit()
            res2 = calculate_wellness_score(user.id)
            assert res2['score'] == 50

    def test_bounds_and_rounding(self, app, db):
        with app.app_context():
            user = create_test_student(db, email='w_bounds@test.com', password='pass')
            
            for i in range(10):
                db.session.add(MoodEntry(user_id=user.id, mood='Very good', valence=2, created_at=datetime.now(timezone.utc)))
            db.session.commit()
            
            res = calculate_wellness_score(user.id)
            assert res['score'] == 100

    def test_missing_mood_older_than_7_days(self, app, db):
        with app.app_context():
            user = create_test_student(db, email='w_old@test.com', password='pass')
            user.created_at = datetime.now(timezone.utc) - timedelta(days=10)
            
            m = MoodEntry(user_id=user.id, mood='Very good', valence=2, created_at=datetime.now(timezone.utc) - timedelta(days=10))
            db.session.add(m)
            
            s = StressAssessment(user_id=user.id, score=22, level='Moderate', responses=[0]*11)
            db.session.add(s)
            db.session.commit()
            
            res = calculate_wellness_score(user.id)
            assert res['score'] == 43
