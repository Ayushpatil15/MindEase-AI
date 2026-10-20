"""Phase 0 tests — verify scaffolding, app factory, models, and CLI."""

import pytest
from unittest.mock import patch

from app import create_app
from app.config import ProductionConfig, DevelopmentConfig
from app.extensions import db as _db
from app.models import (
    User, UserPreferences, StressAssessment, MoodEntry,
    EmotionScan, Recommendation, WellnessTask,
    MOOD_VALENCE_MAP, VALID_MOODS, EMOTION_CLASSES,
    RECOMMENDATION_CATEGORIES, WELLNESS_GOALS,
)
from app.services.stress_service import score_assessment, REVERSE_SCORED_INDICES
from tests.conftest import create_test_student, create_test_admin


class TestAppFactory:
    """Verify the Flask application factory."""

    def test_create_app_returns_flask_instance(self, app):
        assert app is not None
        assert app.name == 'app'

    def test_testing_config(self, app):
        assert app.config['TESTING'] is True
        assert app.config['WTF_CSRF_ENABLED'] is False
        assert 'memory' in app.config['SQLALCHEMY_DATABASE_URI']

    def test_blueprints_registered(self, app):
        assert 'auth' in app.blueprints
        assert 'student' in app.blueprints
        assert 'admin' in app.blueprints
        assert 'api' in app.blueprints

    def test_health_endpoint(self, client):
        response = client.get('/api/health')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'ok'

    def test_production_config_missing_secret_key_raises(self):
        from flask import Flask
        with patch.dict('os.environ', {}, clear=True):
            test_app = Flask('dummy')
            with pytest.raises(RuntimeError, match="SECRET_KEY environment variable is required in production"):
                test_app.config.from_object(ProductionConfig)

    def test_production_config_with_secret_key_succeeds(self):
        from flask import Flask
        with patch.dict('os.environ', {'SECRET_KEY': 'prod-secret-test-key'}):
            test_app = Flask('dummy')
            test_app.config.from_object(ProductionConfig)
            assert test_app.config['SECRET_KEY'] == 'prod-secret-test-key'

    def test_development_config_fallback_secret_key(self):
        from flask import Flask
        with patch.dict('os.environ', {}, clear=True):
            test_app = Flask('dummy')
            test_app.config.from_object(DevelopmentConfig)
            assert test_app.config['SECRET_KEY'] == 'dev-secret-key-change-in-production'


class TestUserModel:
    """Verify User model and password hashing."""

    def test_create_student(self, db):
        user = create_test_student(db)
        assert user.id is not None
        assert user.role == 'student'
        assert user.is_student is True
        assert user.is_admin is False

    def test_create_admin(self, db):
        user = create_test_admin(db)
        assert user.id is not None
        assert user.role == 'admin'
        assert user.is_admin is True
        assert user.is_student is False

    def test_password_hashing(self, db):
        user = create_test_student(db)
        assert user.password_hash != 'testpass123'
        assert user.check_password('testpass123') is True
        assert user.check_password('wrongpassword') is False

    def test_no_plaintext_password(self, db):
        user = create_test_student(db)
        assert 'testpass123' not in user.password_hash

    def test_unique_username(self, db):
        create_test_student(db, username='unique1', email='a@test.com')
        with pytest.raises(Exception):
            create_test_student(db, username='unique1', email='b@test.com')

    def test_unique_email(self, db):
        create_test_student(db, username='user1', email='same@test.com')
        with pytest.raises(Exception):
            create_test_student(db, username='user2', email='same@test.com')


class TestUserPreferencesModel:
    """Verify onboarding preferences model."""

    def test_create_preferences(self, db):
        user = create_test_student(db)
        prefs = UserPreferences(
            user_id=user.id,
            wellness_goals=['stress', 'mood', 'sleep'],
            preferred_checkin_time='09:00',
        )
        db.session.add(prefs)
        db.session.commit()
        assert prefs.id is not None
        assert prefs.wellness_goals == ['stress', 'mood', 'sleep']
        assert prefs.preferred_checkin_time == '09:00'

    def test_optional_preferences(self, db):
        user = create_test_student(db)
        prefs = UserPreferences(user_id=user.id)
        db.session.add(prefs)
        db.session.commit()
        assert prefs.wellness_goals is None
        assert prefs.preferred_checkin_time is None

    def test_user_relationship(self, db):
        user = create_test_student(db)
        prefs = UserPreferences(
            user_id=user.id,
            wellness_goals=['stress'],
        )
        db.session.add(prefs)
        db.session.commit()
        assert user.preferences is not None
        assert user.preferences.wellness_goals == ['stress']


class TestWellnessGoalsConstant:
    """Verify the canonical wellness goals list."""

    def test_initial_goals(self):
        assert 'stress' in WELLNESS_GOALS
        assert 'study_productivity' in WELLNESS_GOALS
        assert 'mood' in WELLNESS_GOALS
        assert 'sleep' in WELLNESS_GOALS
        assert 'healthier_routines' in WELLNESS_GOALS
        assert len(WELLNESS_GOALS) == 5


class TestMoodConstants:
    """Verify locked mood specification."""

    def test_exactly_five_moods(self):
        assert len(VALID_MOODS) == 5

    def test_all_moods_present(self):
        expected = {'Very good', 'Good', 'Okay', 'Low', 'Very low'}
        assert set(VALID_MOODS) == expected

    def test_valence_range(self):
        for mood, valence in MOOD_VALENCE_MAP.items():
            assert -2 <= valence <= 2, f'{mood} has invalid valence {valence}'

    def test_valence_mapping(self):
        assert MOOD_VALENCE_MAP['Very good'] == 2
        assert MOOD_VALENCE_MAP['Good'] == 1
        assert MOOD_VALENCE_MAP['Okay'] == 0
        assert MOOD_VALENCE_MAP['Low'] == -1
        assert MOOD_VALENCE_MAP['Very low'] == -2


class TestEmotionConstants:
    """Verify locked emotion specification."""

    def test_exactly_five_classes(self):
        assert len(EMOTION_CLASSES) == 5

    def test_correct_classes(self):
        expected = {'Angry', 'Fear', 'Happy', 'Sad', 'Neutral'}
        assert set(EMOTION_CLASSES) == expected

    def test_no_disgust_or_surprise(self):
        assert 'Disgust' not in EMOTION_CLASSES
        assert 'Surprise' not in EMOTION_CLASSES


class TestRecommendationConstants:
    """Verify recommendation categories."""

    def test_categories(self):
        expected = {'MINDFULNESS', 'ACADEMIC', 'PHYSICAL', 'SOCIAL', 'CLINICAL_SUPPORT'}
        assert set(RECOMMENDATION_CATEGORIES) == expected


class TestStressScoring:
    """Verify the LOCKED stress scoring logic."""

    def test_all_zeros(self):
        result = score_assessment([0] * 11)
        assert result['score'] == 16  # 4 reverse items × 4 = 16
        assert result['level'] == 'Moderate'

    def test_all_fours(self):
        result = score_assessment([4] * 11)
        assert result['score'] == 28  # 7 normal × 4 = 28, 4 reverse × 0 = 0
        assert result['level'] == 'Moderate'

    def test_low_stress(self):
        # All reverse items = 4 (reverse to 0), all normal = 0 → score = 0
        responses = [0] * 11
        for i in REVERSE_SCORED_INDICES:
            responses[i] = 4
        result = score_assessment(responses)
        assert result['score'] == 0
        assert result['level'] == 'Low'

    def test_high_stress(self):
        # All normal items = 4, all reverse items = 0 (reverse to 4) → score = 44
        responses = [4] * 11
        for i in REVERSE_SCORED_INDICES:
            responses[i] = 0
        result = score_assessment(responses)
        assert result['score'] == 44
        assert result['level'] == 'High'

    def test_threshold_boundary_low_moderate(self):
        # Exactly 14 → Low
        # Need: normal items sum + reverse items sum = 14
        # 7 normal + 4 reverse(4-raw) = 14
        # Set all to 0 → score = 16 (too high)
        # Manually compute: need exactly 14
        # 7 normals at 2 each = 14, 4 reverse at 4 each (4-4=0) = 0 → total = 14
        responses = [2] * 11
        for i in REVERSE_SCORED_INDICES:
            responses[i] = 4
        result = score_assessment(responses)
        assert result['score'] == 14
        assert result['level'] == 'Low'

    def test_threshold_boundary_moderate_start(self):
        # Score of 15 → Moderate
        responses = [2] * 11
        for i in REVERSE_SCORED_INDICES:
            responses[i] = 4
        # Current = 14, bump one normal +1
        responses[0] = 3  # now 15
        result = score_assessment(responses)
        assert result['score'] == 15
        assert result['level'] == 'Moderate'

    def test_threshold_boundary_moderate_high(self):
        # Score of 28 → Moderate
        responses = [4] * 11
        for i in REVERSE_SCORED_INDICES:
            responses[i] = 0  # reverse to 4
        # All 4s, reverse to 4 → total 44, need 28
        # 7 normal × 4 = 28, reverse × 0 = 0 → need reverse items at 4
        # wait, let me recalculate
        # score = sum_normal + sum_reversed
        # max normal = 7*4 = 28, max reverse = 4*4 = 16
        # For exactly 28: normal all 4 (=28), reverse all 4 (4-4=0) → 28
        responses = [4] * 11
        for i in REVERSE_SCORED_INDICES:
            responses[i] = 4
        result = score_assessment(responses)
        assert result['score'] == 28
        assert result['level'] == 'Moderate'

    def test_threshold_boundary_high_start(self):
        # Score of 29 → High
        responses = [4] * 11
        for i in REVERSE_SCORED_INDICES:
            responses[i] = 4
        # Currently 28, need +1: set one reverse item to 3 (4-3=1 instead of 0)
        first_reverse = min(REVERSE_SCORED_INDICES)
        responses[first_reverse] = 3
        result = score_assessment(responses)
        assert result['score'] == 29
        assert result['level'] == 'High'

    def test_exactly_11_questions_required(self):
        with pytest.raises(ValueError, match='11'):
            score_assessment([0] * 10)
        with pytest.raises(ValueError, match='11'):
            score_assessment([0] * 12)

    def test_valid_range_enforcement(self):
        with pytest.raises(ValueError):
            score_assessment([5] + [0] * 10)
        with pytest.raises(ValueError):
            score_assessment([-1] + [0] * 10)

    def test_reverse_scored_indices(self):
        # Q4, Q5, Q7, Q8 → 0-indexed: 3, 4, 6, 7
        assert REVERSE_SCORED_INDICES == {3, 4, 6, 7}

    def test_deterministic(self):
        responses = [1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1]
        r1 = score_assessment(responses)
        r2 = score_assessment(responses)
        assert r1 == r2


class TestModelsExist:
    """Verify all models can be instantiated and persisted."""

    def test_stress_assessment(self, db):
        user = create_test_student(db)
        sa = StressAssessment(
            user_id=user.id,
            responses=[0] * 11,
            score=16,
            level='Moderate',
        )
        db.session.add(sa)
        db.session.commit()
        assert sa.id is not None
        assert sa.reflection is None

    def test_mood_entry(self, db):
        user = create_test_student(db)
        me = MoodEntry(user_id=user.id, mood='Happy', valence=1)
        db.session.add(me)
        db.session.commit()
        assert me.id is not None
        assert me.journal_note is None

    def test_emotion_scan(self, db):
        user = create_test_student(db)
        es = EmotionScan(user_id=user.id, emotion='Happy', confidence=0.92)
        db.session.add(es)
        db.session.commit()
        assert es.id is not None

    def test_recommendation(self, db):
        user = create_test_student(db)
        rec = Recommendation(
            user_id=user.id,
            category='MINDFULNESS',
            title='Box Breathing',
            description='Try 4-4-4-4 breathing.',
            reason='Your stress is elevated.',
        )
        db.session.add(rec)
        db.session.commit()
        assert rec.id is not None

    def test_wellness_task(self, db):
        user = create_test_student(db)
        task = WellnessTask(
            user_id=user.id,
            title='10-minute walk',
        )
        db.session.add(task)
        db.session.commit()
        assert task.id is not None
        assert task.completed is False
        assert task.completed_at is None

    def test_task_mark_complete(self, db):
        user = create_test_student(db)
        task = WellnessTask(user_id=user.id, title='Breathing exercise')
        db.session.add(task)
        db.session.commit()
        task.mark_complete()
        db.session.commit()
        assert task.completed is True
        assert task.completed_at is not None


class TestCreateAdminCLI:
    """Verify the create-admin CLI command."""

    def test_create_admin_command(self, runner):
        result = runner.invoke(
            args=['create-admin', '--username', 'cliadmin', '--email', 'cli@test.com', '--password', 'securepass123'],
        )
        assert 'created successfully' in result.output
        from app.models import User
        admin = User.query.filter_by(username='cliadmin').first()
        assert admin is not None
        assert admin.role == 'admin'
        assert admin.check_password('securepass123')

    def test_duplicate_username_rejected(self, runner):
        runner.invoke(
            args=['create-admin', '--username', 'dupeuser', '--email', 'a@test.com', '--password', 'securepass123'],
        )
        result = runner.invoke(
            args=['create-admin', '--username', 'dupeuser', '--email', 'b@test.com', '--password', 'securepass123'],
        )
        assert 'already exists' in result.output or result.exit_code != 0

    def test_duplicate_email_rejected(self, runner):
        runner.invoke(
            args=['create-admin', '--username', 'admin1', '--email', 'same@test.com', '--password', 'securepass123'],
        )
        result = runner.invoke(
            args=['create-admin', '--username', 'admin2', '--email', 'same@test.com', '--password', 'securepass123'],
        )
        assert 'already registered' in result.output or result.exit_code != 0

    def test_short_password_rejected(self, runner):
        result = runner.invoke(
            args=['create-admin', '--username', 'weakadmin', '--email', 'w@test.com', '--password', 'short'],
        )
        assert result.exit_code != 0
