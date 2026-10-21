"""Tests for Phase 2 review leftovers (Phase 3 Step 0).

Verifies:
1. APP_ENV / FLASK_ENV precedence and error handling on invalid configurations.
2. Real create_app('production') startup failure without SECRET_KEY and success with it.
3. Rendering of both history pages, parsing of embedded JSON, and ISO 8601 date validation.
4. Model smoke test loading configured TFLite model and inferencing on dummy frame.
"""

import os
import ast
import json
import re
from datetime import datetime, date, timezone
from unittest.mock import patch
import numpy as np
import pytest

from app import create_app
from app.config import Config
from app.ml.predictor import EmotionPredictor
from app.models.emotion import EMOTION_CLASSES
from app.models import MoodEntry, StressAssessment, EmotionScan, WellnessTask
from tests.conftest import create_test_student


class TestAppEnvPrecedence:
    """Tests for APP_ENV / FLASK_ENV environment variable precedence and validation."""

    def test_run_py_debug_not_hardcoded_true(self):
        """Verify run.py does not hard-code debug=True and retrieves debug setting dynamically."""
        run_path = os.path.join(os.path.dirname(__file__), '..', 'run.py')
        with open(run_path, 'r', encoding='utf-8') as f:
            source = f.read()

        assert 'debug=True' not in source
        assert 'debug = True' not in source

        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == 'debug':
                        if isinstance(keyword.value, ast.Constant):
                            assert keyword.value.value is not True

    def test_app_env_only(self):
        """When only APP_ENV is set, create_app() uses APP_ENV."""
        with patch.dict('os.environ', {'APP_ENV': 'testing'}, clear=True):
            app = create_app()
            assert app.config['TESTING'] is True

    def test_flask_env_only(self):
        """When only FLASK_ENV is set, create_app() uses FLASK_ENV."""
        with patch.dict('os.environ', {'FLASK_ENV': 'testing'}, clear=True):
            app = create_app()
            assert app.config['TESTING'] is True

    def test_app_env_precedence_over_flask_env(self):
        """APP_ENV takes precedence over FLASK_ENV when both are set."""
        with patch.dict('os.environ', {'APP_ENV': 'testing', 'FLASK_ENV': 'development'}, clear=True):
            app = create_app()
            assert app.config['TESTING'] is True

    def test_neither_set_defaults_to_development(self):
        """When neither is set, create_app() defaults to development."""
        with patch.dict('os.environ', {}, clear=True):
            app = create_app()
            assert app.config['DEBUG'] is True
            assert app.config['TESTING'] is False

    def test_invalid_config_name_raises_clear_error(self):
        """An invalid config name raises a clear ValueError."""
        with pytest.raises(ValueError, match="Invalid configuration 'nonexistent'"):
            create_app('nonexistent')

        with patch.dict('os.environ', {'APP_ENV': 'invalid_env'}, clear=True):
            with pytest.raises(ValueError, match="Invalid configuration 'invalid_env'"):
                create_app()

    def test_production_create_app_fails_without_secret_key(self):
        """Real create_app('production') startup fails if SECRET_KEY is missing."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(RuntimeError, match="SECRET_KEY environment variable is required in production"):
                create_app('production')

    def test_production_create_app_succeeds_with_secret_key(self):
        """Real create_app('production') startup succeeds when SECRET_KEY is provided."""
        with patch.dict('os.environ', {'SECRET_KEY': 'real-prod-secret-key-12345'}):
            app = create_app('production')
            assert app.config['DEBUG'] is False
            assert app.config['SECRET_KEY'] == 'real-prod-secret-key-12345'


class TestHistoryPagesEmbeddedJsonAndIsoDates:
    """Test rendering of both history pages and verification of embedded JSON and ISO dates."""

    def test_history_pages_embedded_json_and_iso_dates(self, client, db):
        """Render /mood/history and /history with sample data, verify JSON parses, and dates are ISO."""
        student = create_test_student(db, username='history_iso_user', email='iso@mindease.ai')
        client.post('/auth/login', data={'email': student.email, 'password': 'testpass123'})

        # Seed sample data across all dimensions
        now = datetime.now(timezone.utc)
        m = MoodEntry(user_id=student.id, mood='Good', valence=1, journal_note='Feeling alright')
        s = StressAssessment(user_id=student.id, responses=[2]*11, score=22, level='Moderate', reflection='Exam stress')
        e = EmotionScan(user_id=student.id, emotion='Happy', confidence=0.88)
        t = WellnessTask(user_id=student.id, title='Deep breathing', activity_id='breathing_1', completed=True, completed_at=now)
        db.session.add_all([m, s, e, t])
        db.session.commit()

        # 1. Render /mood/history
        resp_mood = client.get('/mood/history')
        assert resp_mood.status_code == 200
        html_mood = resp_mood.get_data(as_text=True)

        # Extract embedded JSON: const trendData = [...];
        trend_match = re.search(r'const trendData\s*=\s*(\[.*?\]);', html_mood, re.DOTALL)
        assert trend_match is not None, "Failed to find trendData embedded JSON in mood/history.html"
        trend_json = json.loads(trend_match.group(1))
        assert isinstance(trend_json, list)
        assert len(trend_json) >= 1

        for entry in trend_json:
            assert 'date' in entry
            # Check ISO date (YYYY-MM-DD)
            parsed_date = date.fromisoformat(entry['date'])
            assert parsed_date is not None
            assert 'valence' in entry
            assert 'mood' in entry

        # 2. Render /history (student history)
        resp_student = client.get('/history')
        assert resp_student.status_code == 200
        html_student = resp_student.get_data(as_text=True)

        # Extract embedded JSON: const history = {...};
        history_match = re.search(r'const history\s*=\s*(\{.*?\});', html_student, re.DOTALL)
        assert history_match is not None, "Failed to find history embedded JSON in student/history.html"
        history_json = json.loads(history_match.group(1))
        assert isinstance(history_json, dict)

        # Verify time window dates are ISO
        timeframe = history_json['timeframe']
        assert date.fromisoformat(timeframe['start_date']) is not None
        assert date.fromisoformat(timeframe['end_date']) is not None

        # Verify mood trend dates and created_at timestamps are ISO
        for item in history_json['mood']['trend']:
            assert date.fromisoformat(item['date']) is not None
            # datetime.fromisoformat handles ISO 8601 with timezone
            assert datetime.fromisoformat(item['created_at']) is not None

        # Verify stress assessment dates and created_at timestamps are ISO
        for item in history_json['stress']['assessments']:
            assert date.fromisoformat(item['date']) is not None
            assert datetime.fromisoformat(item['created_at']) is not None

        # Verify emotion scans dates and created_at timestamps are ISO
        for item in history_json['emotion']['scans']:
            assert date.fromisoformat(item['date']) is not None
            assert datetime.fromisoformat(item['created_at']) is not None

        # Verify activity completions completed_at timestamp is ISO
        for item in history_json['activities']['completions']:
            if item['completed_at']:
                assert datetime.fromisoformat(item['completed_at']) is not None


class TestModelSmokeTest:
    """Smoke test for production TFLite model loading and inference."""

    def test_tflite_model_smoke_test(self):
        """Load configured TFLite model and run inference on dummy 48x48 grayscale frame."""
        model_path = Config.TFLITE_MODEL_PATH
        predictor = EmotionPredictor(model_path=model_path, num_threads=1)
        assert predictor.tflite_loaded is True
        assert predictor.backend is not None

        # Dummy 48x48 grayscale frame, batch size 1
        dummy_tensor = np.zeros((1, 48, 48, 1), dtype=np.float32)
        result = predictor.predict(dummy_tensor)

        assert 'emotion' in result
        assert result['emotion'] in EMOTION_CLASSES
        assert 'confidence' in result
        assert 0.0 <= result['confidence'] <= 1.0
        assert 'probabilities' in result
        assert len(result['probabilities']) == len(EMOTION_CLASSES)
        assert abs(sum(result['probabilities']) - 1.0) < 1e-4
