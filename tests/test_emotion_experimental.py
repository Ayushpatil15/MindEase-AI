"""Tests for Phase 3 Item 3 — Experimental Facial Emotion Check-In.

Verifies:
1. EMOTION_MIN_CONFIDENCE defaults to 0.5 in configuration.
2. Low-confidence predictions (< 0.5) return 'Not sure' and are not persisted.
3. Predictions at or above threshold (>= 0.5) are persisted.
4. Saved emotion scans do not alter recommendations.
5. Saved emotion scans do not alter daily-plan priority focus.
6. Saved emotion scans do not alter the wellness score.
"""

from unittest.mock import patch
import pytest

from app.models.emotion import EmotionScan
from app.models.mood import MoodEntry
from app.models.stress import StressAssessment
from app.services import (
    emotion_service,
    recommendation_service,
    daily_plan_service,
    wellness_calculator,
    stress_service,
    mood_service,
)
from tests.conftest import create_test_student


def test_emotion_min_confidence_config_default(app):
    """Verify EMOTION_MIN_CONFIDENCE defaults to 0.5 in configuration."""
    assert app.config.get('EMOTION_MIN_CONFIDENCE') == 0.5


@patch('app.routes.emotion.predict')
def test_low_confidence_prediction_not_saved(mock_predict, client, db):
    """A low-confidence prediction (< EMOTION_MIN_CONFIDENCE) returns 'Not sure' and is not saved."""
    user = create_test_student(db, email='low_conf@test.com', password='pass')
    client.post('/auth/login', data={'email': 'low_conf@test.com', 'password': 'pass'})

    # Mock prediction with confidence 0.42 (below 0.5)
    mock_predict.return_value = {
        'status': 'success',
        'emotion': 'Sad',
        'confidence': 0.42,
    }

    dummy_payload = {'image': 'data:image/jpeg;base64,dGVzdA=='}
    resp = client.post('/emotion/api/predict', json=dummy_payload)

    assert resp.status_code == 200
    data = resp.get_json()
    assert data['status'] == 'low_confidence'
    assert data['emotion'] == 'Not sure'
    assert data['confidence'] == 0.42

    # Verify no record was saved to the database
    scans = EmotionScan.query.filter_by(user_id=user.id).all()
    assert len(scans) == 0


@patch('app.routes.emotion.predict')
def test_high_confidence_prediction_saved(mock_predict, client, db):
    """A prediction at or above EMOTION_MIN_CONFIDENCE (>= 0.5) is saved."""
    user = create_test_student(db, email='high_conf@test.com', password='pass')
    client.post('/auth/login', data={'email': 'high_conf@test.com', 'password': 'pass'})

    mock_predict.return_value = {
        'status': 'success',
        'emotion': 'Happy',
        'confidence': 0.88,
    }

    dummy_payload = {'image': 'data:image/jpeg;base64,dGVzdA=='}
    resp = client.post('/emotion/api/predict', json=dummy_payload)

    assert resp.status_code == 200
    data = resp.get_json()
    assert data['status'] == 'success'
    assert data['emotion'] == 'Happy'
    assert data['confidence'] == 0.88

    # Verify record was saved in database
    scans = EmotionScan.query.filter_by(user_id=user.id).all()
    assert len(scans) == 1
    assert scans[0].emotion == 'Happy'
    assert scans[0].confidence == 0.88


def test_saved_scan_does_not_change_recommendations(app, db):
    """A saved emotion scan does not alter recommendations."""
    with app.app_context():
        user = create_test_student(db, email='rec_test@test.com', password='pass')

        # Baseline recommendations for brand new user
        baseline_recs = recommendation_service.generate_recommendations(user.id)
        baseline_titles = [r.title for r in baseline_recs]
        assert 'Welcome to MindEase AI' in baseline_titles

        # Save an emotion scan (even a negative one like 'Angry' or 'Sad')
        emotion_service.save_emotion_scan(user.id, emotion='Angry', confidence=0.92)

        # Regenerate recommendations
        new_recs = recommendation_service.generate_recommendations(user.id)
        new_titles = [r.title for r in new_recs]

        # Recommendations must remain identical
        assert new_titles == baseline_titles
        assert 'Physical Stretch & Tension Release' not in new_titles


def test_saved_scan_does_not_change_daily_plan_priority(app, db):
    """A saved emotion scan does not alter daily-plan priority focus."""
    with app.app_context():
        user = create_test_student(db, email='plan_test@test.com', password='pass')

        # Daily plan before any check-in defaults to grounding exercise
        plan_before = daily_plan_service.get_daily_plan(user.id)
        assert plan_before['focus']['activity_id'] == 'grounding_54321'

        # Save emotion scan with 'Sad'
        emotion_service.save_emotion_scan(user.id, emotion='Sad', confidence=0.90)

        # Daily plan focus after scan must NOT change to tension_release
        plan_after = daily_plan_service.get_daily_plan(user.id)
        assert plan_after['focus']['activity_id'] == 'grounding_54321'
        assert plan_after['focus']['activity_id'] != 'tension_release'


def test_saved_scan_does_not_change_wellness_score(app, db):
    """A saved emotion scan does not alter the wellness score."""
    with app.app_context():
        user = create_test_student(db, email='wellness_test@test.com', password='pass')

        # Add mood check-in to provide sufficient data for wellness score
        mood_service.record_checkin(user.id, mood='Good')

        # Wellness score before scan
        score_before = wellness_calculator.calculate_wellness_score(user.id)
        assert score_before['status'] == 'available'
        assert score_before['score'] is not None

        # Save an emotion scan
        emotion_service.save_emotion_scan(user.id, emotion='Fear', confidence=0.85)

        # Wellness score after scan must be identical
        score_after = wellness_calculator.calculate_wellness_score(user.id)
        assert score_after['status'] == score_before['status']
        assert score_after['score'] == score_before['score']


@patch('app.routes.emotion.predict')
def test_low_confidence_scan_creates_no_db_record_and_not_on_history_page(mock_predict, client, db):
    """A low-confidence scan creates no DB record and does not appear on the history page."""
    import json
    import re
    user = create_test_student(db, username='low_conf_hist_user', email='low_conf_hist@test.com', password='pass')
    client.post('/auth/login', data={'email': user.email, 'password': 'pass'})

    # Mock low confidence prediction
    mock_predict.return_value = {
        'status': 'success',
        'emotion': 'Sad',
        'confidence': 0.35,
    }

    dummy_payload = {'image': 'data:image/jpeg;base64,dGVzdA=='}
    resp = client.post('/emotion/api/predict', json=dummy_payload)
    assert resp.status_code == 200
    assert resp.get_json()['status'] == 'low_confidence'

    # 1. DB assertion: 0 records
    assert EmotionScan.query.filter_by(user_id=user.id).count() == 0

    # 2. History page assertion: /history
    hist_resp = client.get('/history')
    assert hist_resp.status_code == 200
    html = hist_resp.get_data(as_text=True)

    # Extract embedded history JSON
    history_match = re.search(r'const history\s*=\s*(\{.*?\});', html, re.DOTALL)
    assert history_match is not None
    history_json = json.loads(history_match.group(1))

    # Emotion scans array is completely empty
    assert history_json['emotion']['count'] == 0
    assert len(history_json['emotion']['scans']) == 0


@patch('app.routes.emotion.predict')
def test_emotion_predict_endpoint_rate_limiting(mock_predict, app, db):
    """Verify POST /emotion/api/predict enforces rate limiting when enabled."""
    from app.extensions import limiter
    user = create_test_student(db, username='rate_emo_user', email='rate_emo@test.com', password='pass')

    mock_predict.return_value = {
        'status': 'success',
        'emotion': 'Happy',
        'confidence': 0.85,
    }

    app.config['RATELIMIT_ENABLED'] = True
    limiter.enabled = True
    limiter.reset()

    try:
        client = app.test_client()
        client.post('/auth/login', data={'email': user.email, 'password': 'pass'})

        dummy_payload = {'image': 'data:image/jpeg;base64,dGVzdA=='}
        # Rate limit is 15/minute on POST /emotion/api/predict
        for _ in range(15):
            res = client.post('/emotion/api/predict', json=dummy_payload)
            assert res.status_code == 200

        # 16th attempt triggers 429 Too Many Requests
        res_blocked = client.post('/emotion/api/predict', json=dummy_payload)
        assert res_blocked.status_code == 429
    finally:
        app.config['RATELIMIT_ENABLED'] = False
        limiter.enabled = False
        limiter.reset()
