"""Tests for Phase 6 Student Dashboard & Holistic Wellness Integration."""

import json
import pytest
from app.models.stress import StressAssessment
from app.models.mood import MoodEntry
from app.models.emotion import EmotionScan
from app.models.task import WellnessTask
from app.models.user_preferences import UserPreferences
from app.services import (
    dashboard_service,
    stress_service,
    mood_service,
    emotion_service,
    recommendation_service,
    task_service,
)
from tests.conftest import create_test_student, create_test_admin


# --- 1. Service Layer Tests ---

def test_dashboard_service_empty_user(app, db):
    """Test dashboard service for a brand-new user with zero records."""
    with app.app_context():
        user = create_test_student(db, username="dash_empty_user", email="dash_empty@test.com", password="password123")
        data = dashboard_service.get_student_dashboard_data(user.id)

        assert data is not None
        assert data['user']['username'] == user.username
        assert data['wellness_score']['status'] == 'insufficient_data'
        assert data['latest_stress'] is None
        assert data['latest_mood'] is None
        assert data['latest_emotion'] is None
        assert data['mood_trend_7d'] == []
        assert len(data['recommendations']) > 0  # Default welcome recommendation generated
        assert data['recommendations'][0]['title'] == 'Welcome to MindEase AI'
        assert data['tasks']['pending'] == []
        assert data['tasks']['completed_7d_count'] == 0


def test_dashboard_service_full_user(app, db):
    """Test dashboard service for a student with complete wellness records."""
    with app.app_context():
        user = create_test_student(db, username="dash_full_user", email="dash_full@test.com", password="password123")

        # 1. Log stress (Low stress score = 0)
        stress_service.save_assessment(user.id, [0, 0, 0, 4, 4, 0, 4, 4, 0, 0, 0])

        # 2. Log mood
        mood_service.record_checkin(user.id, "Good")

        # 3. Log emotion
        emotion_service.save_emotion_scan(user.id, "Happy", 0.95)

        # 4. Create preferences
        pref = UserPreferences(user_id=user.id, wellness_goals=["sleep", "study_productivity"])
        db.session.add(pref)

        # 5. Create wellness task
        task_service.create_task(user.id, "Hydration Goal")
        db.session.commit()

        data = dashboard_service.get_student_dashboard_data(user.id)

        assert data['wellness_score']['status'] == 'available'
        assert data['wellness_score']['score'] is not None
        assert data['latest_stress']['level'] == 'Low'
        assert data['latest_mood']['mood'] == 'Good'
        assert data['latest_emotion']['emotion'] == 'Happy'
        assert len(data['mood_trend_7d']) == 1
        assert len(data['tasks']['pending']) == 1
        assert data['tasks']['pending'][0]['title'] == "Hydration Goal"
        assert len(data['recommendations']) > 0


# --- 2. Deterministic Recommendation Engine Tests ---

def test_recommendations_high_stress_rule(app, db):
    """High stress assessment must trigger Box Breathing recommendation."""
    with app.app_context():
        user = create_test_student(db, username="high_stress_user", email="high_stress_rec@test.com", password="password123")
        # [4,4,4,0,0,4,0,0,4,4,4] yields 44/44 (High stress score)
        stress_service.save_assessment(user.id, [4, 4, 4, 0, 0, 4, 0, 0, 4, 4, 4])

        recs = recommendation_service.generate_recommendations(user.id)
        titles = [r.title for r in recs]

        assert "Box Breathing & Grounding Exercise" in titles


def test_recommendations_low_mood_rule(app, db):
    """Very low mood check-in must trigger Self-Compassion recommendation."""
    with app.app_context():
        user = create_test_student(db, username="low_mood_user", email="low_mood_rec@test.com", password="password123")
        mood_service.record_checkin(user.id, "Very low")

        recs = recommendation_service.generate_recommendations(user.id)
        titles = [r.title for r in recs]

        assert "Self-Compassion & Reflection" in titles


# --- 3. Authorization & IDOR Tests ---

def test_dashboard_requires_login(client):
    """Unauthenticated access to GET / must redirect to login."""
    res = client.get('/')
    assert res.status_code == 302
    assert '/auth/login' in res.location


def test_dashboard_api_requires_login(client):
    """Unauthenticated access to GET /api/dashboard must return 302 or 401."""
    res = client.get('/api/dashboard')
    assert res.status_code in (302, 401)


def test_dashboard_api_success_for_student(client, app, db):
    """Authenticated student can fetch GET /api/dashboard."""
    with app.app_context():
        user = create_test_student(db, username="api_student_user", email="api_student@test.com", password="password123")
        username = user.username

    # Login
    client.post('/auth/login', data={'email': 'api_student@test.com', 'password': 'password123'})

    res = client.get('/api/dashboard')
    assert res.status_code == 200
    json_data = res.get_json()
    assert json_data['status'] == 'success'
    assert json_data['dashboard']['user']['username'] == username


def test_task_create_and_complete_api(client, app, db):
    """Test task creation and completion via AJAX API endpoints."""
    with app.app_context():
        user = create_test_student(db, username="task_api_user", email="task_api@test.com", password="password123")

    client.post('/auth/login', data={'email': 'task_api@test.com', 'password': 'password123'})

    # Create task
    res_create = client.post('/api/tasks', json={'title': 'Read 10 pages', 'description': 'Mindfulness book'})
    assert res_create.status_code == 201
    task_id = res_create.get_json()['task']['id']

    # Complete task
    res_comp = client.post(f'/api/tasks/{task_id}/complete')
    assert res_comp.status_code == 200
    assert res_comp.get_json()['task']['completed'] is True


def test_task_complete_idor_protection(client, app, db):
    """User B cannot complete User A's task via API."""
    with app.app_context():
        user_a = create_test_student(db, username="user_a_task", email="user_a_task@test.com", password="password123")
        user_b = create_test_student(db, username="user_b_task", email="user_b_task@test.com", password="password123")
        task_a = task_service.create_task(user_a.id, "User A Private Task")
        task_a_id = task_a.id

    # Login as User B
    client.post('/auth/login', data={'email': 'user_b_task@test.com', 'password': 'password123'})

    # Try to complete User A's task
    res = client.post(f'/api/tasks/{task_a_id}/complete')
    assert res.status_code == 404
    assert res.get_json()['status'] == 'error'


def test_api_csrf_protection_validation(client, app, db):
    """Test CSRF enforcement on state-changing API endpoints when WTF_CSRF_ENABLED is True."""
    from flask_wtf.csrf import generate_csrf

    with app.app_context():
        user = create_test_student(db, username="csrf_test_user", email="csrf@test.com", password="password123")
        task = task_service.create_task(user.id, "CSRF Test Task")
        task_id = task.id

    client.post('/auth/login', data={'email': 'csrf@test.com', 'password': 'password123'})

    app.config['WTF_CSRF_ENABLED'] = True
    try:
        # 1. Missing CSRF token -> Rejected (400)
        res_no_csrf = client.post('/api/tasks', json={'title': 'No CSRF Task'})
        assert res_no_csrf.status_code == 400

        # 2. Invalid CSRF token -> Rejected (400)
        res_invalid_csrf = client.post('/api/tasks', json={'title': 'Bad CSRF Task'}, headers={'X-CSRFToken': 'invalid-token'})
        assert res_invalid_csrf.status_code == 400

        # 3. Complete task missing CSRF token -> Rejected (400)
        res_comp_no_csrf = client.post(f'/api/tasks/{task_id}/complete')
        assert res_comp_no_csrf.status_code == 400

        # 4. GET /api/dashboard is read-only and not blocked by CSRF (200)
        res_get = client.get('/api/dashboard')
        assert res_get.status_code == 200

    finally:
        app.config['WTF_CSRF_ENABLED'] = False



