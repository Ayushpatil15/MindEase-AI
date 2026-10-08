"""Test suite for Phase 9 — Personalized Wellness Plan & Daily Check-In Flow."""

from datetime import datetime, timedelta, timezone
import pytest

from app.models import User, MoodEntry, StressAssessment, EmotionScan, WellnessTask, UserPreferences
from app.services import (
    mood_service,
    stress_service,
    emotion_service,
    task_service,
)
from app.services.daily_plan_service import get_daily_plan
from tests.conftest import create_test_student


@pytest.fixture
def student_user(db):
    """Create and return a test student user within active test session."""
    return create_test_student(db, username='daily_student', email='daily@mindease.ai')


@pytest.fixture
def other_student_user(db):
    """Create and return a second test student user for IDOR testing."""
    return create_test_student(db, username='other_daily_student', email='other_daily@mindease.ai')


def test_daily_plan_new_user(app, db, student_user):
    """Test get_daily_plan for a brand new user with 0 check-ins."""
    plan = get_daily_plan(student_user.id)

    assert 'date' in plan
    assert plan['checkins']['mood_completed'] is False
    assert plan['checkins']['stress_completed'] is False
    assert plan['checkins']['emotion_completed'] is False
    assert plan['checkins']['all_completed'] is False
    assert plan['focus'] is not None
    assert plan['focus']['activity_id'] == 'grounding_54321'
    assert plan['focus']['is_active_task'] is False
    assert len(plan['next_actions']) >= 1


def test_daily_plan_checkin_completion_status(app, db, student_user):
    """Test check-in completion status detection for today UTC."""
    # 1. Record mood check-in today
    mood_service.record_checkin(student_user.id, mood='Good')

    plan = get_daily_plan(student_user.id)
    assert plan['checkins']['mood_completed'] is True
    assert plan['checkins']['stress_completed'] is False
    assert plan['checkins']['all_completed'] is False

    # 2. Complete stress assessment today
    stress_service.save_assessment(student_user.id, responses=[1]*11)

    # 3. Complete emotion scan today
    emotion_service.save_emotion_scan(student_user.id, emotion='Happy', confidence=0.90)

    plan_updated = get_daily_plan(student_user.id)
    assert plan_updated['checkins']['mood_completed'] is True
    assert plan_updated['checkins']['stress_completed'] is True
    assert plan_updated['checkins']['emotion_completed'] is True
    assert plan_updated['checkins']['all_completed'] is True


def test_daily_plan_focus_priority_active_task(app, db, student_user):
    """Priority 1: Existing active activity task takes precedence as primary focus."""
    # Start an activity task
    task, created = task_service.start_activity_task(student_user.id, 'pomodoro_focus')

    plan = get_daily_plan(student_user.id)
    assert plan['focus']['activity_id'] == 'pomodoro_focus'
    assert plan['focus']['is_active_task'] is True
    assert plan['focus']['task_id'] == task.id


def test_daily_plan_focus_priority_high_stress(app, db, student_user):
    """Priority 2: High stress assessment prioritizes Box Breathing."""
    # Reverse indices (3, 4, 6, 7) receive 0 (scored as 4-0=4), normal indices receive 4 -> Total score = 44 (High)
    high_responses = [4, 4, 4, 0, 0, 4, 0, 0, 4, 4, 4]
    stress_service.save_assessment(student_user.id, responses=high_responses)

    plan = get_daily_plan(student_user.id)
    assert plan['focus']['activity_id'] == 'box_breathing'
    assert plan['focus']['is_active_task'] is False


def test_daily_plan_focus_priority_low_mood(app, db, student_user):
    """Priority 3: Low mood prioritizes Self-Compassion exercise."""
    mood_service.record_checkin(student_user.id, mood='Very low')

    plan = get_daily_plan(student_user.id)
    assert plan['focus']['activity_id'] == 'self_compassion'
    assert plan['focus']['is_active_task'] is False


def test_daily_plan_focus_priority_facial_tension(app, db, student_user):
    """Priority 4: Facial expression tension prioritizes Stretch & Tension Release."""
    emotion_service.save_emotion_scan(student_user.id, emotion='Angry', confidence=0.85)

    plan = get_daily_plan(student_user.id)
    assert plan['focus']['activity_id'] == 'tension_release'
    assert plan['focus']['is_active_task'] is False


def test_daily_plan_reminder_window(app, db, student_user):
    """Test preferred check-in time reminder window detection."""
    now_hour = datetime.now(timezone.utc).hour
    pref_str = f"{now_hour:02d}:00"

    pref = UserPreferences(user_id=student_user.id, preferred_checkin_time=pref_str)
    db.session.add(pref)
    db.session.commit()

    plan = get_daily_plan(student_user.id)
    assert plan['reminder']['preferred_time'] == pref_str
    assert plan['reminder']['is_reminder_time'] is True
    assert plan['reminder']['message'] is not None


def test_daily_plan_api_endpoint(client, db, student_user):
    """Test GET /api/daily-plan returns 200 OK and normalized payload."""
    client.post('/auth/login', data={'email': student_user.email, 'password': 'testpass123'})

    response = client.get('/api/daily-plan')
    assert response.status_code == 200
    json_resp = response.get_json()

    assert json_resp['status'] == 'success'
    assert 'daily_plan' in json_resp
    dp = json_resp['daily_plan']
    assert 'checkins' in dp
    assert 'focus' in dp
    assert 'next_actions' in dp
    assert 'reminder' in dp
    assert 'progress' in dp


def test_daily_plan_api_privacy_and_idor(client, db, student_user, other_student_user):
    """Test authentication requirement, user isolation, and privacy leak protection."""
    # Unauthenticated request -> 302 or 401
    unauth_resp = client.get('/api/daily-plan')
    assert unauth_resp.status_code in (302, 401)

    # Populate records
    m = MoodEntry(user_id=student_user.id, mood='Good', valence=1, journal_note='SECRET_JOURNAL')
    s = StressAssessment(user_id=student_user.id, responses=[1]*11, score=12, level='Low', reflection='SECRET_REFLECTION')
    db.session.add_all([m, s])
    db.session.commit()

    # Log in as student_user
    client.post('/auth/login', data={'email': student_user.email, 'password': 'testpass123'})

    resp = client.get('/api/daily-plan')
    assert resp.status_code == 200
    raw_text = resp.get_data(as_text=True)

    # Privacy: secret notes must NOT be present
    assert 'SECRET_JOURNAL' not in raw_text
    assert 'SECRET_REFLECTION' not in raw_text
