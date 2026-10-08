"""Test suite for Phase 8 — Wellness History, Insights & Progress Tracking."""

from datetime import datetime, timedelta, timezone
import pytest

from app.models import MoodEntry, StressAssessment, EmotionScan, WellnessTask
from app.services import (
    mood_service,
    stress_service,
    emotion_service,
    task_service,
)
from app.services.history_service import (
    get_wellness_history_data,
    generate_deterministic_insights,
    parse_timeframe,
)
from tests.conftest import create_test_student


@pytest.fixture
def student_user(db):
    """Create and return a test student user within active test session."""
    return create_test_student(db, username='history_student', email='history@mindease.ai')


@pytest.fixture
def other_student_user(db):
    """Create and return a second test student user for IDOR testing within active test session."""
    return create_test_student(db, username='other_student', email='other@mindease.ai')


def test_parse_timeframe():
    """Test parse_timeframe helper function."""
    assert parse_timeframe('7d') == ('7d', 7)
    assert parse_timeframe('30d') == ('30d', 30)
    assert parse_timeframe('90d') == ('90d', 90)
    # Invalid fallback
    assert parse_timeframe('invalid') == ('7d', 7)
    assert parse_timeframe(None) == ('7d', 7)


def test_history_service_empty_state(app, db, student_user):
    """Test get_wellness_history_data for a new user with zero records."""
    data = get_wellness_history_data(student_user.id, timeframe='7d')
    
    assert data['timeframe']['range'] == '7d'
    assert data['timeframe']['days'] == 7
    assert data['mood']['count'] == 0
    assert data['mood']['average_valence'] is None
    assert data['mood']['trend'] == []
    assert data['stress']['count'] == 0
    assert data['stress']['assessments'] == []
    assert data['emotion']['count'] == 0
    assert data['emotion']['distribution'] == {}
    assert data['activities']['completed_count'] == 0
    assert data['consistency']['active_days'] == 0
    assert data['consistency']['percentage'] == 0.0
    assert len(data['insights']) > 0
    assert "No check-ins recorded" in data['insights'][0]


def test_history_service_single_record(app, db, student_user):
    """Test get_wellness_history_data with exactly 1 mood check-in."""
    mood_service.record_checkin(student_user.id, mood='Good', journal_note='Felt nice')

    data = get_wellness_history_data(student_user.id, timeframe='7d')

    assert data['mood']['count'] == 1
    assert data['mood']['average_valence'] == 1.0
    assert len(data['mood']['trend']) == 1
    assert data['mood']['trend'][0]['mood'] == 'Good'
    assert data['consistency']['active_days'] == 1

    # Check insight avoids false trend claims
    insights_text = " ".join(data['insights'])
    assert "1 mood check-in recorded" in insights_text or "1 of the last 7 days" in insights_text
    assert "predominantly" not in insights_text


def test_history_service_timeframe_boundaries(app, db, student_user):
    """Test filtering records strictly by window boundaries (7d vs 30d vs 90d)."""
    now = datetime.now(timezone.utc)

    # Record 1: Today (2 hours ago)
    m1 = MoodEntry(user_id=student_user.id, mood='Good', valence=1, created_at=now - timedelta(hours=2))
    # Record 2: 5 days ago
    m2 = MoodEntry(user_id=student_user.id, mood='Okay', valence=0, created_at=now - timedelta(days=5))
    # Record 3: 15 days ago
    m3 = MoodEntry(user_id=student_user.id, mood='Very good', valence=2, created_at=now - timedelta(days=15))
    # Record 4: 45 days ago
    m4 = MoodEntry(user_id=student_user.id, mood='Low', valence=-1, created_at=now - timedelta(days=45))

    db.session.add_all([m1, m2, m3, m4])
    db.session.commit()

    # 7d window -> m1 and m2
    data_7d = get_wellness_history_data(student_user.id, timeframe='7d')
    assert data_7d['mood']['count'] == 2

    # 30d window -> m1, m2, m3
    data_30d = get_wellness_history_data(student_user.id, timeframe='30d')
    assert data_30d['mood']['count'] == 3

    # 90d window -> m1, m2, m3, m4
    data_90d = get_wellness_history_data(student_user.id, timeframe='90d')
    assert data_90d['mood']['count'] == 4


def test_history_service_deterministic_insights(app, db, student_user):
    """Test deterministic insight generation rules and safety boundaries."""
    now = datetime.now(timezone.utc)

    # 1. Add 5 mood check-ins (4 positive) across 5 different days
    for i, m_name in enumerate(['Very good', 'Good', 'Good', 'Good', 'Okay']):
        m = MoodEntry(
            user_id=student_user.id,
            mood=m_name,
            valence=1 if m_name == 'Good' else (2 if m_name == 'Very good' else 0),
            created_at=now - timedelta(days=i)
        )
        db.session.add(m)

    # 2. Add 2 stress assessments (score 25 then score 15 -> improvement)
    s1 = StressAssessment(
        user_id=student_user.id, responses=[2]*11, score=25, level='Moderate',
        created_at=now - timedelta(days=4)
    )
    s2 = StressAssessment(
        user_id=student_user.id, responses=[1]*11, score=15, level='Moderate',
        created_at=now - timedelta(days=1)
    )
    db.session.add_all([s1, s2])

    # 3. Add 2 completed wellness tasks
    t1 = WellnessTask(
        user_id=student_user.id, title='Box Breathing', activity_id='box_breathing',
        completed=True, completed_at=now - timedelta(days=2)
    )
    t2 = WellnessTask(
        user_id=student_user.id, title='Quick Walk', activity_id='quick_walk',
        completed=True, completed_at=now - timedelta(days=1)
    )
    db.session.add_all([t1, t2])

    # 4. Add 3 emotion scans
    e1 = EmotionScan(user_id=student_user.id, emotion='Happy', confidence=0.85, created_at=now - timedelta(days=3))
    e2 = EmotionScan(user_id=student_user.id, emotion='Happy', confidence=0.90, created_at=now - timedelta(days=2))
    e3 = EmotionScan(user_id=student_user.id, emotion='Neutral', confidence=0.75, created_at=now - timedelta(days=1))
    db.session.add_all([e1, e2, e3])

    db.session.commit()

    data = get_wellness_history_data(student_user.id, timeframe='7d')
    insights = data['insights']

    insights_text = " ".join(insights)
    assert "High check-in consistency!" in insights_text
    assert "predominantly positive" in insights_text
    assert "showed an improvement" in insights_text
    assert "completed 2 wellness activity sessions" in insights_text
    assert "most frequent detection was 'Happy'" in insights_text

    # Verify strict non-clinical boundary (no medical terms)
    for term in ['depressed', 'anxious', 'disorder', 'cured', 'diagnosed', 'treatment']:
        assert term not in insights_text.lower()


def test_history_api_endpoint(client, db, student_user):
    """Test GET /api/history endpoint returns normalized JSON payload."""
    client.post('/auth/login', data={'email': student_user.email, 'password': 'testpass123'})

    response = client.get('/api/history?range=7d')
    assert response.status_code == 200
    json_resp = response.get_json()

    assert json_resp['status'] == 'success'
    assert 'history' in json_resp
    h = json_resp['history']
    assert h['timeframe']['range'] == '7d'
    assert 'mood' in h
    assert 'stress' in h
    assert 'emotion' in h
    assert 'activities' in h
    assert 'consistency' in h
    assert 'insights' in h


def test_history_api_privacy_protection(client, db, student_user):
    """Test private journal notes and reflection text are excluded from /api/history JSON."""
    m = MoodEntry(
        user_id=student_user.id, mood='Good', valence=1,
        journal_note='SECRET_PRIVATE_JOURNAL_NOTE'
    )
    s = StressAssessment(
        user_id=student_user.id, responses=[1]*11, score=12, level='Low',
        reflection='SECRET_PRIVATE_STRESS_REFLECTION'
    )
    db.session.add_all([m, s])
    db.session.commit()

    client.post('/auth/login', data={'email': student_user.email, 'password': 'testpass123'})

    response = client.get('/api/history')
    assert response.status_code == 200
    raw_text = response.get_data(as_text=True)

    # Assert secret text is NEVER present in JSON response
    assert 'SECRET_PRIVATE_JOURNAL_NOTE' not in raw_text
    assert 'SECRET_PRIVATE_STRESS_REFLECTION' not in raw_text


def test_history_api_authorization_and_idor(client, db, student_user, other_student_user):
    """Test authentication requirements and user-scoped IDOR protection."""
    # 1. Unauthenticated request -> 401 or redirect
    resp = client.get('/api/history')
    assert resp.status_code in (401, 302)

    # 2. Populate records for student_user and other_student_user
    m1 = MoodEntry(user_id=student_user.id, mood='Good', valence=1)
    m2 = MoodEntry(user_id=other_student_user.id, mood='Very low', valence=-2)
    db.session.add_all([m1, m2])
    db.session.commit()

    # 3. Log in as student_user
    client.post('/auth/login', data={'email': student_user.email, 'password': 'testpass123'})

    resp = client.get('/api/history')
    assert resp.status_code == 200
    h = resp.get_json()['history']

    # Must contain student_user's mood ('Good') and NOT other_student_user's mood ('Very low')
    assert h['mood']['count'] == 1
    assert h['mood']['trend'][0]['mood'] == 'Good'


def test_history_web_route(client, db, student_user):
    """Test GET /history web page renders HTML successfully."""
    client.post('/auth/login', data={'email': student_user.email, 'password': 'testpass123'})

    response = client.get('/history?range=30d')
    assert response.status_code == 200
    html = response.get_data(as_text=True)

    assert "Wellness History & Insights" in html
    assert "30 Days" in html
    assert "Key Progress Observations" in html
