"""Tests for Phase 7 — Guided Wellness Activities & Task Integration (Revised Lifecycle)."""

import pytest
from app.services.activity_service import get_all_activities, get_activity_by_id, get_activities_by_category
from app.services import task_service, recommendation_service, stress_service
from app.models.task import WellnessTask
from tests.conftest import create_test_student, create_test_admin


# --- 1. Activity Registry Unit Tests ---

def test_activity_registry_definitions():
    """Verify registry loads all 9 activities with complete metadata."""
    activities = get_all_activities()
    assert len(activities) == 9

    expected_categories = {'Mindfulness', 'Academic', 'Physical', 'Reflection'}
    found_categories = {a['category'] for a in activities}
    assert expected_categories == found_categories

    for act in activities:
        assert 'id' in act
        assert 'title' in act
        assert 'short_description' in act
        assert 'duration_minutes' in act
        assert 'steps' in act
        assert len(act['steps']) > 0
        assert 'has_timer' in act


def test_get_activity_by_id():
    """Verify lookup by activity ID."""
    box = get_activity_by_id('box_breathing')
    assert box is not None
    assert box['title'] == 'Box Breathing & Grounding Exercise'
    assert box['has_timer'] is True
    assert box['timer_seconds'] == 300

    invalid = get_activity_by_id('non_existent_activity')
    assert invalid is None


def test_get_activities_by_category():
    """Verify filtering by category."""
    mindfulness = get_activities_by_category('Mindfulness')
    assert len(mindfulness) == 2
    ids = {a['id'] for a in mindfulness}
    assert 'box_breathing' in ids
    assert 'grounding_54321' in ids


# --- 2. Activity Routes & Authorization Tests ---

def test_activities_catalog_route_unauthenticated(client):
    """Anonymous request to /activities should redirect to login."""
    res = client.get('/activities')
    assert res.status_code == 302
    assert '/auth/login' in res.location


def test_activities_catalog_route_authenticated(client, db):
    """Authenticated student can view the activities catalog."""
    user = create_test_student(db, username="act_student_1", email="act1@test.com", password="pass")
    client.post('/auth/login', data={'email': 'act1@test.com', 'password': 'pass'})

    res = client.get('/activities')
    assert res.status_code == 200
    assert b"Guided Wellness Activities" in res.data


def test_activity_detail_route_is_read_only(client, db):
    """GET /activities/<activity_id> must be read-only and NOT create any WellnessTask."""
    user = create_test_student(db, username="act_student_2", email="act2@test.com", password="pass")
    client.post('/auth/login', data={'email': 'act2@test.com', 'password': 'pass'})

    # 1. Access page
    res = client.get('/activities/box_breathing')
    assert res.status_code == 200

    # 2. Confirm ZERO tasks created
    task = WellnessTask.query.filter_by(user_id=user.id, activity_id='box_breathing').first()
    assert task is None


def test_activity_detail_route_invalid(client, db):
    """Invalid activity detail request returns 404."""
    user = create_test_student(db, username="act_student_3", email="act3@test.com", password="pass")
    client.post('/auth/login', data={'email': 'act3@test.com', 'password': 'pass'})

    res = client.get('/activities/invalid_slug')
    assert res.status_code == 404


# --- 3. Explicit Start & Complete API Tests ---

def test_start_activity_api_endpoint(client, db):
    """POST /api/activities/<id>/start creates an active task and avoids duplicates on retry."""
    user = create_test_student(db, username="api_start_user", email="api_start@test.com", password="pass")
    client.post('/auth/login', data={'email': 'api_start@test.com', 'password': 'pass'})

    # 1. First start call -> 201 Created
    res1 = client.post('/api/activities/pomodoro_focus/start', json={})
    assert res1.status_code == 201
    p1 = res1.get_json()
    assert p1['status'] == 'success'
    assert p1['created'] is True

    # 2. Repeated start call -> 200 OK (reuses active task)
    res2 = client.post('/api/activities/pomodoro_focus/start', json={})
    assert res2.status_code == 200
    p2 = res2.get_json()
    assert p2['status'] == 'success'
    assert p2['created'] is False
    assert p2['task']['id'] == p1['task']['id']


def test_complete_activity_api_without_start_fails(client, db):
    """POST /api/activities/<id>/complete without an active task returns 400 Bad Request."""
    user = create_test_student(db, username="api_no_start_user", email="api_nostart@test.com", password="pass")
    client.post('/auth/login', data={'email': 'api_nostart@test.com', 'password': 'pass'})

    res = client.post('/api/activities/screen_break/complete', json={})
    assert res.status_code == 400
    p = res.get_json()
    assert p['status'] == 'error'
    assert "No active task found" in p['message']


def test_start_and_complete_activity_flow(client, db):
    """Full flow: start activity -> complete activity."""
    user = create_test_student(db, username="flow_student", email="flow@test.com", password="pass")
    client.post('/auth/login', data={'email': 'flow@test.com', 'password': 'pass'})

    # Start
    res_start = client.post('/api/activities/movement_break/start', json={})
    assert res_start.status_code == 201

    # Complete
    res_comp = client.post('/api/activities/movement_break/complete', json={})
    assert res_comp.status_code == 200
    p_comp = res_comp.get_json()
    assert p_comp['status'] == 'success'
    assert p_comp['task']['completed'] is True


def test_cross_user_isolation(client, db):
    """User B cannot complete or affect User A's active tasks."""
    user_a = create_test_student(db, username="user_a", email="usera@test.com", password="passa")
    user_b = create_test_student(db, username="user_b", email="userb@test.com", password="passb")

    # User A logs in and starts task
    client.post('/auth/login', data={'email': 'usera@test.com', 'password': 'passa'})
    client.post('/api/activities/box_breathing/start', json={})
    client.post('/auth/logout')

    # User B logs in and attempts to complete box_breathing (User B has no active task)
    client.post('/auth/login', data={'email': 'userb@test.com', 'password': 'passb'})
    res_b = client.post('/api/activities/box_breathing/complete', json={})
    assert res_b.status_code == 400

    # User A's task remains incomplete
    task_a = WellnessTask.query.filter_by(user_id=user_a.id, activity_id='box_breathing').first()
    assert task_a is not None
    assert task_a.completed is False


def test_recommendation_to_activity_mapping(app, db):
    """Verify recommendation service attaches valid activity_id slugs."""
    with app.app_context():
        user = create_test_student(db, username="rec_act_user", email="rec_act@test.com")

        # Log High Stress (score 44: non-reverse=4, reverse=0) to trigger Box Breathing
        high_stress_responses = [4, 4, 4, 0, 0, 4, 0, 0, 4, 4, 4]
        stress_service.save_assessment(user.id, high_stress_responses)
        recs = recommendation_service.generate_recommendations(user.id)

        assert len(recs) > 0
        box_rec = recs[0]
        act_id = recommendation_service.get_activity_id_for_recommendation(box_rec)
        assert act_id == 'box_breathing'
        assert get_activity_by_id(act_id) is not None
