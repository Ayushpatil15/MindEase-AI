"""Tests for Task Chunking Activity (Atomicity, Duplicate Guard, Crisis Screening)."""

import pytest
from app.models.task import WellnessTask
from app.services import task_service
from tests.conftest import create_test_student


def test_chunking_success_atomic(app, client, db):
    """Test that all 3 micro-steps are created atomically in a single transaction."""
    task_service.clear_chunking_duplicate_cache()
    student = create_test_student(db, username='chunker1', email='chunker1@test.com')
    with client.session_transaction() as sess:
        sess['_user_id'] = str(student.id)
        sess['_fresh'] = True

    initial_count = WellnessTask.query.filter_by(user_id=student.id).count()

    response = client.post(
        '/api/activities/task_chunking/chunk',
        json={
            'big_task': 'Submit Biology Assignment',
            'micro_steps': [
                'Read chapter 4 summary',
                'Outline three main arguments',
                'Write conclusion paragraph',
            ]
        }
    )
    assert response.status_code == 201
    data = response.get_json()
    assert data['status'] == 'success'
    assert len(data['tasks']) == 3

    new_count = WellnessTask.query.filter_by(user_id=student.id).count()
    assert new_count == initial_count + 3
    created_tasks = WellnessTask.query.filter_by(user_id=student.id).order_by(WellnessTask.id.desc()).limit(3).all()
    titles = {t.title for t in created_tasks}
    assert 'Read chapter 4 summary' in titles
    assert 'Outline three main arguments' in titles
    assert 'Write conclusion paragraph' in titles
    for t in created_tasks:
        assert t.description == 'Part of: Submit Biology Assignment'
        assert t.completed is False


def test_chunking_validation_failure_creates_nothing(app, client, db):
    """Test atomicity: if any step fails validation (e.g. empty or >200 chars), create nothing."""
    task_service.clear_chunking_duplicate_cache()
    student = create_test_student(db, username='chunker2', email='chunker2@test.com')
    with client.session_transaction() as sess:
        sess['_user_id'] = str(student.id)
        sess['_fresh'] = True

    initial_count = WellnessTask.query.filter_by(user_id=student.id).count()

    # Step 2 is empty
    response = client.post(
        '/api/activities/task_chunking/chunk',
        json={
            'big_task': 'Math Homework',
            'micro_steps': [
                'Do problem 1',
                '   ',
                'Do problem 3',
            ]
        }
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data['status'] == 'error'
    assert 'Step 2 cannot be empty' in data['message']

    # Assert no tasks were created
    current_count = WellnessTask.query.filter_by(user_id=student.id).count()
    assert current_count == initial_count

    # Step 3 exceeds 200 chars
    response2 = client.post(
        '/api/activities/task_chunking/chunk',
        json={
            'big_task': 'Math Homework',
            'micro_steps': [
                'Do problem 1',
                'Do problem 2',
                'X' * 201,
            ]
        }
    )
    assert response2.status_code == 400
    assert 'exceeds the 200 character limit' in response2.get_json()['message']

    assert WellnessTask.query.filter_by(user_id=student.id).count() == initial_count


def test_chunking_duplicate_guard(app, client, db):
    """Test duplicate batch rejection from the same user within 30s."""
    task_service.clear_chunking_duplicate_cache()
    student = create_test_student(db, username='chunker3', email='chunker3@test.com')
    with client.session_transaction() as sess:
        sess['_user_id'] = str(student.id)
        sess['_fresh'] = True

    payload = {
        'big_task': 'Prepare Presentation',
        'micro_steps': [
            'Pick slide theme',
            'Write bullet points',
            'Practice timing once',
        ]
    }

    # First attempt: succeeds
    res1 = client.post('/api/activities/task_chunking/chunk', json=payload)
    assert res1.status_code == 201

    # Second attempt immediately: rejected with 409 duplicate error
    res2 = client.post('/api/activities/task_chunking/chunk', json=payload)
    assert res2.status_code == 409
    data2 = res2.get_json()
    assert data2['status'] == 'error'
    assert 'Duplicate batch detected' in data2['message']

    # After cache reset (simulating > 30s elapsed), succeeds
    task_service.clear_chunking_duplicate_cache()
    res3 = client.post('/api/activities/task_chunking/chunk', json=payload)
    assert res3.status_code == 201


def test_chunking_crisis_language_screening(app, client, db):
    """Test crisis language in task chunking triggers support card immediately without persisting or logging."""
    task_service.clear_chunking_duplicate_cache()
    student = create_test_student(db, username='chunker4', email='chunker4@test.com')
    with client.session_transaction() as sess:
        sess['_user_id'] = str(student.id)
        sess['_fresh'] = True

    initial_count = WellnessTask.query.filter_by(user_id=student.id).count()

    response = client.post(
        '/api/activities/task_chunking/chunk',
        json={
            'big_task': 'I want to end my life',
            'micro_steps': [
                'Step 1',
                'Step 2',
                'Step 3',
            ]
        }
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'crisis_detected'
    assert data['crisis_detected'] is True
    assert 'support_card' in data
    assert 'tele_manas' in data['support_card']
    assert data['support_card']['tele_manas']['phone'] == '14416'

    # Assert no task was created, no text was stored, and no DB column/flag was added
    current_count = WellnessTask.query.filter_by(user_id=student.id).count()
    assert current_count == initial_count
