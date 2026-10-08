"""Phase 2 tests — Stress Assessment."""

import pytest
from app.models.stress import StressAssessment
from tests.conftest import create_test_student, create_test_admin

class TestStressAuthorization:
    def test_unauthenticated_access(self, client):
        response = client.get('/stress/', follow_redirects=True)
        assert b'Welcome back' in response.data  # Redirects to login

    def test_admin_access(self, client, db):
        create_test_admin(db, email='admin_stress@test.com', password='pass')
        client.post('/auth/login', data={'email': 'admin_stress@test.com', 'password': 'pass'})
        response = client.get('/stress/')
        assert response.status_code == 403


class TestStressValidation:
    def test_valid_submission(self, client, db):
        user = create_test_student(db, email='stress_student@test.com', password='pass')
        client.post('/auth/login', data={'email': 'stress_student@test.com', 'password': 'pass'})
        
        # Q4, Q5, Q7, Q8 are reverse scored.
        # Indices: 3, 4, 6, 7
        data = {
            'q1': 0, 'q2': 0, 'q3': 0, 'q4': 4, # q4 is reverse -> 0
            'q5': 4, 'q6': 0, 'q7': 4, 'q8': 4, # q5,q7,q8 reverse -> 0
            'q9': 0, 'q10': 0, 'q11': 0,
            'reflection': 'I feel great.'
        }
        response = client.post('/stress/', data=data, follow_redirects=True)
        assert response.status_code == 200
        assert b'Stress assessment submitted successfully' in response.data
        
        # Verify result in DB
        assessment = StressAssessment.query.filter_by(user_id=user.id).first()
        assert assessment is not None
        assert assessment.score == 0
        assert assessment.level == 'Low'
        assert assessment.reflection == 'I feel great.'

    def test_missing_responses_rejected(self, client, db):
        create_test_student(db, email='stress_student2@test.com', password='pass')
        client.post('/auth/login', data={'email': 'stress_student2@test.com', 'password': 'pass'})
        
        # Missing q11
        data = {
            'q1': 0, 'q2': 0, 'q3': 0, 'q4': 0, 
            'q5': 0, 'q6': 0, 'q7': 0, 'q8': 0, 
            'q9': 0, 'q10': 0
        }
        response = client.post('/stress/', data=data)
        assert b'This field is required.' in response.data

    def test_out_of_range_responses_rejected(self, client, db):
        create_test_student(db, email='stress_student3@test.com', password='pass')
        client.post('/auth/login', data={'email': 'stress_student3@test.com', 'password': 'pass'})
        
        # q1 is out of range
        data = {
            'q1': 5, 'q2': 0, 'q3': 0, 'q4': 0, 
            'q5': 0, 'q6': 0, 'q7': 0, 'q8': 0, 
            'q9': 0, 'q10': 0, 'q11': 0
        }
        response = client.post('/stress/', data=data)
        # WTForms coerce=int with RadioField and choices 0-4 will flag it as invalid choice
        assert b'Not a valid choice' in response.data

    def test_malformed_responses_rejected(self, client, db):
        create_test_student(db, email='stress_student4@test.com', password='pass')
        client.post('/auth/login', data={'email': 'stress_student4@test.com', 'password': 'pass'})
        
        data = {
            'q1': 'invalid', 'q2': 0, 'q3': 0, 'q4': 0, 
            'q5': 0, 'q6': 0, 'q7': 0, 'q8': 0, 
            'q9': 0, 'q10': 0, 'q11': 0
        }
        response = client.post('/stress/', data=data)
        assert b'Not a valid choice' in response.data


class TestStressOwnership:
    def test_cannot_view_others_assessment(self, client, db):
        user1 = create_test_student(db, username='owner', email='owner@test.com', password='pass')
        user2 = create_test_student(db, username='snooper', email='snooper@test.com', password='pass')
        
        # User 1 takes assessment
        client.post('/auth/login', data={'email': 'owner@test.com', 'password': 'pass'})
        data = {f'q{i}': 2 for i in range(1, 12)}
        client.post('/stress/', data=data)
        client.post('/auth/logout')
        
        assessment = StressAssessment.query.filter_by(user_id=user1.id).first()
        
        # User 2 tries to view User 1's assessment
        client.post('/auth/login', data={'email': 'snooper@test.com', 'password': 'pass'})
        response = client.get(f'/stress/result/{assessment.id}')
        assert response.status_code == 403

    def test_can_view_own_assessment(self, client, db):
        user = create_test_student(db, email='viewer@test.com', password='pass')
        client.post('/auth/login', data={'email': 'viewer@test.com', 'password': 'pass'})
        
        data = {f'q{i}': 2 for i in range(1, 12)}
        client.post('/stress/', data=data)
        
        assessment = StressAssessment.query.filter_by(user_id=user.id).first()
        
        response = client.get(f'/stress/result/{assessment.id}')
        assert response.status_code == 200
        assert b'Assessment Result' in response.data


class TestStressHistory:
    def test_history_shows_own_assessments(self, client, db):
        user = create_test_student(db, email='history@test.com', password='pass')
        client.post('/auth/login', data={'email': 'history@test.com', 'password': 'pass'})
        
        data = {f'q{i}': 2 for i in range(1, 12)}
        client.post('/stress/', data=data)
        client.post('/stress/', data=data)
        
        response = client.get('/stress/history')
        assert response.status_code == 200
        assert response.data.count(b'View Details') == 2
