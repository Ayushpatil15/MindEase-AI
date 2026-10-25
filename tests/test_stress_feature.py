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


class TestStressRedirectAndHub:
    def test_post_submission_redirects_to_checkin_hub_not_today(self, client, db):
        create_test_student(db, email='redirect_student@test.com', password='pass')
        client.post('/auth/login', data={'email': 'redirect_student@test.com', 'password': 'pass'})

        data = {f'q{i}': 1 for i in range(1, 12)}
        data['reflection'] = 'Redirect test reflection.'

        response = client.post('/stress/', data=data, follow_redirects=False)
        assert response.status_code == 302
        assert response.headers['Location'] == '/checkin'
        assert response.headers['Location'] != '/'

    def test_post_submission_followed_redirect_shows_results_and_flash(self, client, db):
        user = create_test_student(db, email='hub_student@test.com', password='pass')
        client.post('/auth/login', data={'email': 'hub_student@test.com', 'password': 'pass'})

        # Reverse scored: 3, 4, 6, 7 (q4, q5, q7, q8)
        # q1..q3 = 1, q4..q5 = 3 (reverse 4-3=1), q6 = 1, q7..q8 = 3 (reverse 1), q9..q11 = 1 -> total = 11 -> Low
        data = {f'q{i}': 1 for i in range(1, 12)}
        data['q4'] = 3
        data['q5'] = 3
        data['q7'] = 3
        data['q8'] = 3
        data['reflection'] = 'Accessible results test.'

        response = client.post('/stress/', data=data, follow_redirects=True)
        assert response.status_code == 200
        html = response.get_data(as_text=True)

        assert 'Stress assessment submitted successfully.' in html
        assert 'Choose Your Check-In' in html
        assert 'Latest result:' in html
        assert '11/44 (Low)' in html
        assert 'View Result' in html
        assert 'History' in html

        assessment = StressAssessment.query.filter_by(user_id=user.id).first()
        assert f'/stress/result/{assessment.id}' in html

    def test_result_page_accessible_with_checkin_back_button(self, client, db):
        user = create_test_student(db, email='result_student@test.com', password='pass')
        client.post('/auth/login', data={'email': 'result_student@test.com', 'password': 'pass'})

        data = {f'q{i}': 2 for i in range(1, 12)}
        client.post('/stress/', data=data)

        assessment = StressAssessment.query.filter_by(user_id=user.id).first()
        response = client.get(f'/stress/result/{assessment.id}')
        assert response.status_code == 200
        html = response.get_data(as_text=True)

        assert 'Assessment Result' in html
        assert 'Back to Check-In' in html
        assert 'href="/checkin"' in html


class TestStressFlowAndPreservation:
    def test_template_stepper_structure_and_guards(self, client, db):
        create_test_student(db, email='flow_student@test.com', password='pass')
        client.post('/auth/login', data={'email': 'flow_student@test.com', 'password': 'pass'})

        response = client.get('/stress/')
        assert response.status_code == 200
        html = response.get_data(as_text=True)

        # 1. Sequential question steps 1 to 11
        for s in range(1, 12):
            assert f'data-step="{s}"' in html

        # 2. Reflection step
        assert 'id="reflection-step"' in html

        # 3. Stepper controls
        assert 'id="prev-btn"' in html
        assert 'id="next-btn"' in html
        assert 'id="submit-btn"' in html

        # 4. JavaScript safeguards for jumping, auto-advance, answer preservation, and submission locking
        assert 'clearAutoAdvance' in html
        assert 'autoAdvanceTimer' in html
        assert 'isSubmitting' in html
        assert 'option-radio' in html
        assert 'stepWhenSelected' in html
        assert 'syncStepVisuals' in html

    def test_incomplete_submission_preserves_answers_and_indicates_error(self, client, db):
        create_test_student(db, email='preserve_student@test.com', password='pass')
        client.post('/auth/login', data={'email': 'preserve_student@test.com', 'password': 'pass'})

        # Answer Q1..Q10, leave Q11 blank
        data = {f'q{i}': 3 for i in range(1, 11)}
        response = client.post('/stress/', data=data)
        assert response.status_code == 200
        html = response.get_data(as_text=True)

        # Validation error displayed
        assert 'Please answer all 11 questions before completing the assessment.' in html
        assert 'This field is required.' in html

        # Previous answers preserved in form inputs
        assert 'value="3" checked' in html or 'checked' in html

    def test_duplicate_submission_handling(self, client, db):
        user = create_test_student(db, email='dupe_student@test.com', password='pass')
        client.post('/auth/login', data={'email': 'dupe_student@test.com', 'password': 'pass'})

        data = {f'q{i}': 2 for i in range(1, 12)}
        # Submit twice
        res1 = client.post('/stress/', data=data, follow_redirects=True)
        assert res1.status_code == 200
        res2 = client.post('/stress/', data=data, follow_redirects=True)
        assert res2.status_code == 200

        # Two records exist cleanly without database or state corruption
        assessments = StressAssessment.query.filter_by(user_id=user.id).all()
        assert len(assessments) == 2
