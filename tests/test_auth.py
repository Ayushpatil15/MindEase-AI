"""Phase 1 tests — authentication, UI states, CSRF, and authorization boundaries."""

import pytest
from flask import url_for
from unittest.mock import patch

from app.models.user import User
from app.models.user_preferences import UserPreferences
from tests.conftest import create_test_student, create_test_admin


def test_404_error_page(client):
    """Verify custom 404 template renders correctly."""
    response = client.get('/nonexistent_route')
    assert response.status_code == 404
    assert b'Page Not Found' in response.data


def test_500_error_page(app, client):
    """Verify custom 500 template renders."""
    with patch('app.routes.api.jsonify', side_effect=Exception('Simulated server error')):
        # We must disable propagate_exceptions in test client temporarily
        app.config['PROPAGATE_EXCEPTIONS'] = False
        response = client.get('/api/health')
        assert response.status_code == 500
        assert b'Internal Server Error' in response.data


class TestRegistration:
    def test_registration_success(self, client, db):
        response = client.post('/auth/register', data={
            'username': 'newstudent',
            'email': 'new@test.com',
            'password': 'securepassword123',
            'confirm_password': 'securepassword123'
        }, follow_redirects=True)
        
        assert response.status_code == 200
        # Should redirect to onboarding
        assert b'Welcome, newstudent' in response.data
        
        user = User.query.filter_by(username='newstudent').first()
        assert user is not None
        assert user.email == 'new@test.com'
        assert user.role == 'student'

    def test_registration_duplicate_username(self, client, db):
        create_test_student(db, username='existing')
        response = client.post('/auth/register', data={
            'username': 'existing',
            'email': 'other@test.com',
            'password': 'password123',
            'confirm_password': 'password123'
        })
        assert b'Username is already taken' in response.data

    def test_registration_duplicate_email(self, client, db):
        create_test_student(db, email='taken@test.com')
        response = client.post('/auth/register', data={
            'username': 'anotheruser',
            'email': 'taken@test.com',
            'password': 'password123',
            'confirm_password': 'password123'
        })
        assert b'Email is already registered' in response.data

    def test_registration_invalid_input(self, client):
        response = client.post('/auth/register', data={
            'username': 'ab',  # too short
            'email': 'not-an-email',
            'password': 'short',
            'confirm_password': 'not-matching'
        })
        assert b'Field must be between 3 and 64 characters long.' in response.data
        assert b'Invalid email address.' in response.data
        assert b'Field must be at least 8 characters long.' in response.data
        assert b'Passwords must match.' in response.data

    def test_password_hashing(self, db):
        user = create_test_student(db, password='mysecretpassword')
        assert user.password_hash != 'mysecretpassword'
        assert user.check_password('mysecretpassword')


class TestLoginLogout:
    def test_login_success(self, client, db):
        create_test_student(db, email='login@test.com', password='mypassword123')
        response = client.post('/auth/login', data={
            'email': 'login@test.com',
            'password': 'mypassword123'
        }, follow_redirects=True)
        assert b'Logged in successfully' in response.data
        # Standard redirect for student is dashboard
        assert b'Student Dashboard' in response.data

    def test_login_invalid(self, client, db):
        create_test_student(db, email='login@test.com', password='mypassword123')
        response = client.post('/auth/login', data={
            'email': 'login@test.com',
            'password': 'wrongpassword'
        })
        assert b'Invalid email or password' in response.data

    def test_login_redirect_valid_next(self, client, db):
        create_test_student(db, email='login_valid_next@test.com', password='mypassword123')
        response = client.post('/auth/login?next=/history', data={
            'email': 'login_valid_next@test.com',
            'password': 'mypassword123'
        }, follow_redirects=False)
        assert response.status_code == 302
        assert response.headers['Location'] == '/history'

    def test_login_redirect_open_redirect_double_slash(self, client, db):
        create_test_student(db, email='login_evil1@test.com', password='mypassword123')
        response = client.post('/auth/login?next=//evil.com', data={
            'email': 'login_evil1@test.com',
            'password': 'mypassword123'
        }, follow_redirects=False)
        assert response.status_code == 302
        assert response.headers['Location'] == '/'

    def test_login_redirect_open_redirect_backslash(self, client, db):
        create_test_student(db, email='login_evil2@test.com', password='mypassword123')
        response = client.post('/auth/login?next=/\\evil.com', data={
            'email': 'login_evil2@test.com',
            'password': 'mypassword123'
        }, follow_redirects=False)
        assert response.status_code == 302
        assert response.headers['Location'] == '/'

    def test_logout(self, client, db):
        # Login first
        create_test_student(db, email='logout@test.com', password='mypassword123')
        client.post('/auth/login', data={
            'email': 'logout@test.com',
            'password': 'mypassword123'
        })
        
        # Logout using POST
        response = client.post('/auth/logout', follow_redirects=True)
        assert b'You have been logged out' in response.data
        assert b'Welcome back' in response.data  # Login page
        
    def test_logout_get_method_not_allowed(self, client, db):
        create_test_student(db, email='logout2@test.com', password='pass')
        client.post('/auth/login', data={'email': 'logout2@test.com', 'password': 'pass'})
        response = client.get('/auth/logout')
        assert response.status_code == 405  # Method Not Allowed


class TestAuthorizationBoundaries:
    def test_student_cannot_access_admin(self, client, db):
        create_test_student(db, email='student@test.com', password='pass')
        client.post('/auth/login', data={'email': 'student@test.com', 'password': 'pass'})
        
        response = client.get('/admin/')
        assert response.status_code == 403

    def test_admin_cannot_access_student(self, client, db):
        create_test_admin(db, email='admin@test.com', password='pass')
        client.post('/auth/login', data={'email': 'admin@test.com', 'password': 'pass'})
        
        response = client.get('/')
        assert response.status_code == 403

    def test_unauthenticated_cannot_access_protected(self, client):
        response = client.get('/', follow_redirects=True)
        assert b'Welcome back' in response.data  # Redirected to login
        
        response = client.get('/admin/', follow_redirects=True)
        assert b'Welcome back' in response.data


class TestOnboarding:
    def test_onboarding_persistence(self, client, db):
        user = create_test_student(db, email='onboard@test.com', password='pass')
        client.post('/auth/login', data={'email': 'onboard@test.com', 'password': 'pass'})
        
        response = client.post('/auth/onboarding', data={
            'wellness_goals': ['stress', 'sleep'],
            'preferred_checkin_time': '09:00'
        }, follow_redirects=True)
        
        assert b'Preferences saved' in response.data
        
        prefs = UserPreferences.query.filter_by(user_id=user.id).first()
        assert prefs is not None
        assert set(prefs.wellness_goals) == {'stress', 'sleep'}
        assert prefs.preferred_checkin_time == '09:00'

    def test_optional_onboarding_fields(self, client, db):
        user = create_test_student(db, email='onboard2@test.com', password='pass')
        client.post('/auth/login', data={'email': 'onboard2@test.com', 'password': 'pass'})
        
        # Post completely empty form
        response = client.post('/auth/onboarding', data={}, follow_redirects=True)
        
        prefs = UserPreferences.query.filter_by(user_id=user.id).first()
        assert prefs is not None
        assert prefs.wellness_goals == []
        assert prefs.preferred_checkin_time is None
        
    def test_onboarding_update_existing(self, client, db):
        user = create_test_student(db, email='update@test.com', password='pass')
        client.post('/auth/login', data={'email': 'update@test.com', 'password': 'pass'})
        
        # First save
        client.post('/auth/onboarding', data={'wellness_goals': ['stress']})
        # Second save (update)
        response = client.post('/auth/onboarding', data={
            'wellness_goals': ['mood', 'sleep'],
            'preferred_checkin_time': '12:00'
        }, follow_redirects=True)
        
        assert b'Preferences updated successfully' in response.data
        assert UserPreferences.query.filter_by(user_id=user.id).count() == 1
        
        prefs = UserPreferences.query.filter_by(user_id=user.id).first()
        assert set(prefs.wellness_goals) == {'mood', 'sleep'}
        assert prefs.preferred_checkin_time == '12:00'
        
    def test_invalid_checkin_time(self, client, db):
        user = create_test_student(db, email='invalid_time@test.com', password='pass')
        client.post('/auth/login', data={'email': 'invalid_time@test.com', 'password': 'pass'})
        
        response = client.post('/auth/onboarding', data={
            'preferred_checkin_time': '25:00'  # Invalid hour
        })
        assert b'Must be a valid 24-hour time' in response.data
        
    def test_invalid_wellness_goal(self, client, db):
        user = create_test_student(db, email='invalid_goal@test.com', password='pass')
        client.post('/auth/login', data={'email': 'invalid_goal@test.com', 'password': 'pass'})
        
        response = client.post('/auth/onboarding', data={
            'wellness_goals': ['not_a_real_goal']
        })
        assert b'is not a valid choice' in response.data

    def test_admin_skips_onboarding(self, client, db):
        create_test_admin(db, email='admin_onboard@test.com', password='pass')
        client.post('/auth/login', data={'email': 'admin_onboard@test.com', 'password': 'pass'})
        
        response = client.get('/auth/onboarding', follow_redirects=True)
        # Should redirect to admin dashboard
        assert b'Admin dashboard' in response.data


class TestCSRF:
    def test_csrf_protection_enabled(self, client):
        # We need to test CSRF when it's actually enabled
        # The test app fixture disables it by default (WTF_CSRF_ENABLED = False)
        pass

def test_csrf_actually_protects(app, db):
    # Create a fresh client with CSRF enabled
    app.config['WTF_CSRF_ENABLED'] = True
    client = app.test_client()
    
    response = client.post('/auth/register', data={
        'username': 'csrf_test',
        'email': 'csrf@test.com',
        'password': 'password123',
        'confirm_password': 'password123'
    })
    # CSRF token missing -> 400 Bad Request
    assert response.status_code == 400
    assert b'The CSRF token is missing.' in response.data
    
    app.config['WTF_CSRF_ENABLED'] = False


def test_rate_limiting_login_blocks_brute_force(app, db):
    """Verify that enabling rate limiting returns 429 after exceeding limit."""
    from app.extensions import limiter
    app.config['RATELIMIT_ENABLED'] = True
    limiter.enabled = True
    limiter.reset()
    try:
        client = app.test_client()
        # Limit is 5/minute on POST /auth/login
        for _ in range(5):
            res = client.post('/auth/login', data={'email': 'rate@test.com', 'password': 'wrong'})
            assert res.status_code != 429
        # GET should not be rate limited
        res_get = client.get('/auth/login')
        assert res_get.status_code == 200
        # 6th attempt must trigger 429 Too Many Requests
        res = client.post('/auth/login', data={'email': 'rate@test.com', 'password': 'wrong'})
        assert res.status_code == 429
    finally:
        app.config['RATELIMIT_ENABLED'] = False
        limiter.enabled = False
        limiter.reset()
