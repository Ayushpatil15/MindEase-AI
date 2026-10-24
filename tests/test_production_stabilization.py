"""Production Stabilization Regression Tests — Phase 1.

Verifies:
1. Emotion prediction endpoint handles no-face-detected cleanly with HTTP 200 (no DB scan created).
2. Cascade missing returns HTTP 503 (service unavailable, not 400 client error).
3. CSRF token enforcement on POST /emotion/api/predict (400 when missing/invalid).
4. ProxyFix reverse-proxy handling sets is_secure, scheme, and client IP correctly.
5. Health and readiness endpoints (/health and /api/health) verify dependencies without leaking secrets.
"""

import os
from unittest.mock import patch, MagicMock
import pytest
import sqlalchemy as sa
from flask import request

from app import create_app
from app.config import Config, TestingConfig
from app.models.emotion import EmotionScan
from tests.conftest import create_test_student


def test_emotion_predict_no_face_detected_handled_cleanly(client, db):
    """When no face is detected in the frame, return HTTP 200 with no_face_detected status and do not save a record."""
    user = create_test_student(db, email='noface@test.com', password='pass')
    client.post('/auth/login', data={'email': 'noface@test.com', 'password': 'pass'})

    with patch('app.routes.emotion.predict') as mock_predict:
        mock_predict.return_value = {
            'status': 'no_face_detected',
            'message': 'No face detected in camera frame. Please adjust position and lighting.'
        }

        resp = client.post(
            '/emotion/api/predict',
            json={'image': 'data:image/jpeg;base64,dGVzdA=='}
        )

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'no_face_detected'
        assert 'No face detected' in data['message']

        # Ensure no scan was persisted
        scans = EmotionScan.query.filter_by(user_id=user.id).all()
        assert len(scans) == 0


def test_emotion_predict_cascade_missing_returns_503(client, db):
    """When the Haar cascade classifier is missing, return HTTP 503 rather than HTTP 400."""
    create_test_student(db, email='casc@test.com', password='pass')
    client.post('/auth/login', data={'email': 'casc@test.com', 'password': 'pass'})

    with patch('app.routes.emotion.predict') as mock_predict:
        mock_predict.return_value = {
            'status': 'cascade_missing',
            'message': 'Face detection classifier not found.'
        }

        resp = client.post(
            '/emotion/api/predict',
            json={'image': 'data:image/jpeg;base64,dGVzdA=='}
        )

        assert resp.status_code == 503
        data = resp.get_json()
        assert data['status'] == 'cascade_missing'


def test_emotion_predict_csrf_enforcement():
    """Verify CSRF protection rejects requests without CSRF token when CSRF is enabled."""
    class CsrfEnabledTestingConfig(TestingConfig):
        WTF_CSRF_ENABLED = True
        WTF_CSRF_CHECK_DEFAULT = True

    with patch.dict('app.config.config', {'csrf_test': CsrfEnabledTestingConfig}):
        csrf_app = create_app('csrf_test')
        with csrf_app.app_context():
            from app.extensions import db as csrf_db
            csrf_db.create_all()
            user = create_test_student(csrf_db, email='csrf@test.com', password='pass')

            c = csrf_app.test_client()
            # Login to establish authenticated session
            # Extract CSRF token from meta tag (exact method used by checkin.html)
            login_page = c.get('/auth/login')
            import re
            match = re.search(r'<meta name="csrf-token" content="([^"]+)"', login_page.get_data(as_text=True))
            assert match, "CSRF token meta tag should be present in page"
            token = match.group(1)

            c.post('/auth/login', data={'email': 'csrf@test.com', 'password': 'pass', 'csrf_token': token})

            # 1. POST without X-CSRFToken header must fail with HTTP 400
            resp_no_csrf = c.post(
                '/emotion/api/predict',
                json={'image': 'data:image/jpeg;base64,dGVzdA=='}
            )
            assert resp_no_csrf.status_code == 400

            # 2. POST with valid X-CSRFToken header passes CSRF verification
            # (Mock predict so it doesn't need actual ML)
            with patch('app.routes.emotion.predict') as mock_predict:
                mock_predict.return_value = {
                    'status': 'no_face_detected',
                    'message': 'No face'
                }
                resp_with_csrf = c.post(
                    '/emotion/api/predict',
                    headers={'X-CSRFToken': token},
                    json={'image': 'data:image/jpeg;base64,dGVzdA=='}
                )
                assert resp_with_csrf.status_code == 200

            csrf_db.drop_all()


def test_proxyfix_https_and_client_ip():
    """Verify ProxyFix middleware correctly interprets X-Forwarded-Proto and X-Forwarded-For."""
    class ProxyEnabledConfig(TestingConfig):
        BEHIND_PROXY = True

    with patch.dict('app.config.config', {'proxy_test': ProxyEnabledConfig}):
        proxy_app = create_app('proxy_test')

        @proxy_app.route('/test-proxy-context')
        def test_proxy():
            return {
                'scheme': request.scheme,
                'is_secure': request.is_secure,
                'remote_addr': request.remote_addr,
            }

        c = proxy_app.test_client()
        resp = c.get(
            '/test-proxy-context',
            headers={
                'X-Forwarded-Proto': 'https',
                'X-Forwarded-For': '198.51.100.42',
            }
        )

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['scheme'] == 'https'
        assert data['is_secure'] is True
        assert data['remote_addr'] == '198.51.100.42'


def test_health_and_readiness_endpoints(client):
    """Verify root /health and /api/health report readiness without leaking secrets."""
    for path in ('/health', '/api/health'):
        resp = client.get(path)
        assert resp.status_code == 200
        data = resp.get_json()

        assert data['status'] == 'ok'
        assert data['message'] == 'MindEase AI is running.'
        assert 'checks' in data
        assert data['checks']['database'] == 'connected'
        assert data['checks']['model'] == 'ready'
        assert data['checks']['cascade'] == 'ready'

        # Ensure no secrets, DB passwords, or internal keys are exposed
        body_text = resp.get_data(as_text=True)
        assert 'password' not in body_text.lower()
        assert 'secret' not in body_text.lower()
        assert 'sqlite' not in body_text.lower()
        assert 'postgresql' not in body_text.lower()


def test_health_endpoint_db_failure(client):
    """When the database is unreachable, health check returns HTTP 503 unhealthy."""
    with patch('app.extensions.db.session.execute', side_effect=Exception("DB connection timeout")):
        resp = client.get('/health')
        assert resp.status_code == 503
        data = resp.get_json()
        assert data['status'] == 'unhealthy'
        assert data['checks']['database'] == 'unavailable'


def test_health_endpoint_missing_optional_assets_returns_200_degraded(client):
    """Missing optional CNN assets degrade emotion scanning but keep website available (HTTP 200)."""
    with patch('os.path.isfile', side_effect=lambda p: False if 'tflite' in str(p) else True):
        resp = client.get('/health')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'degraded'
        assert data['checks']['model'] == 'missing'
        assert data['checks']['database'] == 'connected'

