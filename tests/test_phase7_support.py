"""Tests for Phase 7 — Support Center & Safety Disclaimers."""

import json
import os
import pytest
from tests.conftest import create_test_student


def test_support_resources_json_config(app):
    """Verify support_resources.json file exists and adheres to deployment-safe schema."""
    config_path = os.path.join(app.root_path, 'config', 'support_resources.json')
    assert os.path.exists(config_path)

    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    assert 'resources' in data
    resources = data['resources']
    assert len(resources) > 0

    for item in resources:
        assert 'name' in item
        assert 'category' in item
        assert 'description' in item
        assert 'contact' in item
        assert 'is_configured' in item
        assert item['category'] in ('counseling', 'wellness', 'professional', 'crisis')


def test_support_center_unauthenticated(client):
    """Anonymous request to /support should redirect to login."""
    res = client.get('/support')
    assert res.status_code == 302
    assert '/auth/login' in res.location


def test_support_center_authenticated(client, db):
    """Authenticated student can view the Support Center."""
    create_test_student(db, username="support_student", email="sup@test.com", password="testpass123")
    client.post('/auth/login', data={'email': 'sup@test.com', 'password': 'testpass123'})

    res = client.get('/support')
    assert res.status_code == 200
    assert b"Support Center" in res.data
    assert b"Non-Clinical Disclaimer" in res.data
    assert b"Administrator Setup Required" in res.data or b"Configure" in res.data
