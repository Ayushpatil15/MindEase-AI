"""Tests for Phase 7 / Phase 3 — Support Center & Safety Disclaimers."""

import json
import os
import pytest
from tests.conftest import create_test_student


def test_support_resources_json_config(app):
    """Verify support_resources.json file exists and adheres to deployment-safe schema."""
    content_path = os.path.join(app.root_path, 'content', 'support_resources.json')
    assert os.path.exists(content_path)

    with open(content_path, 'r', encoding='utf-8') as f:
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


def test_support_center_authenticated_hides_unconfigured(client, db):
    """Authenticated student views Support Center with unconfigured items hidden."""
    create_test_student(db, username="support_student", email="sup@test.com", password="testpass123")
    client.post('/auth/login', data={'email': 'sup@test.com', 'password': 'testpass123'})

    res = client.get('/support')
    assert res.status_code == 200
    html = res.get_data(as_text=True)

    # Tele-MANAS and 112 are always shown
    assert "Tele-MANAS" in html
    assert "14416" in html
    assert "112" in html
    assert "Non-Clinical Disclaimer" in html

    # Unconfigured resources are hidden
    assert "Configure Institutional Connection" not in html
    assert "[Configure" not in html

    # Short note is displayed because default JSON has no campus resource configured
    assert "No campus counseling resources have been configured" in html


def test_support_center_shows_configured_campus_resource(client, db, tmp_path, monkeypatch, app):
    """When a campus resource is configured (is_configured=true), it is displayed."""
    create_test_student(db, username="campus_sup_student", email="campussup@test.com", password="testpass123")
    client.post('/auth/login', data={'email': 'campussup@test.com', 'password': 'testpass123'})

    custom_json = {
        "resources": [
            {
                "id": "tele_manas",
                "name": "Tele-MANAS",
                "category": "crisis",
                "is_emergency": True,
                "is_configured": True,
                "contact": {"phone": "14416"}
            },
            {
                "id": "campus_counselor_abc",
                "name": "Campus Wellness Center (Main Hall)",
                "category": "counseling",
                "is_emergency": False,
                "is_configured": True,
                "contact": {"phone": "011-23456789", "email": "counselor@campus.edu"},
                "location": "Building A, Room 101"
            }
        ]
    }
    custom_path = str(tmp_path / "custom_support.json")
    with open(custom_path, 'w', encoding='utf-8') as f:
        json.dump(custom_json, f)

    app.config['SUPPORT_RESOURCES_PATH'] = custom_path
    try:
        res = client.get('/support')
        assert res.status_code == 200
        html = res.get_data(as_text=True)

        assert "Campus Wellness Center (Main Hall)" in html
        assert "011-23456789" in html
        assert "Building A, Room 101" in html
        assert "No campus counseling resources have been configured" not in html
    finally:
        app.config['SUPPORT_RESOURCES_PATH'] = os.path.join(app.root_path, 'content', 'support_resources.json')
