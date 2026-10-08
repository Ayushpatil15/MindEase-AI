"""Phase 3 tests — Current Mood Check-in."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from app.extensions import db
from app.models.mood import MoodEntry, MOOD_VALENCE_MAP, VALID_MOODS
from tests.conftest import create_test_student, create_test_admin


class TestMoodAuthorization:
    def test_unauthenticated_access(self, client):
        response = client.get('/mood/', follow_redirects=True)
        assert b'Welcome back' in response.data  # Redirects to login

    def test_admin_access(self, client, db):
        create_test_admin(db, email='admin_mood@test.com', password='pass')
        client.post('/auth/login', data={'email': 'admin_mood@test.com', 'password': 'pass'})
        response = client.get('/mood/')
        assert response.status_code == 403


class TestMoodValidation:
    def test_valid_submission(self, client, db):
        user = create_test_student(db, email='mood_student@test.com', password='pass')
        client.post('/auth/login', data={'email': 'mood_student@test.com', 'password': 'pass'})
        
        data = {
            'mood': 'Good',
            'journal_note': 'Had a productive day.'
        }
        response = client.post('/mood/', data=data, follow_redirects=True)
        assert response.status_code == 200
        assert b'Mood check-in saved successfully' in response.data
        
        # Verify persistence and valence derivation
        entry = MoodEntry.query.filter_by(user_id=user.id).first()
        assert entry is not None
        assert entry.mood == 'Good'
        assert entry.valence == 1
        assert entry.journal_note == 'Had a productive day.'

    def test_missing_mood_rejected(self, client, db):
        create_test_student(db, email='mood2@test.com', password='pass')
        client.post('/auth/login', data={'email': 'mood2@test.com', 'password': 'pass'})
        
        response = client.post('/mood/', data={'journal_note': 'No mood selected.'})
        # WTForms InputRequired handles missing radio fields
        assert b'Please select a mood' in response.data
        assert MoodEntry.query.count() == 0

    def test_invalid_mood_rejected(self, client, db):
        create_test_student(db, email='mood3@test.com', password='pass')
        client.post('/auth/login', data={'email': 'mood3@test.com', 'password': 'pass'})
        
        response = client.post('/mood/', data={'mood': 'Ecstatic'}) # Old phase 0 mood
        assert b'Not a valid choice' in response.data
        assert MoodEntry.query.count() == 0

    def test_valence_cannot_be_overridden_by_client(self, client, db):
        user = create_test_student(db, email='mood4@test.com', password='pass')
        client.post('/auth/login', data={'email': 'mood4@test.com', 'password': 'pass'})
        
        # Submit 'Very low' (-2) but try to hack valence to +2
        data = {
            'mood': 'Very low',
            'valence': 2,
            'journal_note': 'Trying to trick the server.'
        }
        response = client.post('/mood/', data=data, follow_redirects=True)
        
        entry = MoodEntry.query.filter_by(user_id=user.id).first()
        assert entry.mood == 'Very low'
        assert entry.valence == -2  # Server derivation is authoritative


class TestMoodHistoryAndTrend:
    def test_history_shows_own_entries_only(self, client, db):
        user1 = create_test_student(db, username='mood_owner', email='owner@test.com', password='pass')
        user2 = create_test_student(db, username='mood_snooper', email='snooper@test.com', password='pass')
        
        # User 1 logs mood
        client.post('/auth/login', data={'email': 'owner@test.com', 'password': 'pass'})
        client.post('/mood/', data={'mood': 'Okay', 'journal_note': 'Owner note'})
        client.post('/auth/logout')
        
        # User 2 logs mood
        client.post('/auth/login', data={'email': 'snooper@test.com', 'password': 'pass'})
        client.post('/mood/', data={'mood': 'Good', 'journal_note': 'Snooper note'})
        
        response = client.get('/mood/history')
        assert b'Snooper note' in response.data
        assert b'Owner note' not in response.data

    def test_trend_data_is_accurate(self, client, app, db):
        user = create_test_student(db, email='trend@test.com', password='pass')
        client.post('/auth/login', data={'email': 'trend@test.com', 'password': 'pass'})
        
        # Create history directly in DB to manipulate timestamps
        now = datetime.now(timezone.utc)
        
        entry1 = MoodEntry(user_id=user.id, mood='Low', valence=-1, created_at=now - timedelta(days=2))
        entry2 = MoodEntry(user_id=user.id, mood='Good', valence=1, created_at=now - timedelta(days=1))
        
        # This one is 10 days old, should not be in trend
        entry3 = MoodEntry(user_id=user.id, mood='Very low', valence=-2, created_at=now - timedelta(days=10))
        
        db.session.add_all([entry1, entry2, entry3])
        db.session.commit()
        
        response = client.get('/mood/history')
        assert response.status_code == 200
        
        # Extract the JSON trend data from the rendered template
        # We can look for it in the script block
        assert b'Low' in response.data
        assert b'Good' in response.data
        
        # Trend should only have 2 elements (entry1 and entry2)
        # We will parse the response string for the JSON array
        # This is a bit hacky but ensures the template receives the right data length
        html = response.data.decode('utf-8')
        assert '"valence": -1' in html
        assert '"valence": 1' in html
        assert '"valence": -2' not in html

    def test_empty_history_state(self, client, db):
        user = create_test_student(db, email='empty@test.com', password='pass')
        client.post('/auth/login', data={'email': 'empty@test.com', 'password': 'pass'})
        
        response = client.get('/mood/history')
        assert b"You haven't logged any moods." in response.data
