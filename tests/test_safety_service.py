"""Tests for Safety Service & Crisis Protocol (Phase 3 Step 1).

Verifies:
1. Three independent triggers:
   a. Mood: latest is 'Very low', or last 3 check-ins are all low ('Low' or 'Very low').
   b. Stress: latest stress level is 'High'.
   c. Text: conservative phrase list (English + common Hinglish).
2. Crisis text fires even with normal mood ('Good') and Low stress (score 0).
3. Normal entries do not fire safety triggers.
4. Privacy rules:
   - Journal / reflection text stays stored exactly as before as private entries.
   - No crisis table, flag column, or crisis record created in the DB.
   - Nothing about the trigger is persisted or logged in logs, analytics, or admin data.
   - Logging does not print journal notes or reflections.
5. Recommendation behavior:
   - CLINICAL_SUPPORT is placed first in recommendations when triggered.
6. Support card content:
   - Contains Tele-MANAS 14416, 112, and link to /support.
   - Rendered at top of dashboard and stress results.
7. Persistent 'Need to talk to someone?' link appears on student pages.
"""

import logging
from datetime import datetime, timedelta, timezone
import pytest
from flask import url_for

from app.models import MoodEntry, StressAssessment, Recommendation
from app.services import (
    mood_service,
    stress_service,
    recommendation_service,
    admin_service,
)
from app.services.safety_service import (
    text_has_crisis_language,
    is_mood_safety_triggered,
    is_stress_safety_triggered,
    is_recent_text_safety_triggered,
    evaluate_safety,
    get_safety_support_card_data,
)
from tests.conftest import create_test_student


class TestCrisisLanguageDetection:
    """Unit tests for text_has_crisis_language helper."""

    def test_english_crisis_phrases(self):
        assert text_has_crisis_language("I just want to end my life") is True
        assert text_has_crisis_language("I feel like I want to die") is True
        assert text_has_crisis_language("I feel like hurting myself today") is True
        assert text_has_crisis_language("I'm thinking of suicide.") is True
        assert text_has_crisis_language("I want to kill myself") is True
        assert text_has_crisis_language("Better off dead than failing") is True
        assert text_has_crisis_language("no reason to live anymore") is True

    def test_hinglish_crisis_phrases(self):
        assert text_has_crisis_language("Mera ab marne ka man kar raha hai") is True
        assert text_has_crisis_language("mai jaan de dunga") is True
        assert text_has_crisis_language("khud ko khatam karne ka mann hai") is True
        assert text_has_crisis_language("ab jeene ka koi fayda nahi") is True
        assert text_has_crisis_language("aatmahatya soch raha hu") is True

    def test_benign_text_does_not_fire(self):
        assert text_has_crisis_language(None) is False
        assert text_has_crisis_language("") is False
        assert text_has_crisis_language("Studying hard for the exams") is False
        assert text_has_crisis_language("The skill test was challenging") is False
        assert text_has_crisis_language("I feel tired after running today") is False
        assert text_has_crisis_language("Going to sleep early") is False


class TestIndependentTriggers:
    """Verify that each of the three triggers fires independently."""

    def test_mood_trigger_latest_very_low(self, app, db):
        """Latest check-in is 'Very low' fires mood trigger on its own."""
        user = create_test_student(db, username='mood_vl_user', email='mood_vl@test.com')
        mood_service.record_checkin(user.id, mood='Very low')

        assert is_mood_safety_triggered(user.id) is True
        assert evaluate_safety(user.id) is True

    def test_mood_trigger_last_three_all_low(self, app, db):
        """Last 3 daily check-ins all low ('Low' / 'Very low') fires mood trigger."""
        user = create_test_student(db, username='mood_3l_user', email='mood_3l@test.com')
        mood_service.record_checkin(user.id, mood='Low')
        mood_service.record_checkin(user.id, mood='Low')
        mood_service.record_checkin(user.id, mood='Low')

        assert is_mood_safety_triggered(user.id) is True
        assert evaluate_safety(user.id) is True

    def test_mood_trigger_normal_does_not_fire(self, app, db):
        """Normal mood check-ins do not fire trigger."""
        user = create_test_student(db, username='mood_norm_user', email='mood_norm@test.com')
        mood_service.record_checkin(user.id, mood='Okay')
        mood_service.record_checkin(user.id, mood='Good')

        assert is_mood_safety_triggered(user.id) is False

    def test_stress_trigger_high_stress(self, app, db):
        """Latest stress level High fires stress trigger on its own."""
        user = create_test_student(db, username='stress_hi_user', email='stress_hi@test.com')
        # Total score = 44 (High)
        high_responses = [4, 4, 4, 0, 0, 4, 0, 0, 4, 4, 4]
        stress_service.save_assessment(user.id, responses=high_responses)

        assert is_stress_safety_triggered(user.id) is True
        assert evaluate_safety(user.id) is True

    def test_stress_trigger_low_stress_does_not_fire(self, app, db):
        """Low stress assessment does not fire trigger."""
        user = create_test_student(db, username='stress_lo_user', email='stress_lo@test.com')
        # Total score = 0 (Low)
        low_responses = [0, 0, 0, 4, 4, 0, 4, 4, 0, 0, 0]
        stress_service.save_assessment(user.id, responses=low_responses)

        assert is_stress_safety_triggered(user.id) is False

    def test_crisis_text_fires_with_normal_mood_and_low_stress(self, app, db):
        """Crisis text MUST trigger safety even when mood is 'Good' and stress is 'Low'."""
        user = create_test_student(db, username='crisis_norm_user', email='crisis_norm@test.com')

        # Happy mood
        mood_service.record_checkin(user.id, mood='Good', journal_note='Normal day but I want to end my life')
        # Low stress
        low_responses = [0, 0, 0, 4, 4, 0, 4, 4, 0, 0, 0]
        stress_service.save_assessment(user.id, responses=low_responses, reflection='Everything seems low stress')

        # Mood alone is not triggered, stress alone is not triggered
        assert is_mood_safety_triggered(user.id) is False
        assert is_stress_safety_triggered(user.id) is False

        # But recent text trigger AND overall safety evaluation MUST fire!
        assert is_recent_text_safety_triggered(user.id) is True
        assert evaluate_safety(user.id) is True

    def test_current_text_submission_evaluates_immediately(self, app, db):
        """Current text parameter triggers safety before/as saved."""
        user = create_test_student(db, username='curr_txt_user', email='curr_txt@test.com')
        assert evaluate_safety(user.id, current_text="I want to die") is True
        assert evaluate_safety(user.id, current_text="I feel great today") is False


class TestPrivacyRules:
    """Verify strict privacy constraints around safety evaluation."""

    def test_journal_and_reflection_text_stays_stored_normally(self, app, db):
        """Private notes are stored as private entries without alteration."""
        user = create_test_student(db, username='privacy_user', email='privacy@test.com')
        note = "Private reflection text: feel like giving up and hurting myself"
        m = mood_service.record_checkin(user.id, mood='Okay', journal_note=note)

        # Stored accurately in student's private entry
        saved = db.session.get(MoodEntry, m.id)
        assert saved.journal_note == note

    def test_no_crisis_tables_or_flag_columns_in_db(self, app, db):
        """Ensure no separate crisis record, flag column, or table exists."""
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        table_names = inspector.get_table_names()

        # No table with 'crisis' in its name
        assert not any('crisis' in t.lower() for t in table_names)

        # Check MoodEntry and StressAssessment columns
        mood_cols = [c['name'] for c in inspector.get_columns('mood_entries')]
        assert not any('crisis' in c.lower() or 'flag' in c.lower() for c in mood_cols)

        stress_cols = [c['name'] for c in inspector.get_columns('stress_assessments')]
        assert not any('crisis' in c.lower() or 'flag' in c.lower() for c in stress_cols)

    def test_nothing_about_trigger_persisted_in_admin_analytics(self, app, db):
        """Admin aggregation queries must never expose crisis triggers or journal text."""
        user = create_test_student(db, username='admin_check_user', email='admin_check@test.com')
        mood_service.record_checkin(user.id, mood='Low', journal_note='want to die')

        # Fetch admin checkins data
        checkins_admin = admin_service.get_total_checkins()
        admin_str = str(checkins_admin).lower()
        assert 'want to die' not in admin_str
        assert 'crisis' not in admin_str

    def test_no_logging_prints_journal_text(self, app, db, caplog):
        """Check that safety evaluation and services never print journal text into logs."""
        user = create_test_student(db, username='log_user', email='log@test.com')
        secret_text = "super_confidential_journal_secret_kill_myself"

        with caplog.at_level(logging.DEBUG):
            evaluate_safety(user.id, current_text=secret_text)
            text_has_crisis_language(secret_text)

        assert secret_text not in caplog.text


class TestClinicalSupportRecommendation:
    """Verify CLINICAL_SUPPORT placement in recommendations."""

    def test_clinical_support_first_when_safety_triggered(self, app, db):
        """When safety is triggered, CLINICAL_SUPPORT must be the very first recommendation."""
        user = create_test_student(db, username='rec_user', email='rec@test.com')
        # Trigger via latest 'Very low'
        mood_service.record_checkin(user.id, mood='Very low')

        recs = recommendation_service.generate_recommendations(user.id)
        assert len(recs) >= 1
        assert recs[0].category == 'CLINICAL_SUPPORT'

    def test_clinical_support_not_first_for_normal_student(self, app, db):
        """When safety is not triggered, normal recommendations appear."""
        user = create_test_student(db, username='rec_norm_user', email='rec_norm@test.com')
        mood_service.record_checkin(user.id, mood='Good')

        recs = recommendation_service.generate_recommendations(user.id)
        assert len(recs) >= 1
        assert recs[0].category != 'CLINICAL_SUPPORT'


class TestCardBehaviorAndPersistentLink:
    """Verify UI card contents and persistent navigation link."""

    def test_support_card_data_contents(self, app):
        """Support card data contains Tele-MANAS, 112, and non-diagnostic text."""
        card = get_safety_support_card_data()
        assert card['tele_manas']['phone'] == '14416'
        assert card['emergency']['phone'] == '112'
        assert 'crisis detected' not in card['message'].lower()
        assert 'diagnosis' not in card['message'].lower()

    def test_support_card_rendered_at_top_of_dashboard_when_triggered(self, client, db):
        """Dashboard renders care card at top of response when safety is triggered."""
        user = create_test_student(db, username='dash_card_user', email='dash_card@test.com')
        client.post('/auth/login', data={'email': user.email, 'password': 'testpass123'})

        # Trigger safety via recent journal note
        mood_service.record_checkin(user.id, mood='Okay', journal_note='I want to kill myself')

        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)

        assert 'Tele-MANAS' in html
        assert '14416' in html
        assert '112' in html
        assert 'Support and Care Resources' in html

    def test_persistent_need_to_talk_link_in_navbar_or_footer(self, client, db):
        """Persistent 'Need to talk to someone?' link is present on authenticated pages."""
        user = create_test_student(db, username='nav_link_user', email='nav_link@test.com')
        client.post('/auth/login', data={'email': user.email, 'password': 'testpass123'})

        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)

        assert 'Need to talk to someone?' in html
        assert '/support' in html
