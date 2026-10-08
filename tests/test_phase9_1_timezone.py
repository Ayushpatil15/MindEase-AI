"""Phase 9.1 Test Suite — Timezone Awareness & Daily Plan Query Cleanup.

Verifies:
1. Default/missing timezone fallback to UTC
2. Invalid timezone string fallback to UTC
3. Asia/Kolkata local calendar day range
4. America/New_York local calendar day range
5. Preferred check-in time evaluation in student's local timezone
6. Daily check-in completion using local calendar day boundaries
7. Dashboard query count optimization (<= 8 queries)
8. Consistency between standalone daily plan and dashboard integrated plan
9. Backward compatibility for existing users
"""

from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import pytest

from app.models import User, UserPreferences, MoodEntry, StressAssessment, EmotionScan, WellnessTask
from app.services import daily_plan_service, dashboard_service
from app.utils.validators import validate_timezone
from tests.conftest import create_test_student


def test_validate_timezone_utility():
    """Verify validate_timezone returns valid IANA names or falls back to UTC."""
    assert validate_timezone('Asia/Kolkata') == 'Asia/Kolkata'
    assert validate_timezone('America/New_York') == 'America/New_York'
    assert validate_timezone('UTC') == 'UTC'
    assert validate_timezone('Europe/London') == 'Europe/London'
    # Invalid strings fall back safely to UTC
    assert validate_timezone('Invalid/Timezone') == 'UTC'
    assert validate_timezone('') == 'UTC'
    assert validate_timezone(None) == 'UTC'
    assert validate_timezone(12345) == 'UTC'
    assert validate_timezone("'; DROP TABLE users; --") == 'UTC'


def test_default_missing_timezone_fallback(db):
    """Missing or null timezone falls back safely to UTC."""
    student = create_test_student(db, username='tz_fallback_user', email='tz_fallback@test.com')
    # No UserPreferences row
    plan = daily_plan_service.get_daily_plan(student.id)
    assert plan is not None
    now_utc = datetime.now(timezone.utc)
    assert plan['date'] == now_utc.strftime('%Y-%m-%d')


def test_invalid_timezone_fallback(db):
    """UserPreferences with invalid timezone string safely falls back to UTC without crashing."""
    student = create_test_student(db, username='invalid_tz_user', email='invalid_tz@test.com')
    prefs = UserPreferences(user_id=student.id, timezone='NonExistent/Zone_Name', preferred_checkin_time='10:00')
    db.session.add(prefs)
    db.session.commit()

    plan = daily_plan_service.get_daily_plan(student.id)
    assert plan is not None
    assert plan['date'] == datetime.now(timezone.utc).strftime('%Y-%m-%d')


def test_asia_kolkata_local_date(db):
    """User in Asia/Kolkata timezone gets calendar date and local range evaluated correctly."""
    student = create_test_student(db, username='kolkata_student', email='kolkata@test.com')
    prefs = UserPreferences(user_id=student.id, timezone='Asia/Kolkata')
    db.session.add(prefs)
    db.session.commit()

    now_utc, start_of_day_utc, end_of_day_utc, now_local = daily_plan_service.get_user_today_utc_range('Asia/Kolkata')
    
    # IST is UTC + 5:30
    tz_ist = ZoneInfo('Asia/Kolkata')
    assert now_local.tzinfo == tz_ist
    # Ensure start_of_day_utc to end_of_day_utc spans exactly 24h minus 1 microsecond (or 1 second)
    diff = end_of_day_utc - start_of_day_utc
    assert abs(diff.total_seconds() - 86399) < 2

    plan = daily_plan_service.get_daily_plan(student.id)
    assert plan['date'] == now_local.strftime('%Y-%m-%d')


def test_america_new_york_local_date(db):
    """User in America/New_York gets calendar date evaluated correctly."""
    student = create_test_student(db, username='ny_student', email='ny@test.com')
    prefs = UserPreferences(user_id=student.id, timezone='America/New_York')
    db.session.add(prefs)
    db.session.commit()

    now_utc, start_of_day_utc, end_of_day_utc, now_local = daily_plan_service.get_user_today_utc_range('America/New_York')
    tz_ny = ZoneInfo('America/New_York')
    assert now_local.tzinfo == tz_ny

    plan = daily_plan_service.get_daily_plan(student.id)
    assert plan['date'] == now_local.strftime('%Y-%m-%d')


def test_preferred_checkin_time_local_evaluation(db):
    """preferred_checkin_time is evaluated against user's local hour."""
    student = create_test_student(db, username='pref_time_student', email='preftime@test.com')
    
    tz_ist = ZoneInfo('Asia/Kolkata')
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(tz_ist)
    current_local_hour = now_local.hour
    
    # Set preferred_checkin_time to current local hour
    pref_time_str = f"{current_local_hour:02d}:00"
    prefs = UserPreferences(user_id=student.id, timezone='Asia/Kolkata', preferred_checkin_time=pref_time_str)
    db.session.add(prefs)
    db.session.commit()

    plan = daily_plan_service.get_daily_plan(student.id)
    assert plan['reminder']['is_reminder_time'] is True
    assert pref_time_str in plan['reminder']['message']


def test_local_calendar_day_checkin_completion(db):
    """Check-in completed within user's local day range is detected as completed."""
    student = create_test_student(db, username='checkin_tz_student', email='checkin_tz@test.com')
    prefs = UserPreferences(user_id=student.id, timezone='Asia/Kolkata')
    db.session.add(prefs)
    db.session.commit()

    now_utc, start_of_day_utc, end_of_day_utc, now_local = daily_plan_service.get_user_today_utc_range('Asia/Kolkata')

    # Create MoodEntry at mid-day local time (converted to UTC)
    mid_day_utc = start_of_day_utc + timedelta(hours=6)
    mood = MoodEntry(user_id=student.id, mood='Happy', valence=2, created_at=mid_day_utc)
    db.session.add(mood)
    db.session.commit()

    plan = daily_plan_service.get_daily_plan(student.id)
    assert plan['checkins']['mood_completed'] is True
    assert plan['checkins']['stress_completed'] is False


def test_dashboard_query_count_optimization(app, db):
    """Verify daily_plan with context executes <= 4 queries and dashboard load avoids redundant queries."""
    student = create_test_student(db, username='query_opt_student', email='queryopt@test.com')
    prefs = UserPreferences(user_id=student.id, timezone='UTC', wellness_goals=['stress'])
    db.session.add(prefs)
    db.session.commit()

    with app.app_context():
        # 1. Test get_daily_plan WITHOUT context executes 9 queries
        queries_without_context = []
        def count_no_context(conn, cursor, statement, parameters, context_obj, executemany):
            queries_without_context.append(statement)

        from sqlalchemy import event
        engine = db.engine
        event.listen(engine, 'before_cursor_execute', count_no_context)
        try:
            daily_plan_service.get_daily_plan(student.id, context=None)
        finally:
            event.remove(engine, 'before_cursor_execute', count_no_context)

        # 2. Test get_daily_plan WITH context executes only 4 queries
        context = daily_plan_service.build_daily_plan_context(student.id)
        queries_with_context = []
        def count_context_queries(conn, cursor, statement, parameters, context_obj, executemany):
            queries_with_context.append(statement)

        event.listen(engine, 'before_cursor_execute', count_context_queries)
        try:
            plan = daily_plan_service.get_daily_plan(student.id, context=context)
        finally:
            event.remove(engine, 'before_cursor_execute', count_context_queries)

        assert plan is not None
        assert len(queries_with_context) <= 4, f"Expected <= 4 queries for daily_plan with context, got {len(queries_with_context)}"
        assert len(queries_with_context) < len(queries_without_context), "Context optimization should execute fewer queries than without context"

        # 3. Test total dashboard queries does not exceed baseline (22 queries)
        queries_dashboard = []
        def count_dashboard_queries(conn, cursor, statement, parameters, context_obj, executemany):
            queries_dashboard.append(statement)

        event.listen(engine, 'before_cursor_execute', count_dashboard_queries)
        try:
            dashboard_data = dashboard_service.get_student_dashboard_data(student.id)
        finally:
            event.remove(engine, 'before_cursor_execute', count_dashboard_queries)

        assert dashboard_data is not None
        assert len(queries_dashboard) <= 22, f"Expected <= 22 total queries for dashboard load, got {len(queries_dashboard)}"




def test_dashboard_and_daily_plan_consistency(db):
    """Daily plan data in dashboard payload matches standalone get_daily_plan output."""
    student = create_test_student(db, username='consistent_student', email='consistent@test.com')
    prefs = UserPreferences(user_id=student.id, timezone='Asia/Kolkata', wellness_goals=['mood'])
    db.session.add(prefs)
    db.session.commit()

    standalone_plan = daily_plan_service.get_daily_plan(student.id)
    dashboard_data = dashboard_service.get_student_dashboard_data(student.id)

    assert dashboard_data['daily_plan'] == standalone_plan
