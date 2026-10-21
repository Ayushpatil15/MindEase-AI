"""Tests for Pomodoro Focus Activity (Timestamp-Based Timing Logic and Rendering)."""

import pytest
from app.services.activity_service import get_activity_by_id
from tests.conftest import create_test_student


class PomodoroTimestampModel:
    """Python reference model of the client-side timestamp-based timing math."""

    def __init__(self, focus_seconds=25 * 60, break_seconds=5 * 60):
        self.focus_seconds = focus_seconds
        self.break_seconds = break_seconds
        self.phase = 'idle'
        self.phase_remaining_ms = focus_seconds * 1000
        self.phase_end_time = None
        self.is_running = False

    def start(self, current_time_ms):
        self.is_running = True
        self.phase = 'focus' if self.phase == 'idle' else self.phase
        self.phase_end_time = current_time_ms + self.phase_remaining_ms

    def pause(self, current_time_ms):
        if not self.is_running:
            return
        self.is_running = False
        self.phase_remaining_ms = max(0, self.phase_end_time - current_time_ms)

    def tick(self, current_time_ms):
        if not self.is_running:
            return self.phase, self.get_remaining_seconds()

        self.phase_remaining_ms = max(0, self.phase_end_time - current_time_ms)

        if self.phase_remaining_ms <= 0:
            if self.phase == 'focus':
                # Switch to break
                self.phase = 'break'
                self.phase_remaining_ms = self.break_seconds * 1000
                self.phase_end_time = current_time_ms + self.phase_remaining_ms
            elif self.phase == 'break':
                self.phase = 'done'
                self.is_running = False

        return self.phase, self.get_remaining_seconds()

    def get_remaining_seconds(self):
        return max(0, (self.phase_remaining_ms + 999) // 1000)


def test_pomodoro_activity_definition():
    """Verify pomodoro_focus definition in registry."""
    act = get_activity_by_id('pomodoro_focus')
    assert act is not None
    assert act['category'] == 'Academic'
    assert act['duration_minutes'] == 30
    assert len(act['steps']) >= 2


def test_pomodoro_timestamp_simulation_background_throttling():
    """Test that timestamp-based math stays accurate under arbitrary browser background tab throttling."""
    timer = PomodoroTimestampModel()
    start_ms = 1_000_000_000_000  # Arbitrary epoch ms

    timer.start(start_ms)
    assert timer.phase == 'focus'
    assert timer.get_remaining_seconds() == 25 * 60

    # Simulate tab in background: next tick happens 10 minutes later (600 seconds)
    simulated_now = start_ms + (600 * 1000)
    phase, remaining = timer.tick(simulated_now)
    assert phase == 'focus'
    assert remaining == 15 * 60  # Exactly 15 minutes remaining

    # Another background delay: 15 minutes later (remaining expires)
    simulated_now += (15 * 60 * 1000)
    phase, remaining = timer.tick(simulated_now)
    # Automatic transition to 5m break
    assert phase == 'break'
    assert remaining == 5 * 60  # Exactly 5 minutes break

    # Background delay: 5 minutes later
    simulated_now += (5 * 60 * 1000)
    phase, remaining = timer.tick(simulated_now)
    assert phase == 'done'
    assert remaining == 0


def test_pomodoro_pause_and_resume():
    """Test pause and resume preservation of remaining time."""
    timer = PomodoroTimestampModel()
    t0 = 1_700_000_000_000

    timer.start(t0)
    # 5 minutes elapse
    t1 = t0 + (300 * 1000)
    timer.tick(t1)
    timer.pause(t1)
    assert timer.get_remaining_seconds() == 20 * 60

    # User leaves page paused for 1 hour
    t2 = t1 + (3600 * 1000)
    timer.tick(t2)
    assert timer.get_remaining_seconds() == 20 * 60  # Remains paused!

    # Resumes
    timer.start(t2)
    t3 = t2 + (120 * 1000)  # 2 minutes after resume
    phase, remaining = timer.tick(t3)
    assert phase == 'focus'
    assert remaining == 18 * 60


def test_pomodoro_detail_page_renders_components(client, db):
    """Verify that /activities/pomodoro_focus renders the progress ring, phase label, and controls."""
    student = create_test_student(db, username='pomo_user', email='pomo@test.com')
    with client.session_transaction() as sess:
        sess['_user_id'] = str(student.id)
        sess['_fresh'] = True

    res = client.get('/activities/pomodoro_focus')
    assert res.status_code == 200
    html = res.get_data(as_text=True)

    # Progress ring
    assert 'pomodoro-ring-circle' in html
    assert 'pomodoro-ring-svg' in html
    # Timer display and phase badge
    assert 'pomodoro-timer-display' in html
    assert 'pomodoro-phase-badge' in html
    # Controls
    assert 'btn-pomodoro-start' in html
    assert 'btn-pomodoro-pause' in html
    assert 'btn-pomodoro-reset' in html
    assert 'btn-pomodoro-skip' in html
