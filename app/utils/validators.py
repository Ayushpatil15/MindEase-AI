"""Input validation helpers.

Centralizes sanitization and validation so it can be reused across
route handlers and services.
"""

import re

from app.models.mood import VALID_MOODS
from app.models.emotion import EMOTION_CLASSES
from app.models.recommendation import RECOMMENDATION_CATEGORIES


# ── Username ──────────────────────────────────────────────────────────

USERNAME_PATTERN = re.compile(r'^[a-zA-Z0-9_]{3,64}$')


def validate_username(username):
    """Return (cleaned_username, error_message | None)."""
    if not username or not isinstance(username, str):
        return None, 'Username is required.'
    username = username.strip()
    if not USERNAME_PATTERN.match(username):
        return None, 'Username must be 3–64 characters: letters, digits, or underscores.'
    return username, None


# ── Email ─────────────────────────────────────────────────────────────

EMAIL_PATTERN = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')


def validate_email(email):
    """Return (cleaned_email, error_message | None)."""
    if not email or not isinstance(email, str):
        return None, 'Email is required.'
    email = email.strip().lower()
    if len(email) > 120:
        return None, 'Email must be 120 characters or fewer.'
    if not EMAIL_PATTERN.match(email):
        return None, 'Please enter a valid email address.'
    return email, None


# ── Password ──────────────────────────────────────────────────────────

def validate_password(password):
    """Return (password, error_message | None)."""
    if not password or not isinstance(password, str):
        return None, 'Password is required.'
    if len(password) < 8:
        return None, 'Password must be at least 8 characters.'
    return password, None


# ── Stress Responses ──────────────────────────────────────────────────

def validate_stress_responses(responses):
    """Validate exactly 11 integers each in [0, 4]."""
    if not isinstance(responses, list):
        return None, 'Responses must be a list.'
    if len(responses) != 11:
        return None, 'Exactly 11 responses are required.'
    try:
        values = [int(r) for r in responses]
    except (TypeError, ValueError):
        return None, 'Each response must be an integer.'
    if any(v < 0 or v > 4 for v in values):
        return None, 'Each response must be between 0 and 4.'
    return values, None


def validate_mood_submission(mood, journal_note=None):
    """Validate mood string against canonical VALID_MOODS."""
    from app.models.mood import VALID_MOODS
    
    if not mood:
        return None, None, 'Mood is required.'
        
    if mood not in VALID_MOODS:
        return None, None, 'Invalid mood selected.'
        
    clean_note, err = validate_journal_note(journal_note)
    if err:
        return None, None, err
        
    return mood, clean_note, None


# ── Mood ──────────────────────────────────────────────────────────────

def validate_mood(mood):
    """Return (mood, error_message | None)."""
    if not mood or mood not in VALID_MOODS:
        return None, f'Mood must be one of: {", ".join(VALID_MOODS)}.'
    return mood, None


# ── Journal Note ──────────────────────────────────────────────────────

MAX_JOURNAL_LENGTH = 5000


def validate_journal_note(note):
    """Sanitize and length-check an optional journal note.

    Returns (note_or_None, error_message | None).
    """
    if not note or not isinstance(note, str):
        return None, None  # journal is optional
    note = note.strip()
    if len(note) > MAX_JOURNAL_LENGTH:
        return None, f'Journal note must be {MAX_JOURNAL_LENGTH} characters or fewer.'
    return note, None


# ── Task Title ────────────────────────────────────────────────────────

def validate_task_title(title):
    """Return (title, error_message | None)."""
    if not title or not isinstance(title, str):
        return None, 'Task title is required.'
    title = title.strip()
    if len(title) > 200:
        return None, 'Task title must be 200 characters or fewer.'
    if len(title) < 1:
        return None, 'Task title cannot be empty.'
    return title, None


# ── Timezone ──────────────────────────────────────────────────────────

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def validate_timezone(tz_name):
    """Validate and sanitize an IANA timezone string.

    Returns:
        str: Valid IANA timezone string, or 'UTC' if invalid or missing.
    """
    if not tz_name or not isinstance(tz_name, str):
        return 'UTC'
    clean_tz = tz_name.strip()
    try:
        ZoneInfo(clean_tz)
        return clean_tz
    except (ZoneInfoNotFoundError, ValueError, TypeError, KeyError):
        return 'UTC'

