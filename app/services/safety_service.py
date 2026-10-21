"""Safety Service — Crisis language detection and support card triggers.

SAFETY ARCHITECTURE & PRIVACY PRINCIPLES:
1. Three Independent Triggers:
   - Mood: latest check-in is 'Very low', or the last 3 check-ins are all low ('Low' or 'Very low').
   - Stress: latest assessment level is 'High'.
   - Text: text_has_crisis_language(text) detects conservative self-harm / suicide phrasing.
   Any single trigger is sufficient to display the care & support card.
2. Privacy Guarantees:
   - Journal notes and reflection text remain stored strictly as the student's private entry.
   - Never create a separate crisis record, flag column, or database table.
   - Never log or persist the matched phrase, trigger result, or text in application logs,
     analytics, or admin data. All safety evaluations occur in memory at request time only.
   - Request and error logging must never print student journal or reflection text.
3. Card Behavior:
   - Shows Tele-MANAS (14416 / 1800-891-4416), emergency response (112), and campus counseling (if configured).
   - Warm, plain, non-diagnostic wording: never claims the app "detected a crisis" or provides diagnosis.
   - Puts CLINICAL_SUPPORT recommendation category first in recommendations when triggered.

NOTE: The keyword check is a simple safety net, not a reliable detector and not a diagnostic tool.
"""

import json
import os
import re
import string
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

from flask import current_app
from app.extensions import db
from app.models.mood import MoodEntry
from app.models.stress import StressAssessment
from app.services import mood_service, stress_service


# Conservative phrase list for self-harm and suicide wording (English + common Hinglish + Devanagari Hindi).
# Note: The keyword check is a simple safety net, not a reliable detector and not a diagnostic tool.
CRISIS_PHRASES = [
    # English crisis and self-harm phrases
    "kill myself",
    "killing myself",
    "suicide",
    "suicidal",
    "end my life",
    "ending my life",
    "want to die",
    "wanna die",
    "better off dead",
    "self harm",
    "self-harm",
    "cut myself",
    "cutting myself",
    "hang myself",
    "hanging myself",
    "take my own life",
    "taking my own life",
    "no reason to live",
    "no point in living",
    "can't go on living",
    "cannot go on living",
    "hurting myself",
    "harm myself",
    "harming myself",
    "wish i was dead",
    "wish i were dead",
    "end it all",
    "ending it all",
    "don't want to live",
    "dont want to live",
    "do not want to live",
    "tired of living",

    # Common Hinglish phrasing
    "marne ka man",
    "marne ka mann",
    "marne ki iccha",
    "marne ki ichha",
    "jaan de dunga",
    "jaan de dungi",
    "jaan dena chahta",
    "jaan dena chahti",
    "mar jana chahta",
    "mar jana chahti",
    "khud ko khatam",
    "zindagi khatam",
    "aatmahatya",
    "atmahatya",
    "khudkushi",
    "khud kushi",
    "jeene ka koi fayda nahi",
    "jeene ki ichha nahi",
    "ab jeena nahi chahta",
    "ab jeena nahi chahti",
    "khud ko nuksan",
    "khud ko chot",

    # Devanagari Hindi phrases
    "आत्महत्या",
    "खुदकुशी",
    "मरने का मन",
    "मरना चाहता हूँ",
    "मरना चाहता हूं",
    "मरना चाहती हूँ",
    "मरना चाहती हूं",
    "जान देना चाहता हूँ",
    "जान देना चाहती हूँ",
    "जीने की इच्छा नहीं",
    "खुद को नुकसान",
]


def normalize_text_for_safety(text: str) -> str:
    """Normalize text by lowercasing, resolving curly apostrophes/quotes,
    stripping punctuation (including Devanagari dandas), and collapsing repeated spaces.
    """
    if not text:
        return ""

    cleaned = text.lower()

    # Normalize curly apostrophes and quotes
    for curly_apos in ('’', '‘', 'ʼ', 'ʻ'):
        cleaned = cleaned.replace(curly_apos, "'")
    for curly_quote in ('“', '”'):
        cleaned = cleaned.replace(curly_quote, '"')

    # Remove apostrophes without space so "don't" and "dont", "can't" and "cant" unify
    cleaned = cleaned.replace("'", "")

    # Replace all punctuation and Devanagari dandas with spaces
    punctuation_to_strip = string.punctuation + '।॥'
    for char in punctuation_to_strip:
        cleaned = cleaned.replace(char, ' ')

    # Collapse repeated spaces
    tokens = cleaned.split()
    return ' '.join(tokens)


def text_has_crisis_language(text: Optional[str]) -> bool:
    """Evaluate text against conservative crisis / self-harm phrase list.

    Note: The keyword check is a simple safety net, not a reliable detector
    and not a diagnostic tool. Evaluated in-memory at request time only;
    never logged, cached, or persisted.

    Args:
        text (str, optional): Raw user journal note or reflection text.

    Returns:
        bool: True if crisis language is detected, False otherwise.
    """
    if not text or not isinstance(text, str):
        return False

    normalized_input = normalize_text_for_safety(text)
    if not normalized_input:
        return False

    padded_input = f" {normalized_input} "

    for phrase in CRISIS_PHRASES:
        normalized_phrase = normalize_text_for_safety(phrase)
        if not normalized_phrase:
            continue
        padded_phrase = f" {normalized_phrase} "
        if padded_phrase in padded_input:
            return True

    return False


_UNSET = object()


def is_mood_safety_triggered(user_id: int, latest_mood: Any = _UNSET) -> bool:
    """Evaluate mood trigger.

    Triggers if:
    - Latest check-in is 'Very low', OR
    - The last 3 distinct local calendar days (in student's timezone) with check-ins
      all have a latest check-in of low ('Low' or 'Very low').

    Definition:
    The 3 most recent distinct local calendar days (student's timezone, consistent with
    get_user_today_utc_range) that have a check-in, using each day's latest check-in.
    Multiple entries on the same local calendar day are collapsed to that day's latest entry.

    Args:
        user_id (int): Student user ID.
        latest_mood: Pre-fetched latest mood object, or None if user has none.

    Returns:
        bool: True if mood trigger fires.
    """
    if latest_mood is _UNSET:
        latest_mood = mood_service.get_latest(user_id)
    if not latest_mood:
        return False

    if latest_mood.mood == 'Very low':
        return True

    # If latest mood is not Low, the latest day's check-in cannot be Low
    if latest_mood.mood != 'Low':
        return False

    # Fetch user's local timezone consistent with get_user_today_utc_range
    from app.models.user_preferences import UserPreferences
    from app.utils.validators import validate_timezone
    from zoneinfo import ZoneInfo

    pref = UserPreferences.query.filter_by(user_id=user_id).first()
    tz_str = pref.timezone if (pref and pref.timezone) else 'UTC'
    user_tz = ZoneInfo(validate_timezone(tz_str))

    # Fetch recent entries to find the 3 most recent distinct local calendar days
    recent_entries = (
        MoodEntry.query
        .filter_by(user_id=user_id)
        .order_by(MoodEntry.created_at.desc())
        .limit(50)
        .all()
    )

    distinct_daily_latest = []
    seen_dates = set()

    for entry in recent_entries:
        local_date = _as_utc(entry.created_at).astimezone(user_tz).date()
        if local_date not in seen_dates:
            seen_dates.add(local_date)
            distinct_daily_latest.append(entry)
            if len(distinct_daily_latest) == 3:
                break

    if len(distinct_daily_latest) == 3 and all(e.mood in ('Low', 'Very low') for e in distinct_daily_latest):
        return True

    return False


def is_stress_safety_triggered(user_id: int, latest_stress: Any = _UNSET) -> bool:
    """Evaluate stress trigger.

    Triggers if the latest stress assessment level is 'High'.

    Args:
        user_id (int): Student user ID.
        latest_stress: Pre-fetched latest stress object, or None if user has none.

    Returns:
        bool: True if stress trigger fires.
    """
    if latest_stress is _UNSET:
        latest_stress = stress_service.get_latest(user_id)
    if not latest_stress:
        return False
    return latest_stress.level == 'High'


def _as_utc(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware in UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def is_recent_text_safety_triggered(
    user_id: int,
    latest_mood: Any = _UNSET,
    latest_stress: Any = _UNSET,
) -> bool:
    """Evaluate student's journal and reflection entries from the last 48 hours.

    Evaluated in memory at request time only. Avoids DB queries when latest
    entries are absent or older than 48 hours.

    Args:
        user_id (int): Student user ID.
        latest_mood: Pre-fetched latest mood object, or None if user has none.
        latest_stress: Pre-fetched latest stress object, or None if user has none.

    Returns:
        bool: True if any journal or reflection entry in last 48h has crisis wording.
    """
    two_days_ago = datetime.now(timezone.utc) - timedelta(hours=48)

    # 1. Recent Mood Entries in last 48h
    if latest_mood is _UNSET:
        latest_mood = mood_service.get_latest(user_id)
    if latest_mood and _as_utc(latest_mood.created_at) >= two_days_ago:
        recent_moods = (
            MoodEntry.query
            .filter(
                MoodEntry.user_id == user_id,
                MoodEntry.created_at >= two_days_ago,
                MoodEntry.journal_note.isnot(None),
            )
            .all()
        )
        for m in recent_moods:
            if text_has_crisis_language(m.journal_note):
                return True

    # 2. Recent Stress Assessments in last 48h
    if latest_stress is _UNSET:
        latest_stress = stress_service.get_latest(user_id)
    if latest_stress and _as_utc(latest_stress.created_at) >= two_days_ago:
        recent_stress = (
            StressAssessment.query
            .filter(
                StressAssessment.user_id == user_id,
                StressAssessment.created_at >= two_days_ago,
                StressAssessment.reflection.isnot(None),
            )
            .all()
        )
        for s in recent_stress:
            if text_has_crisis_language(s.reflection):
                return True

    return False


def evaluate_safety(
    user_id: int,
    current_text: Optional[str] = None,
    latest_mood: Any = _UNSET,
    latest_stress: Any = _UNSET,
) -> bool:
    """Evaluate safety status across all three independent triggers.

    Any one independent trigger is enough to show the support card:
    1. Text: current_text or entries from last 48 hours contain crisis language.
    2. Mood: latest is 'Very low' or last 3 daily check-ins are all low.
    3. Stress: latest stress level is 'High'.

    Crisis language must trigger it even when mood is normal and stress is low.

    Args:
        user_id (int): Student user ID.
        current_text (str, optional): Currently submitted text before/as saved.
        latest_mood: Pre-fetched latest mood object, or None if user has none.
        latest_stress: Pre-fetched latest stress object, or None if user has none.

    Returns:
        bool: True if any safety trigger is active.
    """
    # Trigger 1: Current submission text (evaluated before or as it is saved)
    if current_text and text_has_crisis_language(current_text):
        return True

    # Pre-fetch if not provided to eliminate redundant query executions
    if latest_mood is _UNSET:
        latest_mood = mood_service.get_latest(user_id)
    if latest_stress is _UNSET:
        latest_stress = stress_service.get_latest(user_id)

    # Trigger 2: Mood state
    if is_mood_safety_triggered(user_id, latest_mood=latest_mood):
        return True

    # Trigger 3: Stress state
    if is_stress_safety_triggered(user_id, latest_stress=latest_stress):
        return True

    # Trigger 4: Recent text submissions within 48h (e.g. for dashboard load)
    if is_recent_text_safety_triggered(user_id, latest_mood=latest_mood, latest_stress=latest_stress):
        return True

    return False


def get_safety_support_card_data() -> Dict[str, Any]:
    """Retrieve normalized support resource data for rendering the care card.

    Always includes:
    - Tele-MANAS (14416 / 1800-891-4416)
    - National Emergency Services (112)
    - Configured campus counseling resource (if is_configured is True)
    - Link to support center

    Returns:
        dict: Support card content with warm, plain, non-diagnostic wording.
    """
    resource_path = None
    try:
        resource_path = current_app.config.get('SUPPORT_RESOURCES_PATH')
    except Exception:
        pass

    if not resource_path:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        resource_path = os.path.join(base_dir, 'content', 'support_resources.json')

    campus_counselor = None
    if os.path.isfile(resource_path):
        try:
            with open(resource_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                resources = content.get('resources', [])
                for r in resources:
                    if r.get('category') == 'counseling' and r.get('is_configured') is True:
                        campus_counselor = {
                            'name': r.get('name'),
                            'phone': r.get('contact', {}).get('phone'),
                            'email': r.get('contact', {}).get('email'),
                            'location': r.get('location'),
                        }
                        break
        except Exception:
            pass

    return {
        'tele_manas': {
            'name': 'Tele-MANAS',
            'phone': '14416',
            'toll_free': '1800-891-4416',
            'description': '24/7 free, confidential national tele-mental health support.',
        },
        'emergency': {
            'name': 'Emergency Services',
            'phone': '112',
            'notice': 'If you are in immediate danger call 112',
        },
        'campus_counselor': campus_counselor,
        'support_url': '/support',
        'heading': "Support and Care Resources",
        'message': "If things are feeling heavy or overwhelming right now, supportive listeners and immediate care resources are available for you.",
    }
