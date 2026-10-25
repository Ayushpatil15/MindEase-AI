"""MindEase AI — Chat Context & Memory Service.

Assembles personalized, context-aware student wellness companion prompts:
1. User Profile: Real username and existing preferences without fabrication.
2. Wellness Context: Recent mood check-ins and stress assessments from actual DB records.
3. Persistent Memory: Compact, non-sensitive preferences with full user transparency and control.
4. Personality & Guardrails: Warm, witty, casual companion with zero medical diagnosis.
"""

import re
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from app.extensions import db
from app.models.user import User
from app.models.user_preferences import UserPreferences
from app.models.mood import MoodEntry
from app.models.stress import StressAssessment
from app.models.task import WellnessTask
from app.models.chat_memory import ChatMemory
from app.services.safety_service import text_has_crisis_language

logger = logging.getLogger(__name__)

# Wellness-related trigger terms to determine if wellness context is relevant to current turn
WELLNESS_RELEVANCE_KEYWORDS = {
    "stress", "stressed", "stressful", "anxious", "anxiety", "overwhelm",
    "overwhelmed", "burnout", "tired", "exhausted", "sleep", "insomnia",
    "sleeping", "mood", "sad", "down", "depressed", "struggling", "exam",
    "exams", "study", "studying", "procrastinat", "procrastination", "habit",
    "habits", "routine", "wellness", "feeling", "feel", "feels", "checkin",
    "check-in", "assessment", "score", "tasks", "task", "breathe", "breathing",
    "mindful", "mindfulness", "calm", "relax", "relaxing", "panic",
}

# Sensitive terms that must NEVER be extracted into persistent chatbot memory
SENSITIVE_MEMORY_PATTERNS = {
    "suicide", "kill", "die", "death", "depressed", "depression", "anxiety",
    "ptsd", "bipolar", "adhd", "ocd", "disorder", "therapy", "therapist",
    "counselor", "psychiatrist", "medication", "pill", "antidepressant",
    "hospital", "trauma", "abuse", "grief", "self-harm", "cutting", "overdose",
    "eating disorder", "anorexia", "bulimia", "relapse", "addiction",
}

# Base companion persona text
BASE_COMPANION_PERSONA = (
    "You are the MindEase AI Companion, a warm, witty, empathetic, and encouraging "
    "wellness partner for college and university students.\n\n"
    "Core Persona & Boundaries:\n"
    "- Role: Compassionate peer companion, NOT a therapist, doctor, or counselor. "
    "Never provide clinical diagnoses, psychiatric labels, or medical evaluations.\n"
    "- Tone: Friendly, casual, witty, authentic, conversational, and uplifting. Speak naturally "
    "like a supportive college friend. Use light humor when appropriate.\n"
    "- Flexibility: Comfortably handle casual topics, everyday student life, jokes, college banter, "
    "hobbies, dating advice, and lighthearted discussions without turning every topic into a therapy session.\n"
    "- When Emotionally Challenged: Offer warm, non-judgmental presence and evidence-informed self-care "
    "(box breathing, grounding, breaking tasks down). Ask clarifying questions rather than assuming why a student feels a certain way.\n"
    "- Safety: If a student hints at acute distress or self-harm, express heartfelt care and encourage "
    "connecting with free 24/7 helplines: Tele-MANAS (14416 / 1800-891-4416) or Emergency (112)."
)


def is_wellness_relevant(
    user_message: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """Check if the student's message or immediate follow-up pertains to wellness, stress, mood, or routines."""
    if not user_message:
        return False

    msg_lower = user_message.lower()
    words = re.findall(r'[a-zA-Z]+', msg_lower)

    # Check for direct keyword overlap
    for word in words:
        if word in WELLNESS_RELEVANCE_KEYWORDS:
            return True

    # If the message is very short (e.g. "why?", "yeah", "tell me more"), inspect previous turn
    if len(words) <= 4 and conversation_history:
        for turn in reversed(conversation_history[-2:]):
            parts = turn.get("parts", [])
            for p in parts:
                text = p.get("text", "").lower() if isinstance(p, dict) else ""
                if any(k in text for k in WELLNESS_RELEVANCE_KEYWORDS):
                    return True

    return False


def get_student_wellness_summary(user_id: int) -> Optional[str]:
    """Retrieve recent wellness check-in records for authenticated user from database.

    Strictly reads real data:
    - Recent 1-3 MoodEntries (date & mood/valence)
    - Latest StressAssessment (date & category/score)
    - Active WellnessTask counts
    Never reads or exposes private free-text journal reflections.
    """
    try:
        parts: List[str] = []

        # 1. Recent Mood Check-ins
        recent_moods = (
            MoodEntry.query.filter_by(user_id=user_id)
            .order_by(MoodEntry.created_at.desc())
            .limit(2)
            .all()
        )
        if recent_moods:
            mood_strs = []
            for m in recent_moods:
                date_str = m.created_at.strftime("%b %d") if m.created_at else "recently"
                mood_strs.append(f"{m.mood} (valence {m.valence:+d}) on {date_str}")
            parts.append(f"Recent mood check-ins: {'; '.join(mood_strs)}.")

        # 2. Latest Stress Assessment
        latest_stress = (
            StressAssessment.query.filter_by(user_id=user_id)
            .order_by(StressAssessment.created_at.desc())
            .first()
        )
        if latest_stress:
            stress_date = (
                latest_stress.created_at.strftime("%b %d")
                if latest_stress.created_at
                else "recently"
            )
            parts.append(
                f"Latest stress assessment: {latest_stress.level} stress ({latest_stress.score}/44) on {stress_date}."
            )

        # 3. Active Tasks count
        active_tasks_count = (
            WellnessTask.query.filter_by(user_id=user_id, completed=False).count()
        )
        if active_tasks_count > 0:
            parts.append(f"Student currently has {active_tasks_count} active wellness/study task(s).")

        if parts:
            return " ".join(parts)
        return None
    except Exception as e:
        logger.warning("Could not assemble wellness summary for user %s: %s", user_id, type(e).__name__)
        return None


def get_student_memories(user_id: int) -> List[Dict[str, Any]]:
    """Retrieve persistent, non-sensitive preferences and facts for the student."""
    try:
        memories = (
            ChatMemory.query.filter_by(user_id=user_id)
            .order_by(ChatMemory.created_at.asc())
            .limit(10)
            .all()
        )
        return [{"id": m.id, "text": m.memory_text, "category": m.category} for m in memories]
    except Exception as e:
        logger.warning("Could not fetch memories for user %s: %s", user_id, type(e).__name__)
        return []


def is_memory_enabled_for_user(user: User) -> bool:
    """Check if the user has memory enabled in UserPreferences."""
    try:
        prefs = getattr(user, 'preferences', None)
        if prefs is not None:
            return getattr(prefs, 'chat_memory_enabled', True)
        return True
    except Exception:
        return True


def extract_candidate_memory(user_id: int, message: str) -> Optional[str]:
    """Inspect message for safe, non-sensitive student facts and preferences to store.

    Safety:
    - Rejects any message with crisis or sensitive health language.
    - Captures only explicit casual facts (major, preferred study time, hobbies).
    - Ignores clinical, trauma, or emotional disclosures.
    """
    if not message or len(message) < 10 or len(message) > 300:
        return None

    # Safety Screening
    if text_has_crisis_language(message):
        return None

    user_obj = db.session.get(User, user_id)
    if user_obj and not is_memory_enabled_for_user(user_obj):
        return None

    msg_lower = message.lower()
    for sensitive_term in SENSITIVE_MEMORY_PATTERNS:
        if sensitive_term in msg_lower:
            return None

    extracted_text = None
    category = "preference"

    # Pattern 1: Academic major / field
    m_major = re.search(
        r"\b(?:i am|i'm)(?:\s+a)?\s+(?:studying|majoring in|a student of)\s+([a-zA-Z\s]{2,40})",
        message,
        re.IGNORECASE,
    )
    if m_major:
        major = m_major.group(1).strip()
        if len(major) > 2 and not any(bad in major.lower() for bad in SENSITIVE_MEMORY_PATTERNS):
            extracted_text = f"Studies {major}"
            category = "academic"

    # Pattern 2: Preferred study/routine habits
    if not extracted_text:
        m_pref = re.search(
            r"\b(?:i prefer to|i prefer)\s+(study|work|wake up|sleep|focus)\s+([a-zA-Z\s]{2,40})",
            message,
            re.IGNORECASE,
        )
        if m_pref:
            action, detail = m_pref.group(1).strip(), m_pref.group(2).strip()
            extracted_text = f"Prefers to {action} {detail}"
            category = "routine"

    # Pattern 3: Name / Call me
    if not extracted_text:
        m_name = re.search(
            r"\b(?:my name is|call me|you can call me)\s+([A-Z][a-zA-Z]{1,25})\b",
            message,
        )
        if m_name:
            pref_name = m_name.group(1).strip()
            extracted_text = f"Prefers to be called {pref_name}"
            category = "identity"

    # Pattern 4: Hobbies / Interests
    if not extracted_text:
        m_hobby = re.search(
            r"\b(?:i really enjoy|i love to|my favorite hobby is)\s+([a-zA-Z\s]{2,40})",
            message,
            re.IGNORECASE,
        )
        if m_hobby:
            hobby = m_hobby.group(1).strip()
            if not any(bad in hobby.lower() for bad in SENSITIVE_MEMORY_PATTERNS):
                extracted_text = f"Enjoys {hobby}"
                category = "hobby"

    if not extracted_text:
        return None

    # Clean and cap
    clean_memory = extracted_text.strip()[:200]

    try:
        # Check if already exists
        exists = ChatMemory.query.filter_by(user_id=user_id, memory_text=clean_memory).first()
        if exists:
            return None

        # Check total count cap (keep at most 10)
        existing_count = ChatMemory.query.filter_by(user_id=user_id).count()
        if existing_count >= 10:
            oldest = (
                ChatMemory.query.filter_by(user_id=user_id)
                .order_by(ChatMemory.created_at.asc())
                .first()
            )
            if oldest:
                db.session.delete(oldest)

        new_mem = ChatMemory(
            user_id=user_id,
            memory_text=clean_memory,
            category=category,
        )
        db.session.add(new_mem)
        db.session.commit()
        logger.info("Saved persistent non-sensitive memory for user %s (category: %s)", user_id, category)
        return clean_memory
    except Exception as e:
        db.session.rollback()
        logger.warning("Could not save candidate memory: %s", type(e).__name__)
        return None


def build_dynamic_system_prompt(
    user: Optional[User] = None,
    user_message: str = "",
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build the dynamic, personalized system prompt for the Gemini companion.

    Includes:
    - Base personality and non-clinical guardrails.
    - Verified user profile (username, preferences) without fabrication.
    - Saved persistent non-sensitive memories (if memory is enabled).
    - Recent wellness context (if message is wellness-relevant).
    - Anti-overstepping directive for casual/academic queries.
    """
    sections = [BASE_COMPANION_PERSONA]

    if not user:
        # Unauthenticated / test fallback
        sections.append(
            "\nContext:\n- Interacting with a student user. Always remain supportive and respectful."
        )
        return "\n\n".join(sections)

    # 1. User Profile Section
    profile_lines = [f"- Student Username: {user.username}"]
    prefs = getattr(user, 'preferences', None)
    if prefs:
        if prefs.wellness_goals and isinstance(prefs.wellness_goals, list):
            goals_clean = [g.replace('_', ' ').title() for g in prefs.wellness_goals if isinstance(g, str)]
            if goals_clean:
                profile_lines.append(f"- Student Wellness Goals: {', '.join(goals_clean)}")
        if prefs.timezone and prefs.timezone != "UTC":
            profile_lines.append(f"- Local Timezone: {prefs.timezone}")

    profile_lines.append(
        "- Profile Policy: Never infer or invent personal details, contact info, or background not listed above."
    )
    sections.append("\nAuthenticated Student Profile:\n" + "\n".join(profile_lines))

    # 2. Persistent Memory Section (if enabled)
    if is_memory_enabled_for_user(user):
        memories = get_student_memories(user.id)
        if memories:
            mem_lines = [f"- {m['text']}" for m in memories]
            sections.append(
                "\nWhat You Remember About This Student (integrate naturally; do NOT recite as a bulleted list):\n"
                + "\n".join(mem_lines)
            )

    # 3. Wellness Context Section
    wellness_relevant = is_wellness_relevant(user_message, conversation_history)
    wellness_summary = get_student_wellness_summary(user.id)

    if wellness_summary:
        if wellness_relevant:
            sections.append(
                "\nRecent MindEase Wellness Context (User is discussing feelings, stress, or self-care):\n"
                f"{wellness_summary}\n"
                "Guidance: You may empathetically incorporate this context. Ask clarifying questions instead of "
                "assuming why their mood or stress changed. Never offer a diagnosis or clinical evaluation."
            )
        else:
            sections.append(
                "\nWellness Context Directive:\n"
                "The student's current message is on an everyday, casual, academic, or random topic. "
                "Do NOT volunteer or recite their past check-in scores, stress assessments, or mood history "
                "in this reply. Engage with their topic directly, warmly, and naturally."
            )

    return "\n\n".join(sections)
