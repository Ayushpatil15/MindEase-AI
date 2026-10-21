"""Activity Service — Structured guided wellness activity content definitions & queries.

MindEase AI Guided Wellness Activities.
All activities are non-clinical, self-reflection/wellness focused tools.
Final set of 6 active activities:
1. box_breathing (Mindfulness, 5 min)
2. grounding_54321 (Mindfulness, 5 min)
3. pomodoro_focus (Academic, 30 min)
4. task_chunking (Academic, 5 min)
5. self_compassion (Reflection, 5 min)
6. stretch_reset (Physical, 5 min - replaces movement_break, tension_release, screen_break)

Retired activities (movement_break, tension_release, screen_break, journal_reflection)
are resolved at read-time to surviving replacements.
"""

# Surviving 6 interactive wellness activities
ACTIVITIES = {
    'box_breathing': {
        'id': 'box_breathing',
        'category': 'Mindfulness',
        'title': 'Box Breathing',
        'short_description': 'Practice box breathing (inhale 4, hold 4, exhale 4, hold 4) for 5 minutes to lower physiological arousal.',
        'duration_minutes': 5,
        'has_timer': True,
        'timer_seconds': 300,
        'steps': [
            {
                'step_number': 1,
                'heading': 'Prepare',
                'instruction': 'Sit comfortably in a quiet space with your feet flat on the floor and shoulders relaxed.',
            },
            {
                'step_number': 2,
                'heading': 'Inhale',
                'instruction': 'Breathe in slowly through your nose for 4 counts, filling your lungs evenly.',
            },
            {
                'step_number': 3,
                'heading': 'Hold',
                'instruction': 'Hold your breath comfortably for 4 counts without straining.',
            },
            {
                'step_number': 4,
                'heading': 'Exhale',
                'instruction': 'Exhale slowly through your mouth for 4 counts, releasing all body tension.',
            },
            {
                'step_number': 5,
                'heading': 'Hold',
                'instruction': 'Hold your breath empty for 4 counts before the next cycle.',
            },
            {
                'step_number': 6,
                'heading': 'Repeat',
                'instruction': 'Continue this box breathing pattern (inhale 4, hold 4, exhale 4, hold 4) until the 5-minute timer completes.',
            },
        ],
        'completion_guidance': 'Notice if your heart rate has slowed or if your muscle tension feels reduced.',
        'safety_note': 'This exercise is intended for general relaxation. If you feel dizzy, pause and resume normal breathing.',
    },
    'grounding_54321': {
        'id': 'grounding_54321',
        'category': 'Mindfulness',
        'title': '5-4-3-2-1 Sensory Grounding',
        'short_description': 'Re-center your mind by noticing 5 things you see, 4 you feel, 3 you hear, 2 you smell, and 1 you taste.',
        'duration_minutes': 5,
        'has_timer': False,
        'timer_seconds': None,
        'steps': [
            {
                'step_number': 1,
                'heading': '5 Things You Can See',
                'instruction': 'Look around your environment. Tap to identify 5 distinct objects, patterns, or colors.',
                'target_count': 5,
                'sense': 'sight',
            },
            {
                'step_number': 2,
                'heading': '4 Things You Can Physically Feel',
                'instruction': 'Focus on physical contact. Tap to notice 4 sensations, such as feet on the floor, texture of clothing, or air temperature.',
                'target_count': 4,
                'sense': 'touch',
            },
            {
                'step_number': 3,
                'heading': '3 Things You Can Hear',
                'instruction': 'Listen attentively to your background. Tap to notice 3 ambient sounds, near or distant.',
                'target_count': 3,
                'sense': 'sound',
            },
            {
                'step_number': 4,
                'heading': '2 Things You Can Smell',
                'instruction': 'Notice scents around you (fresh air, coffee, books) or recall 2 calming scents.',
                'target_count': 2,
                'sense': 'smell',
            },
            {
                'step_number': 5,
                'heading': '1 Thing You Can Taste',
                'instruction': 'Notice the current taste in your mouth, or take a sip of water and notice the sensation.',
                'target_count': 1,
                'sense': 'taste',
            },
        ],
        'completion_guidance': 'Check in with your body. Grounding helps anchor attention in the present moment.',
        'safety_note': 'Focus on whatever your senses naturally observe without judging your surroundings.',
    },
    'pomodoro_focus': {
        'id': 'pomodoro_focus',
        'category': 'Academic',
        'title': 'Pomodoro Focus Session',
        'short_description': 'Structured 25-minute deep focus interval followed by an automatic 5-minute restorative break.',
        'duration_minutes': 30,
        'has_timer': True,
        'timer_seconds': 1500,  # 25 min focus
        'break_seconds': 300,   # 5 min break
        'steps': [
            {
                'step_number': 1,
                'heading': 'Set Single Focus Objective',
                'instruction': 'Choose one clear, manageable task for this block. Remove notifications and visual distractions.',
            },
            {
                'step_number': 2,
                'heading': 'Deep Focus Block (25 min)',
                'instruction': 'Work continuously on your chosen task until the focus interval completes.',
            },
            {
                'step_number': 3,
                'heading': 'Restorative Break (5 min)',
                'instruction': 'Step away from your screen, stretch, drink water, and let your mind recharge.',
            },
        ],
        'completion_guidance': 'Acknowledge your focused effort. Regular study intervals with breaks prevent mental fatigue.',
        'safety_note': 'Honor the 5-minute break completely to sustain cognitive energy for subsequent sessions.',
    },
    'task_chunking': {
        'id': 'task_chunking',
        'category': 'Academic',
        'title': 'Task Chunking & Micro-Steps',
        'short_description': 'Deconstruct an overwhelming project into 3 small, immediately actionable micro-tasks.',
        'duration_minutes': 5,
        'has_timer': False,
        'timer_seconds': None,
        'steps': [
            {
                'step_number': 1,
                'heading': 'Define the Main Task',
                'instruction': 'State the larger project or academic assignment that feels overwhelming.',
            },
            {
                'step_number': 2,
                'heading': 'Micro-Step 1: First 5-Minute Action',
                'instruction': 'Define the very first small physical step you can complete in under 5 minutes.',
            },
            {
                'step_number': 3,
                'heading': 'Micro-Step 2: Immediate Follow-Up',
                'instruction': 'Define the logical next action that builds on the momentum of step 1.',
            },
            {
                'step_number': 4,
                'heading': 'Micro-Step 3: Milestone Step',
                'instruction': 'Define the third small step that brings you to a clear stopping point.',
            },
        ],
        'completion_guidance': 'Focus only on completing the first micro-step today rather than the entire project.',
        'safety_note': 'Chunking reduces cognitive overload and helps overcome task paralysis.',
    },
    'self_compassion': {
        'id': 'self_compassion',
        'category': 'Reflection',
        'title': 'Self-Compassion Pause',
        'short_description': 'Acknowledge difficulty, recognize shared student experience, and offer yourself patient encouragement.',
        'duration_minutes': 5,
        'has_timer': False,
        'timer_seconds': None,
        'steps': [
            {
                'step_number': 1,
                'heading': 'Mindful Awareness of Stress',
                'instruction': 'Notice what feels heavy or challenging right now. Give yourself permission to acknowledge this stress without judgment.',
                'prompt': 'What feels difficult or demanding in this moment?',
            },
            {
                'step_number': 2,
                'heading': 'Recognizing Shared Humanity',
                'instruction': 'Remind yourself that struggle and self-doubt are normal parts of student life. You are not alone in navigating pressure.',
                'prompt': 'What would you say to a friend facing this same pressure?',
            },
            {
                'step_number': 3,
                'heading': 'Offering Self-Kindness',
                'instruction': 'Choose one encouraging, patient statement for yourself. Acknowledge your daily effort regardless of the outcome.',
                'prompt': 'What kind, realistic thought can support you right now?',
            },
        ],
        'completion_guidance': 'Practicing self-compassion calms emotional reactivity and builds long-term resilience.',
        'safety_note': 'Be gentle with yourself — progress is built through patience and small consistent steps.',
    },
    'stretch_reset': {
        'id': 'stretch_reset',
        'category': 'Physical',
        'title': 'Physical Stretch & Eye Reset',
        'short_description': 'Gentle timed upper-body stretches followed by a 20-second eye fatigue break.',
        'duration_minutes': 5,
        'has_timer': True,
        'timer_seconds': 300,
        'steps': [
            {
                'step_number': 1,
                'heading': 'Neck & Shoulder Release',
                'instruction': 'Gently tilt your head toward each shoulder for 15 seconds, then slowly roll shoulders backward.',
                'step_seconds': 30,
            },
            {
                'step_number': 2,
                'heading': 'Upper Back & Chest Expansion',
                'instruction': 'Interlace fingers behind your back or reach arms overhead. Open your chest and take deep breaths.',
                'step_seconds': 30,
            },
            {
                'step_number': 3,
                'heading': 'Wrist & Forearm Stretch',
                'instruction': 'Extend one arm forward, palm up, and gently draw fingers back with the opposite hand. Switch sides.',
                'step_seconds': 30,
            },
            {
                'step_number': 4,
                'heading': '20-20-20 Eye Rest',
                'instruction': 'Look away from all screens. Focus on an object at least 20 feet away for 20 seconds to relax eye muscles.',
                'step_seconds': 20,
            },
            {
                'step_number': 5,
                'heading': 'Hydration & Posture Reset',
                'instruction': 'Take a sip of water, sit or stand tall, and release any lingering tension with a long exhale.',
                'step_seconds': 30,
            },
        ],
        'completion_guidance': 'Notice the renewed circulation, relaxed posture, and reduced digital eye fatigue.',
        'safety_note': 'Perform all movements gently within your comfortable range of motion. Never force a stretch.',
    },
}

# Old-to-new activity mapping table for retired activities
RETIRED_ACTIVITIES_MAP = {
    'movement_break': 'stretch_reset',
    'tension_release': 'stretch_reset',
    'screen_break': 'stretch_reset',
    'journal_reflection': 'self_compassion',
}


def is_retired_activity(activity_id):
    """Check if an activity ID has been retired."""
    return activity_id in RETIRED_ACTIVITIES_MAP


def get_retired_activity_mapping():
    """Return dictionary of retired activity mappings."""
    return dict(RETIRED_ACTIVITIES_MAP)


def get_all_activities():
    """Return all active activity definitions as a list."""
    return list(ACTIVITIES.values())


def get_activity_by_id(activity_id, resolve_retired=True):
    """Retrieve an activity definition by its unique identifier.

    Args:
        activity_id (str): Unique slug (e.g. 'box_breathing').
        resolve_retired (bool): If True, maps retired activities to their surviving replacement.

    Returns:
        dict | None: The activity dictionary, or None if not found.
    """
    if not activity_id:
        return None
    if activity_id in ACTIVITIES:
        return ACTIVITIES[activity_id]
    if resolve_retired and activity_id in RETIRED_ACTIVITIES_MAP:
        target_id = RETIRED_ACTIVITIES_MAP[activity_id]
        if target_id in ACTIVITIES:
            resolved = dict(ACTIVITIES[target_id])
            resolved['is_retired_replacement'] = True
            resolved['retired_original_id'] = activity_id
            return resolved
    return None


def get_activities_by_category(category):
    """Filter active activities by category name."""
    return [a for a in ACTIVITIES.values() if a['category'].lower() == category.lower()]
