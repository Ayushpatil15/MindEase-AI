"""Activity Registry — Structured guided wellness activity content definitions.

MindEase AI Phase 7 guided wellness activities.
All activities are non-clinical, self-reflection/wellness focused tools.
"""

ACTIVITIES = {
    'box_breathing': {
        'id': 'box_breathing',
        'category': 'Mindfulness',
        'title': 'Box Breathing & Grounding Exercise',
        'short_description': 'Practice 4-7-8 deep breathing for 5 minutes to lower physiological arousal.',
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
                'instruction': 'Hold your breath comfortably for 7 counts without straining.',
            },
            {
                'step_number': 4,
                'heading': 'Exhale',
                'instruction': 'Exhale slowly through your mouth for 8 counts, releasing all body tension.',
            },
            {
                'step_number': 5,
                'heading': 'Repeat',
                'instruction': 'Continue this 4-7-8 breathing pattern until the 5-minute timer completes.',
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
                'instruction': 'Look around your room and name 5 distinct objects out loud or in your head.',
            },
            {
                'step_number': 2,
                'heading': '4 Things You Can Touch',
                'instruction': 'Notice 4 physical sensations (e.g., your feet on the floor, texture of your desk).',
            },
            {
                'step_number': 3,
                'heading': '3 Things You Can Hear',
                'instruction': 'Listen quietly and identify 3 ambient sounds around you (e.g., clock, wind, footsteps).',
            },
            {
                'step_number': 4,
                'heading': '2 Things You Can Smell',
                'instruction': 'Identify 2 scents nearby, or recall 2 pleasant smells you enjoy.',
            },
            {
                'step_number': 5,
                'heading': '1 Thing You Can Taste',
                'instruction': 'Focus on 1 taste in your mouth right now or take a sip of fresh water.',
            },
        ],
        'completion_guidance': 'Take one deep breath and bring your awareness back to the present moment.',
        'safety_note': 'Grounding helps anchor attention during moments of busyness or stress.',
    },
    'pomodoro_focus': {
        'id': 'pomodoro_focus',
        'category': 'Academic',
        'title': 'Pomodoro Focus Session',
        'short_description': 'Work with complete focus for 25 minutes, followed by a mandatory 5-minute break.',
        'duration_minutes': 25,
        'has_timer': True,
        'timer_seconds': 1500,
        'steps': [
            {
                'step_number': 1,
                'heading': 'Set Target Task',
                'instruction': 'Choose one clear, specific academic task to focus on during this 25-minute block.',
            },
            {
                'step_number': 2,
                'heading': 'Eliminate Distractions',
                'instruction': 'Close unrelated browser tabs, silence your phone, and start the 25-minute timer.',
            },
            {
                'step_number': 3,
                'heading': 'Single-Task Focus',
                'instruction': 'Work continuously on your target task until the timer alerts you.',
            },
            {
                'step_number': 4,
                'heading': 'Take a 5-Minute Break',
                'instruction': 'When the timer finishes, stand up, stretch, and rest your brain before returning.',
            },
        ],
        'completion_guidance': 'Acknowledge the progress made during this dedicated focus interval.',
        'safety_note': 'Regular breaks prevent mental fatigue and maintain consistent study output.',
    },
    'task_chunking': {
        'id': 'task_chunking',
        'category': 'Academic',
        'title': 'Structured Task Chunking',
        'short_description': 'Break overwhelming assignments into 3 small, manageable micro-goals.',
        'duration_minutes': 10,
        'has_timer': False,
        'timer_seconds': None,
        'steps': [
            {
                'step_number': 1,
                'heading': 'Identify the Main Task',
                'instruction': 'Write down the name of the assignment or project that feels overwhelming.',
            },
            {
                'step_number': 2,
                'heading': 'Break into 3 Micro-Steps',
                'instruction': 'Divide the task into 3 small steps that each take under 15 minutes to finish.',
            },
            {
                'step_number': 3,
                'heading': 'Select Step 1',
                'instruction': 'Choose the easiest or smallest micro-step to start with right away.',
            },
            {
                'step_number': 4,
                'heading': 'Schedule Step 1',
                'instruction': 'Commit to executing Step 1 immediately or in your next study block.',
            },
        ],
        'completion_guidance': 'Focus only on the first micro-step rather than the entire project.',
        'safety_note': 'Chunking reduces cognitive overload and helps overcome task paralysis.',
    },
    'movement_break': {
        'id': 'movement_break',
        'category': 'Physical',
        'title': 'Hydration & Movement Reset',
        'short_description': 'Drink a glass of water and complete a gentle 5-minute physical stretch.',
        'duration_minutes': 5,
        'has_timer': False,
        'timer_seconds': None,
        'steps': [
            {
                'step_number': 1,
                'heading': 'Hydrate',
                'instruction': 'Drink a full glass of fresh water to restore physical hydration.',
            },
            {
                'step_number': 2,
                'heading': 'Shoulder & Neck Stretch',
                'instruction': 'Roll your shoulders backward 5 times and gently tilt your head side to side.',
            },
            {
                'step_number': 3,
                'heading': 'Standing Stretch',
                'instruction': 'Stand up, reach both arms up toward the ceiling, and hold for 15 seconds.',
            },
            {
                'step_number': 4,
                'heading': 'Wrist & Hand Relief',
                'instruction': 'Extend your arms forward, gently flexing wrists and fingers to release typing tension.',
            },
        ],
        'completion_guidance': 'Notice the boost in blood flow and physical energy.',
        'safety_note': 'Gentle movement helps offset sedentary study postures.',
    },
    'tension_release': {
        'id': 'tension_release',
        'category': 'Physical',
        'title': 'Progressive Muscle Tension Release',
        'short_description': 'Systematically tense and release shoulder, neck, and facial muscles.',
        'duration_minutes': 5,
        'has_timer': False,
        'timer_seconds': None,
        'steps': [
            {
                'step_number': 1,
                'heading': 'Facial Muscle Release',
                'instruction': 'Squeeze facial muscles tight for 5 seconds, then completely relax jaw and forehead.',
            },
            {
                'step_number': 2,
                'heading': 'Shoulder Shrugs',
                'instruction': 'Lift shoulders toward ears, hold tension for 5 seconds, then drop shoulders completely.',
            },
            {
                'step_number': 3,
                'heading': 'Hand & Arm Clench',
                'instruction': 'Make tight fists for 5 seconds, then unfurl fingers and feel warmth in hands.',
            },
            {
                'step_number': 4,
                'heading': 'Full Body Exhale',
                'instruction': 'Take one deep breath and release all lingering tension with a long sigh out.',
            },
        ],
        'completion_guidance': 'Compare the physical sensation of tension versus relaxation in your body.',
        'safety_note': 'Perform stretches gently without pushing into discomfort or pain.',
    },
    'screen_break': {
        'id': 'screen_break',
        'category': 'Physical',
        'title': '20-20-20 Screen Rest',
        'short_description': 'Reduce visual fatigue by looking 20 feet away for 20 seconds.',
        'duration_minutes': 3,
        'has_timer': True,
        'timer_seconds': 180,
        'steps': [
            {
                'step_number': 1,
                'heading': 'Look Away',
                'instruction': 'Shift your eyes away from your monitor, phone, or laptop screen.',
            },
            {
                'step_number': 2,
                'heading': 'Focus Distantly',
                'instruction': 'Focus on an object at least 20 feet away (such as a window view or wall art).',
            },
            {
                'step_number': 3,
                'heading': 'Blink Slowly',
                'instruction': 'Blink slowly 5 times to re-moisten your eyes.',
            },
            {
                'step_number': 4,
                'heading': 'Rest Eyelids',
                'instruction': 'Optionally close your eyes for 30 seconds before returning to work.',
            },
        ],
        'completion_guidance': 'Visual breaks protect against digital eye strain during long study sessions.',
        'safety_note': 'Repeat the 20-20-20 rule every hour when using digital displays.',
    },
    'journal_reflection': {
        'id': 'journal_reflection',
        'category': 'Reflection',
        'title': '5-Minute Guided Journal Reflection',
        'short_description': 'Write down your current thoughts, wins, or stressors without judgment.',
        'duration_minutes': 5,
        'has_timer': False,
        'timer_seconds': None,
        'steps': [
            {
                'step_number': 1,
                'heading': 'Check In',
                'instruction': 'Pause and ask yourself: "How am I feeling right now mentally and physically?"',
            },
            {
                'step_number': 2,
                'heading': 'Write Freely',
                'instruction': 'Jot down your current thoughts, worries, or small accomplishments for 3-5 minutes.',
            },
            {
                'step_number': 3,
                'heading': 'Identify One Action',
                'instruction': 'Note one small positive step you can take today for your own wellbeing.',
            },
        ],
        'completion_guidance': 'Reflective writing helps organize thoughts and reduce mental clutter.',
        'safety_note': 'Your reflection notes remain private to your student account.',
    },
    'self_compassion': {
        'id': 'self_compassion',
        'category': 'Reflection',
        'title': 'Self-Compassion & Positive Framing',
        'short_description': 'Acknowledge daily effort and treat yourself with kindness during setbacks.',
        'duration_minutes': 5,
        'has_timer': False,
        'timer_seconds': None,
        'steps': [
            {
                'step_number': 1,
                'heading': 'Acknowledge Difficulty',
                'instruction': 'Silently acknowledge: "This is a challenging moment, and it is natural to feel stressed."',
            },
            {
                'step_number': 2,
                'heading': 'Recognize Shared Humanity',
                'instruction': 'Remind yourself: "Many students experience similar feelings and challenges."',
            },
            {
                'step_number': 3,
                'heading': 'Offer Kindness',
                'instruction': 'Speak to yourself with the same warmth you would offer to a close friend.',
            },
            {
                'step_number': 4,
                'heading': 'Note 1 Positive Effort',
                'instruction': 'Recognize one effort you made today, regardless of the outcome.',
            },
        ],
        'completion_guidance': 'Practicing self-compassion builds emotional resilience during academic pressure.',
        'safety_note': 'Be gentle with yourself — progress is built through consistent small steps.',
    },
}


def get_all_activities():
    """Return all activity definitions as a list."""
    return list(ACTIVITIES.values())


def get_activity_by_id(activity_id):
    """Retrieve an activity definition by its unique identifier.

    Args:
        activity_id (str): Unique slug (e.g. 'box_breathing').

    Returns:
        dict | None: The activity dictionary, or None if not found.
    """
    return ACTIVITIES.get(activity_id)


def get_activities_by_category(category):
    """Filter activities by category name."""
    return [a for a in ACTIVITIES.values() if a['category'].lower() == category.lower()]
