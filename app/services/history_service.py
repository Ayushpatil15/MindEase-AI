"""History and Insights Service.

Provides windowed historical data aggregation (7d, 30d, 90d) and deterministic,
non-clinical insight generation based strictly on persisted database records.
"""

from datetime import datetime, timedelta, timezone
from collections import Counter

from app.models import MoodEntry, StressAssessment, EmotionScan, WellnessTask


TIMEFRAME_DAYS = {
    '7d': 7,
    '30d': 30,
    '90d': 90,
}


def parse_timeframe(timeframe_str):
    """Parse and validate requested timeframe string.

    Args:
        timeframe_str (str): Requested window string ('7d', '30d', '90d').

    Returns:
        tuple[str, int]: Canonical timeframe string and integer days.
    """
    key = str(timeframe_str).lower().strip() if timeframe_str else '7d'
    if key not in TIMEFRAME_DAYS:
        key = '7d'
    return key, TIMEFRAME_DAYS[key]


def get_wellness_history_data(user_id, timeframe='7d'):
    """Aggregate windowed historical data for a student user.

    Args:
        user_id (int): Student user ID.
        timeframe (str): Time window ('7d', '30d', '90d').

    Returns:
        dict: Aggregated historical data dictionary with mood, stress, emotion,
              activities, consistency, and insights.
    """
    tf_key, tf_days = parse_timeframe(timeframe)
    now = datetime.now(timezone.utc)
    start_date = now - timedelta(days=tf_days)

    # 1. Fetch Mood Entries in timeframe (chronological order)
    mood_records = (
        MoodEntry.query
        .filter(
            MoodEntry.user_id == user_id,
            MoodEntry.created_at >= start_date,
        )
        .order_by(MoodEntry.created_at.asc())
        .all()
    )

    # 2. Fetch Stress Assessments in timeframe (chronological order)
    stress_records = (
        StressAssessment.query
        .filter(
            StressAssessment.user_id == user_id,
            StressAssessment.created_at >= start_date,
        )
        .order_by(StressAssessment.created_at.asc())
        .all()
    )

    # 3. Fetch Emotion Scans in timeframe (chronological order)
    emotion_records = (
        EmotionScan.query
        .filter(
            EmotionScan.user_id == user_id,
            EmotionScan.created_at >= start_date,
        )
        .order_by(EmotionScan.created_at.asc())
        .all()
    )

    # 4. Fetch Completed Wellness Tasks in timeframe (chronological order)
    task_records = (
        WellnessTask.query
        .filter(
            WellnessTask.user_id == user_id,
            WellnessTask.completed.is_(True),
            WellnessTask.completed_at >= start_date,
        )
        .order_by(WellnessTask.completed_at.asc())
        .all()
    )

    # --- Process Mood Data ---
    mood_count = len(mood_records)
    avg_valence = (
        round(sum(m.valence for m in mood_records) / mood_count, 2)
        if mood_count > 0 else None
    )
    mood_trend = [
        {
            'id': m.id,
            'date': m.created_at.strftime('%Y-%m-%d'),
            'mood': m.mood,
            'valence': m.valence,
            'created_at': m.created_at.strftime('%Y-%m-%d %H:%M UTC'),
        }
        for m in mood_records
    ]

    # --- Process Stress Data ---
    stress_count = len(stress_records)
    stress_assessments = [
        {
            'id': s.id,
            'date': s.created_at.strftime('%Y-%m-%d'),
            'score': s.score,
            'level': s.level,
            'created_at': s.created_at.strftime('%Y-%m-%d %H:%M UTC'),
        }
        for s in stress_records
    ]

    # --- Process Emotion Data ---
    emotion_count = len(emotion_records)
    emotion_counts = Counter(e.emotion for e in emotion_records)
    emotion_scans = [
        {
            'id': e.id,
            'date': e.created_at.strftime('%Y-%m-%d'),
            'emotion': e.emotion,
            'confidence': round(e.confidence, 4),
            'created_at': e.created_at.strftime('%Y-%m-%d %H:%M UTC'),
        }
        for e in emotion_records
    ]

    # --- Process Activity & Task Data ---
    completed_task_count = len(task_records)
    activity_completions = [
        {
            'id': t.id,
            'activity_id': t.activity_id,
            'title': t.title,
            'completed_at': t.completed_at.strftime('%Y-%m-%d %H:%M UTC') if t.completed_at else None,
        }
        for t in task_records
    ]

    # --- Calculate Consistency ---
    active_dates = set()
    for m in mood_records:
        active_dates.add(m.created_at.date())
    for s in stress_records:
        active_dates.add(s.created_at.date())
    for e in emotion_records:
        active_dates.add(e.created_at.date())
    for t in task_records:
        if t.completed_at:
            active_dates.add(t.completed_at.date())

    active_days_count = len(active_dates)
    consistency_percentage = round((active_days_count / tf_days) * 100, 1)

    raw_data = {
        'timeframe': {
            'range': tf_key,
            'days': tf_days,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': now.strftime('%Y-%m-%d'),
        },
        'mood': {
            'count': mood_count,
            'average_valence': avg_valence,
            'trend': mood_trend,
        },
        'stress': {
            'count': stress_count,
            'assessments': stress_assessments,
        },
        'emotion': {
            'count': emotion_count,
            'distribution': dict(emotion_counts),
            'scans': emotion_scans,
        },
        'activities': {
            'completed_count': completed_task_count,
            'completions': activity_completions,
        },
        'consistency': {
            'active_days': active_days_count,
            'total_days': tf_days,
            'percentage': consistency_percentage,
        },
    }

    # Generate insights strictly from deterministic metrics
    raw_data['insights'] = generate_deterministic_insights(raw_data)

    return raw_data


def generate_deterministic_insights(data):
    """Generate human-readable, safe, non-clinical insights from aggregated metrics.

    Args:
        data (dict): Aggregated historical data payload.

    Returns:
        list[str]: List of deterministic insight strings.
    """
    insights = []
    tf_days = data['timeframe']['days']

    # 1. Consistency Insights
    active_days = data['consistency']['active_days']
    if active_days == 0:
        insights.append("No check-ins recorded for this timeframe yet.")
    elif active_days >= max(3, int(tf_days * 0.5)):
        insights.append(
            f"High check-in consistency! You logged check-ins on {active_days} of the last {tf_days} days."
        )
    else:
        insights.append(
            f"You recorded check-ins on {active_days} of the last {tf_days} days."
        )

    # 2. Mood Pattern Insights
    mood_count = data['mood']['count']
    if mood_count >= 2:
        trend = data['mood']['trend']
        pos_count = sum(1 for m in trend if m['valence'] > 0)
        low_count = sum(1 for m in trend if m['valence'] < 0)
        if pos_count / mood_count >= 0.6:
            insights.append("Your recent mood check-ins were predominantly positive ('Good' or 'Very good').")
        elif low_count / mood_count >= 0.6:
            insights.append("You recorded several lower mood check-ins recently. Consider taking a break with a guided activity.")
        else:
            insights.append("Your recorded moods show a balanced mix across this timeframe.")
    elif mood_count == 1:
        single_mood = data['mood']['trend'][0]['mood']
        insights.append(f"1 mood check-in recorded ('{single_mood}'). Record more check-ins to view mood trends.")

    # 3. Stress Assessment Insights
    stress_count = data['stress']['count']
    if stress_count >= 2:
        assessments = data['stress']['assessments']
        latest = assessments[-1]
        prev = assessments[-2]
        if latest['score'] < prev['score']:
            insights.append(
                f"Your latest stress score ({latest['score']}/44) showed an improvement compared to your previous assessment ({prev['score']}/44)."
            )
        elif latest['score'] > prev['score']:
            insights.append(
                f"Your latest stress score ({latest['score']}/44) was higher than your previous assessment ({prev['score']}/44)."
            )
        else:
            insights.append(
                f"Your stress assessment scores remained steady at {latest['score']}/44 ({latest['level']} Stress)."
            )
    elif stress_count == 1:
        latest = data['stress']['assessments'][0]
        insights.append(
            f"1 stress assessment completed ({latest['score']}/44, {latest['level']} Stress)."
        )

    # 4. Activity Completions Insights
    completed_count = data['activities']['completed_count']
    if completed_count >= 1:
        insights.append(
            f"You completed {completed_count} wellness activity session{'s' if completed_count != 1 else ''} in this timeframe."
        )

    # 5. Facial Expression Scan Insights (strictly non-clinical)
    emotion_count = data['emotion']['count']
    if emotion_count >= 1:
        distribution = data['emotion']['distribution']
        most_common = max(distribution, key=distribution.get)
        insights.append(
            f"Facial-expression check-ins: most frequent detection was '{most_common}' across {emotion_count} scan{'s' if emotion_count != 1 else ''}."
        )

    return insights
