"""Current Mood Check-In Routes."""

import json
from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_required, current_user

from app.forms.mood import MoodCheckinForm
from app.services.mood_service import record_checkin, get_7day_trend, get_history
from app.services.recommendation_service import refresh_recommendations
from app.utils.validators import validate_mood_submission
from app.utils.decorators import student_required

mood_bp = Blueprint('mood', __name__, url_prefix='/mood')


@mood_bp.route('/', methods=['GET', 'POST'])
@login_required
@student_required
def checkin():
    """Render and handle the Current Mood Check-In."""
    form = MoodCheckinForm()

    if form.validate_on_submit():
        raw_mood = form.mood.data
        raw_note = form.journal_note.data

        # Server-side validation boundary
        mood_choice, safe_note, error = validate_mood_submission(raw_mood, raw_note)
        if error:
            flash(f"Validation Error: {error}", "error")
            return render_template('mood/checkin.html', form=form)

        # Service explicitly derives valence based on mood choice.
        # Client-submitted valence is impossible and ignored.
        record_checkin(
            user_id=current_user.id,
            mood=mood_choice,
            journal_note=safe_note
        )
        refresh_recommendations(current_user.id)

        flash('Mood check-in saved successfully.', 'success')
        return redirect(url_for('student.dashboard'))

    return render_template('mood/checkin.html', form=form)


@mood_bp.route('/history', methods=['GET'])
@login_required
@student_required
def history():
    """View mood history and 7-day trend."""
    # Only fetches current user's history from the real database
    trend_data = get_7day_trend(current_user.id)
    history_entries = get_history(current_user.id)
    
    # Serialize trend data for Chart.js
    trend_json = json.dumps(trend_data)
    
    return render_template(
        'mood/history.html', 
        entries=history_entries,
        trend_json=trend_json
    )
