"""Stress Assessment Routes."""

from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_required, current_user
from werkzeug.exceptions import abort

from app.extensions import db
from app.forms.stress import StressAssessmentForm
from app.services.stress_service import save_assessment, get_history
from app.models.stress import StressAssessment
from app.utils.validators import validate_stress_responses, validate_journal_note
from app.utils.decorators import student_required

stress_bp = Blueprint('stress', __name__, url_prefix='/stress')


@stress_bp.route('/', methods=['GET', 'POST'])
@login_required
@student_required
def take_assessment():
    """Render and handle the Stress Assessment."""
    form = StressAssessmentForm()

    if form.validate_on_submit():
        # 1. Extract responses explicitly
        raw_responses = [
            form.q1.data, form.q2.data, form.q3.data, form.q4.data,
            form.q5.data, form.q6.data, form.q7.data, form.q8.data,
            form.q9.data, form.q10.data, form.q11.data
        ]
        
        # 2. Server-side validation boundary
        validated_responses, error = validate_stress_responses(raw_responses)
        if error:
            flash(f"Validation Error: {error}", "error")
            return render_template('stress/assessment.html', form=form)

        validated_reflection, ref_error = validate_journal_note(form.reflection.data)
        if ref_error:
            flash(f"Validation Error: {ref_error}", "error")
            return render_template('stress/assessment.html', form=form)

        # 3. Pass to service layer for reverse scoring and persistence
        assessment = save_assessment(
            user_id=current_user.id,
            responses=validated_responses,
            reflection=validated_reflection
        )

        flash('Stress assessment submitted successfully.', 'success')
        return redirect(url_for('stress.result', id=assessment.id))

    return render_template('stress/assessment.html', form=form)


@stress_bp.route('/result/<int:id>', methods=['GET'])
@login_required
@student_required
def result(id):
    """View a specific stress assessment result."""
    assessment = db.get_or_404(StressAssessment, id)
    
    # Ownership protection
    if assessment.user_id != current_user.id:
        abort(403)

    return render_template('stress/result.html', assessment=assessment)


@stress_bp.route('/history', methods=['GET'])
@login_required
@student_required
def history():
    """View stress assessment history."""
    assessments = get_history(current_user.id)
    return render_template('stress/history.html', assessments=assessments)
