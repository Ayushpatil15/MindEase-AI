"""Stress Assessment Forms."""

from flask_wtf import FlaskForm
from wtforms import RadioField, TextAreaField, SubmitField
from wtforms.validators import InputRequired, Length, Optional

# The 11 core fixed questions
QUESTIONS = [
    "In the last month, how often have you been upset because of something that happened unexpectedly?",
    "In the last month, how often have you felt that you were unable to control the important things in your life?",
    "In the last month, how often have you felt nervous and \"stressed\"?",
    "In the last month, how often have you felt confident about your ability to handle your personal problems?",
    "In the last month, how often have you felt that things were going your way?",
    "In the last month, how often have you found that you could not cope with all the things that you had to do?",
    "In the last month, how often have you been able to control irritations in your life?",
    "In the last month, how often have you felt that you were on top of things?",
    "In the last month, how often have you been angered because of things that were outside of your control?",
    "In the last month, how often have you felt difficulties were piling up so high that you could not overcome them?",
    "In the last month, how often have you felt overwhelmed by your academic workload?"
]

CHOICES = [
    (0, 'Never'),
    (1, 'Almost never'),
    (2, 'Sometimes'),
    (3, 'Fairly often'),
    (4, 'Very often')
]

class StressAssessmentForm(FlaskForm):
    """Form for the 11-question Stress Assessment."""
    
    q1 = RadioField(QUESTIONS[0], choices=CHOICES, coerce=int, validators=[InputRequired()])
    q2 = RadioField(QUESTIONS[1], choices=CHOICES, coerce=int, validators=[InputRequired()])
    q3 = RadioField(QUESTIONS[2], choices=CHOICES, coerce=int, validators=[InputRequired()])
    q4 = RadioField(QUESTIONS[3], choices=CHOICES, coerce=int, validators=[InputRequired()])
    q5 = RadioField(QUESTIONS[4], choices=CHOICES, coerce=int, validators=[InputRequired()])
    q6 = RadioField(QUESTIONS[5], choices=CHOICES, coerce=int, validators=[InputRequired()])
    q7 = RadioField(QUESTIONS[6], choices=CHOICES, coerce=int, validators=[InputRequired()])
    q8 = RadioField(QUESTIONS[7], choices=CHOICES, coerce=int, validators=[InputRequired()])
    q9 = RadioField(QUESTIONS[8], choices=CHOICES, coerce=int, validators=[InputRequired()])
    q10 = RadioField(QUESTIONS[9], choices=CHOICES, coerce=int, validators=[InputRequired()])
    q11 = RadioField(QUESTIONS[10], choices=CHOICES, coerce=int, validators=[InputRequired()])
    
    reflection = TextAreaField(
        'Private Reflection (Optional)',
        validators=[Optional(), Length(max=5000)],
        description='Optional private notes about how you are feeling.'
    )
    
    submit = SubmitField('Submit Assessment')
