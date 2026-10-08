"""Authentication and onboarding forms using Flask-WTF."""

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SelectMultipleField, SubmitField
from wtforms.validators import DataRequired, Length, Email, EqualTo, Regexp, Optional

from app.models.user_preferences import WELLNESS_GOALS


class RegistrationForm(FlaskForm):
    username = StringField(
        'Username',
        validators=[DataRequired(), Length(min=3, max=64)],
        description='Letters, digits, or underscores. Min 3 chars.'
    )
    email = StringField(
        'Email',
        validators=[DataRequired(), Email(), Length(max=120)]
    )
    password = PasswordField(
        'Password',
        validators=[DataRequired(), Length(min=8)]
    )
    confirm_password = PasswordField(
        'Confirm Password',
        validators=[DataRequired(), EqualTo('password', message='Passwords must match.')]
    )
    submit = SubmitField('Sign Up')


class LoginForm(FlaskForm):
    email = StringField(
        'Email',
        validators=[DataRequired(), Email()]
    )
    password = PasswordField(
        'Password',
        validators=[DataRequired()]
    )
    submit = SubmitField('Log In')


class LogoutForm(FlaskForm):
    """Empty form used solely for CSRF protection on POST logout."""
    submit = SubmitField('Log Out')


class OnboardingForm(FlaskForm):
    wellness_goals = SelectMultipleField(
        'What are your primary wellness goals?',
        choices=[(goal, goal.replace('_', ' ').title()) for goal in WELLNESS_GOALS],
        description='Optional. Hold Ctrl/Cmd to select multiple.',
    )
    preferred_checkin_time = StringField(
        'Preferred Check-in Time (Optional)',
        description='Choose a time that works best for you (24-hour format).',
        validators=[
            Optional(),
            Length(max=5),
            Regexp(r'^(?:[01]\d|2[0-3]):[0-5]\d$', message='Must be a valid 24-hour time (HH:MM).')
        ]
    )
    timezone = StringField(
        'Timezone (Optional)',
        validators=[Optional(), Length(max=50)],
        description='Auto-detected browser timezone (e.g. Asia/Kolkata).'
    )
    submit = SubmitField('Complete Onboarding')

