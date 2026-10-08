"""Mood Check-In Forms."""

from flask_wtf import FlaskForm
from wtforms import RadioField, TextAreaField, SubmitField
from wtforms.validators import InputRequired, Length, Optional

from app.models.mood import VALID_MOODS

# Generate choices dynamically from canonical VALID_MOODS list
# E.g., [('Very low', 'Very low'), ('Low', 'Low'), ...]
CHOICES = [(m, m) for m in VALID_MOODS]

class MoodCheckinForm(FlaskForm):
    """Form for logging current mood."""
    
    mood = RadioField(
        'How are you feeling right now?',
        choices=CHOICES,
        validators=[InputRequired(message="Please select a mood.")]
    )
    
    journal_note = TextAreaField(
        'Private Journal Note (Optional)',
        validators=[Optional(), Length(max=2000, message="Note is too long.")],
        description='Optional private thoughts about how you are feeling.'
    )
    
    submit = SubmitField('Save Check-In')
