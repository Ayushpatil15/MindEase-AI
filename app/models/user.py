"""User model — students and administrators."""

from datetime import datetime, timezone

from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

from app.extensions import db


class User(UserMixin, db.Model):
    """Application user with role-based access (student or admin)."""

    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='student')
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # Relationships
    stress_assessments = db.relationship(
        'StressAssessment', backref='user', lazy='dynamic',
        cascade='all, delete-orphan',
    )
    mood_entries = db.relationship(
        'MoodEntry', backref='user', lazy='dynamic',
        cascade='all, delete-orphan',
    )
    emotion_scans = db.relationship(
        'EmotionScan', backref='user', lazy='dynamic',
        cascade='all, delete-orphan',
    )
    recommendations = db.relationship(
        'Recommendation', backref='user', lazy='dynamic',
        cascade='all, delete-orphan',
    )
    wellness_tasks = db.relationship(
        'WellnessTask', backref='user', lazy='dynamic',
        cascade='all, delete-orphan',
    )

    def set_password(self, password):
        """Hash and store the password. Never stores plaintext."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Verify a password against the stored hash."""
        return check_password_hash(self.password_hash, password)

    @property
    def is_admin(self):
        return self.role == 'admin'

    @property
    def is_student(self):
        return self.role == 'student'

    def __repr__(self):
        return f'<User {self.username} ({self.role})>'
