"""Authentication routes — registration, login, logout, and onboarding.

Implements secure session handling, CSRF protection, and duplicate prevention.
"""

from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user

from app.extensions import db
from app.models.user import User
from app.models.user_preferences import UserPreferences
from app.forms.auth import RegistrationForm, LoginForm, LogoutForm, OnboardingForm
from app.utils.validators import validate_username, validate_email, validate_password, validate_timezone

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page and handler."""
    if current_user.is_authenticated:
        return redirect(url_for('student.dashboard'))

    form = RegistrationForm()
    if form.validate_on_submit():
        # Validate using our utilities
        username, uname_err = validate_username(form.username.data)
        email, email_err = validate_email(form.email.data)
        password, pass_err = validate_password(form.password.data)

        if uname_err:
            form.username.errors.append(uname_err)
        if email_err:
            form.email.errors.append(email_err)
        if pass_err:
            form.password.errors.append(pass_err)
            
        if uname_err or email_err or pass_err:
            return render_template('auth/register.html', form=form)

        # Check duplicates
        if User.query.filter_by(username=username).first():
            form.username.errors.append('Username is already taken.')
            return render_template('auth/register.html', form=form)
            
        if User.query.filter_by(email=email).first():
            form.email.errors.append('Email is already registered.')
            return render_template('auth/register.html', form=form)

        # Create user (role='student' by default)
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        login_user(user)
        flash('Registration successful. Welcome to MindEase AI!', 'success')
        return redirect(url_for('auth.onboarding'))

    return render_template('auth/register.html', form=form)


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Login page and handler."""
    if current_user.is_authenticated:
        if current_user.role == 'admin':
            return redirect(url_for('admin.dashboard'))
        return redirect(url_for('student.dashboard'))

    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data.strip().lower()).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid email or password.', 'error')
            return render_template('auth/login.html', form=form)

        login_user(user)
        flash('Logged in successfully.', 'success')
        
        next_page = request.args.get('next')
        if next_page and next_page.startswith('/'):
            return redirect(next_page)
            
        if user.role == 'admin':
            return redirect(url_for('admin.dashboard'))
        return redirect(url_for('student.dashboard'))

    return render_template('auth/login.html', form=form)


@auth_bp.route('/logout', methods=['POST'])
@login_required
def logout():
    """Logout handler (POST only for CSRF protection)."""
    form = LogoutForm()
    if form.validate_on_submit():
        logout_user()
        flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))


@auth_bp.route('/onboarding', methods=['GET', 'POST'])
@login_required
def onboarding():
    """Optional onboarding flow to collect wellness goals and check-in time."""
    # Admins should not fill out student onboarding
    if current_user.role == 'admin':
        return redirect(url_for('admin.dashboard'))

    form = OnboardingForm()
    
    if request.method == 'GET' and current_user.preferences:
        form.wellness_goals.data = current_user.preferences.wellness_goals
        form.preferred_checkin_time.data = current_user.preferences.preferred_checkin_time
        form.timezone.data = current_user.preferences.timezone or 'UTC'

    if form.validate_on_submit():
        goals = form.wellness_goals.data or []
        checkin_time = form.preferred_checkin_time.data or None
        user_tz = validate_timezone(form.timezone.data)

        if current_user.preferences:
            current_user.preferences.wellness_goals = goals
            current_user.preferences.preferred_checkin_time = checkin_time
            current_user.preferences.timezone = user_tz
            flash('Preferences updated successfully.', 'success')
        else:
            prefs = UserPreferences(
                user_id=current_user.id,
                wellness_goals=goals,
                preferred_checkin_time=checkin_time,
                timezone=user_tz
            )
            db.session.add(prefs)
            flash('Preferences saved. Welcome aboard!', 'success')
            
        db.session.commit()
        return redirect(url_for('student.dashboard'))

    return render_template('auth/onboarding.html', form=form)
