"""MindEase AI — Flask Application Factory.

Creates and configures the Flask application with:
- SQLAlchemy database (SQLite MVP, PostgreSQL-ready)
- Flask-Migrate for Alembic-based schema management
- Flask-Login for session authentication
- Flask-WTF CSRF protection
- Blueprint registration (auth, student, admin, api)
- CLI commands (create-admin)
"""

import os
import logging

import click
from flask import Flask, render_template

from app.config import config
from app.extensions import db, login_manager, migrate, csrf, limiter

logger = logging.getLogger(__name__)


def create_app(config_name=None):
    """Application factory.

    Args:
        config_name: One of 'development', 'testing', 'production', or 'default'.
                     Falls back to APP_ENV / FLASK_ENV env var, then 'default'.

    Returns:
        Configured Flask application instance.
    """
    if config_name is None:
        config_name = os.environ.get('APP_ENV') or os.environ.get('FLASK_ENV') or 'default'

    if config_name not in config:
        raise ValueError(
            f"Invalid configuration '{config_name}'. Must be one of: {list(config.keys())}"
        )

    app = Flask(__name__)
    app.config.from_object(config[config_name])

    # ── Initialize Extensions ─────────────────────────────────────
    db.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)
    csrf.init_app(app)
    
    ratelimit_enabled = app.config.get('RATELIMIT_ENABLED', True)
    app.config['RATELIMIT_ENABLED'] = True
    limiter.init_app(app)
    app.config['RATELIMIT_ENABLED'] = ratelimit_enabled
    limiter.enabled = ratelimit_enabled

    # ── User Loader (Flask-Login) ─────────────────────────────────
    from app.models import User

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    # ── Register Blueprints ───────────────────────────────────────
    from app.routes.auth import auth_bp
    from app.routes.student import student_bp
    from app.routes.admin import admin_bp
    from app.routes.api import api_bp
    from app.routes.stress import stress_bp
    from app.routes.mood import mood_bp
    from app.routes.emotion import emotion_bp
    from app.routes.chat import chat_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(student_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(stress_bp)
    app.register_blueprint(mood_bp)
    app.register_blueprint(emotion_bp)
    app.register_blueprint(chat_bp)

    # ── Reverse Proxy Handling (ProxyFix for Render/Production) ───
    if app.config.get('BEHIND_PROXY'):
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(
            app.wsgi_app,
            x_for=1,
            x_proto=1,
            x_host=1,
            x_port=1,
        )

    # ── Root Health & Readiness Route ─────────────────────────────
    @app.route('/health')
    def health_root():
        """Root-level readiness check for load balancers and uptime monitoring."""
        from app.routes.api import perform_health_check
        return perform_health_check()

    # ── Error Handlers ────────────────────────────────────────────
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template('500.html'), 500

    # ── CLI Commands ──────────────────────────────────────────────
    _register_cli(app)

    # ── Ensure Instance Directory ─────────────────────────────────
    os.makedirs(app.instance_path, exist_ok=True)

    # ── Ensure Models Directory ───────────────────────────────────
    models_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'models',
    )
    os.makedirs(models_dir, exist_ok=True)

    logger.info('MindEase AI initialized with config: %s', config_name)

    return app


def _register_cli(app):
    """Register custom Flask CLI commands."""

    @app.cli.command('create-admin')
    @click.option('--username', prompt='Admin username', help='Admin username.')
    @click.option('--email', prompt='Admin email', help='Admin email address.')
    @click.option(
        '--password',
        prompt=True,
        hide_input=True,
        confirmation_prompt=True,
        help='Admin password (min 8 characters).',
    )
    def create_admin_cmd(username, email, password):
        """Create an administrator account securely.

        No default credentials are ever created. The password is entered
        interactively (hidden) or supplied via the prompt.
        """
        from app.models import User
        from app.utils.validators import (
            validate_username, validate_email, validate_password,
        )

        # Validate
        username, err = validate_username(username)
        if err:
            click.echo(f'Error: {err}', err=True)
            raise SystemExit(1)

        email, err = validate_email(email)
        if err:
            click.echo(f'Error: {err}', err=True)
            raise SystemExit(1)

        password, err = validate_password(password)
        if err:
            click.echo(f'Error: {err}', err=True)
            raise SystemExit(1)

        # Check uniqueness
        if User.query.filter_by(username=username).first():
            click.echo(f'Error: Username "{username}" already exists.', err=True)
            raise SystemExit(1)

        if User.query.filter_by(email=email).first():
            click.echo(f'Error: Email "{email}" is already registered.', err=True)
            raise SystemExit(1)

        # Create admin user
        admin = User(username=username, email=email, role='admin')
        admin.set_password(password)
        db.session.add(admin)
        db.session.commit()

        click.echo(f'Admin account "{username}" created successfully.')
