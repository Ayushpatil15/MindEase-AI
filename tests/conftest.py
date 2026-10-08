"""Pytest configuration and shared fixtures.

Provides a test Flask app with in-memory SQLite, a test client,
and helper functions for creating test users.
"""

import pytest

from app import create_app
from app.extensions import db as _db
from app.models import User


@pytest.fixture(scope='session')
def app():
    """Create a Flask application configured for testing.

    Uses in-memory SQLite and disabled CSRF.
    """
    app = create_app('testing')
    yield app


@pytest.fixture(scope='function')
def db(app):
    """Provide a clean database for each test function.

    Creates all tables before the test, drops them after.
    """
    with app.app_context():
        _db.create_all()
        yield _db
        _db.session.rollback()
        _db.drop_all()


@pytest.fixture(scope='function')
def client(app, db):
    """Provide a Flask test client with a clean database."""
    with app.test_client() as client:
        with app.app_context():
            yield client


@pytest.fixture(scope='function')
def runner(app, db):
    """Provide a Flask CLI test runner."""
    return app.test_cli_runner()


def create_test_student(db, username='teststudent', email='student@test.com', password='testpass123'):
    """Helper: create and return a student user."""
    user = User(username=username, email=email, role='student')
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    return user


def create_test_admin(db, username='testadmin', email='admin@test.com', password='adminpass123'):
    """Helper: create and return an admin user."""
    user = User(username=username, email=email, role='admin')
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    return user
