"""Authorization decorators.

Centralizes access control so route handlers remain thin.
"""

from functools import wraps

from flask import abort, flash, redirect, url_for
from flask_login import current_user


def admin_required(f):
    """Restrict access to admin users only.

    Must be used AFTER @login_required so current_user is populated.
    Returns 403 if the logged-in user is not an admin.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('auth.login'))
        if not current_user.is_admin:
            abort(403)
        return f(*args, **kwargs)
    return decorated_function


def student_required(f):
    """Restrict access to student users only.

    Prevents admin accounts from accidentally accessing student-only flows.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('auth.login'))
        if not current_user.is_student:
            abort(403)
        return f(*args, **kwargs)
    return decorated_function
