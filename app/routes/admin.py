"""Admin routes — aggregated analytics dashboard.

Full implementation in Phase 7.
"""

from flask import Blueprint, render_template_string
from flask_login import login_required
from app.utils.decorators import admin_required

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')


@admin_bp.route('/')
@login_required
@admin_required
def dashboard():
    """Admin analytics dashboard — Phase 7."""
    return render_template_string('{% extends "base.html" %}{% block content %}<h1>MindEase AI — Admin</h1><p>Admin dashboard — coming in Phase 7.</p>{% endblock %}')
