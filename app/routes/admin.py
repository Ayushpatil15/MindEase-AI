"""Admin routes — aggregated analytics dashboard and diagnostics.

Full implementation in Phase 7.
"""

import hmac
import os
from flask import Blueprint, abort, jsonify, redirect, render_template_string, request, url_for
from flask_login import current_user, login_required
from app.utils.decorators import admin_required
from app.services.chat_service import send_minimal_gemini_diagnostic

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')


@admin_bp.route('/')
@login_required
@admin_required
def dashboard():
    """Admin analytics dashboard — Phase 7."""
    return render_template_string('{% extends "base.html" %}{% block content %}<h1>MindEase AI — Admin</h1><p>Admin dashboard — coming in Phase 7.</p>{% endblock %}')


@admin_bp.route('/diagnostics/gemini', methods=['GET', 'POST'])
def gemini_diagnostic():
    """Temporary admin-only diagnostic route to verify live Gemini API connectivity.

    Security & Privacy Controls:
    - Restricted to authenticated administrators (Flask-Login session with role='admin').
    - Alternatively accepts a secure diagnostic token via header ('X-Diagnostic-Token')
      or query parameter ('?token=...') matching 'RENDER_DIAGNOSTIC_TOKEN' (enabling
      secure diagnostics on PaaS platforms without shell access to create an admin account).
    - If unauthenticated, redirects to login; if unauthorized (student), returns 403 Forbidden.
    - Sends exactly one minimal generateContent request directly to gemini-3.5-flash-lite.
    - Returns only the HTTP status and sanitized error category/status.
    - Never exposes, logs, or returns the API key, tokens, or sensitive headers.
    - Kept completely isolated from the normal chatbot conversational flow.
    """
    diagnostic_token = os.environ.get('RENDER_DIAGNOSTIC_TOKEN')
    header_token = request.headers.get('X-Diagnostic-Token')
    is_token_auth = bool(
        diagnostic_token
        and header_token
        and hmac.compare_digest(diagnostic_token.strip(), header_token.strip())
    )

    if not is_token_auth:
        if not current_user.is_authenticated:
            return redirect(url_for('auth.login'))
        if not current_user.is_admin:
            abort(403)

    result = send_minimal_gemini_diagnostic()
    return jsonify(result), 200
