"""MindEase AI — Chat Routes.

Authenticated student-facing chat routes connecting to the Gemini chatbot service.
Enforces CSRF verification, rate limiting, role checks, and input boundaries.
"""

import logging
from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user

from app.extensions import limiter
from app.utils.decorators import student_required
from app.services.chat_service import (
    send_chat_message,
    MAX_MESSAGE_LENGTH,
)

logger = logging.getLogger(__name__)

chat_bp = Blueprint('chat', __name__, url_prefix='/chat')

# Payload size bound: 64 KB (more than sufficient for text messages + short history)
MAX_CHAT_PAYLOAD_BYTES = 64 * 1024


@chat_bp.route('/', methods=['GET'])
@login_required
@student_required
def index():
    """Render the student wellness chatbot interface."""
    return render_template('chat/index.html')


@chat_bp.route('/api/message', methods=['POST'])
@login_required
@student_required
@limiter.limit("20/minute", methods=["POST"])
def api_message():
    """Receive student chat message, validate inputs, screen for crisis, and call Gemini service."""
    # 1. Enforce payload size boundary
    content_length = request.content_length
    if content_length is not None and content_length > MAX_CHAT_PAYLOAD_BYTES:
        return jsonify({
            'status': 'error',
            'message': 'Payload exceeds maximum permitted size of 64KB.'
        }), 413

    # 2. Parse JSON safely
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({
            'status': 'error',
            'message': 'Invalid JSON request format.'
        }), 400

    raw_message = data.get('message')
    if not raw_message or not isinstance(raw_message, str):
        return jsonify({
            'status': 'error',
            'message': 'Message cannot be empty.'
        }), 400

    clean_message = raw_message.strip()
    if not clean_message:
        return jsonify({
            'status': 'error',
            'message': 'Message cannot be empty.'
        }), 400

    if len(clean_message) > MAX_MESSAGE_LENGTH:
        return jsonify({
            'status': 'error',
            'message': f'Message exceeds maximum permitted length of {MAX_MESSAGE_LENGTH} characters.'
        }), 400

    # 3. Sanitize conversation history from client
    raw_history = data.get('history')
    if not isinstance(raw_history, list):
        raw_history = []

    # 4. Invoke Chat Service (includes in-memory crisis screening before LLM)
    result = send_chat_message(
        user_message=clean_message,
        conversation_history=raw_history,
    )

    status = result.get('status')

    # 5. Formulate consistent response and HTTP status codes
    if status == 'success':
        return jsonify({
            'status': 'success',
            'reply': result['reply'],
        }), 200

    if status == 'crisis_detected':
        return jsonify({
            'status': 'crisis_detected',
            'reply': result['reply'],
            'support_resources': result.get('support_resources'),
        }), 200

    if status == 'quota_exceeded':
        return jsonify({
            'status': 'quota_exceeded',
            'message': result.get('reply', 'Rate limit exceeded. Please try again shortly.'),
        }), 429

    if status == 'service_unavailable':
        return jsonify({
            'status': 'service_unavailable',
            'message': result.get('reply', 'AI companion is currently unavailable.'),
        }), 503

    if status == 'timeout':
        return jsonify({
            'status': 'timeout',
            'message': result.get('reply', 'Request timed out.'),
        }), 504

    # Default error response
    return jsonify({
        'status': 'error',
        'message': result.get('reply', 'Failed to generate response.'),
    }), 500
