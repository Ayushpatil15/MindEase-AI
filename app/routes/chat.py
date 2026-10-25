"""MindEase AI — Chat Routes.

Authenticated student-facing chat routes connecting to the Gemini chatbot service.
Enforces CSRF verification, rate limiting, role checks, and input boundaries.
"""

import logging
from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user

from app.extensions import db, limiter
from app.utils.decorators import student_required
from app.models.chat_memory import ChatMemory
from app.models.user_preferences import UserPreferences
from app.services.chat_context_service import is_memory_enabled_for_user
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
        user=current_user,
    )

    status = result.get('status')

    # 5. Formulate consistent response and HTTP status codes
    if status == 'success':
        response_payload = {
            'status': 'success',
            'reply': result['reply'],
        }
        if 'duration_seconds' in result:
            response_payload['duration_seconds'] = result['duration_seconds']
        return jsonify(response_payload), 200

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

    if status == 'blocked':
        return jsonify({
            'status': 'blocked',
            'reply': result['reply'],
            'support_resources': result.get('support_resources'),
        }), 200

    if status == 'empty_response':
        return jsonify({
            'status': 'empty_response',
            'reply': result.get('reply', 'I was unable to formulate a response to that. Could you try rephrasing?'),
            'message': result.get('reply', 'I was unable to formulate a response to that. Could you try rephrasing?'),
        }), 200

    # Default upstream failure (502 Bad Gateway rather than internal crash 500)
    return jsonify({
        'status': 'error',
        'message': result.get('reply', 'Failed to generate response.'),
    }), 502


@chat_bp.route('/api/memories', methods=['GET'])
@login_required
@student_required
def get_memories():
    """Retrieve the authenticated student's saved chatbot memories and preference status."""
    memories = (
        ChatMemory.query.filter_by(user_id=current_user.id)
        .order_by(ChatMemory.created_at.desc())
        .all()
    )
    memory_enabled = is_memory_enabled_for_user(current_user)
    return jsonify({
        'status': 'success',
        'memory_enabled': memory_enabled,
        'memories': [
            {
                'id': m.id,
                'text': m.memory_text,
                'category': m.category,
                'created_at': m.created_at.isoformat() if m.created_at else None,
            }
            for m in memories
        ],
    }), 200


@chat_bp.route('/api/memories/<int:memory_id>', methods=['DELETE'])
@login_required
@student_required
def delete_memory(memory_id: int):
    """Delete a specific saved memory owned by the authenticated student."""
    memory = ChatMemory.query.filter_by(id=memory_id, user_id=current_user.id).first()
    if not memory:
        return jsonify({
            'status': 'error',
            'message': 'Memory not found or unauthorized.',
        }), 404

    try:
        db.session.delete(memory)
        db.session.commit()
        return jsonify({
            'status': 'success',
            'message': 'Memory deleted.',
        }), 200
    except Exception as e:
        db.session.rollback()
        logger.error("Error deleting memory %s: %s", memory_id, type(e).__name__)
        return jsonify({
            'status': 'error',
            'message': 'Could not delete memory.',
        }), 500


@chat_bp.route('/api/memories', methods=['DELETE'])
@login_required
@student_required
def clear_all_memories():
    """Clear all saved chatbot memories for the authenticated student."""
    try:
        ChatMemory.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
        return jsonify({
            'status': 'success',
            'message': 'All chatbot memories cleared.',
        }), 200
    except Exception as e:
        db.session.rollback()
        logger.error("Error clearing memories for user %s: %s", current_user.id, type(e).__name__)
        return jsonify({
            'status': 'error',
            'message': 'Could not clear memories.',
        }), 500


@chat_bp.route('/api/memories/toggle', methods=['POST'])
@login_required
@student_required
def toggle_memory():
    """Toggle the persistent chatbot memory setting for the authenticated student."""
    data = request.get_json(silent=True) or {}
    prefs = current_user.preferences
    if not prefs:
        prefs = UserPreferences(user_id=current_user.id)
        db.session.add(prefs)

    current_val = getattr(prefs, 'chat_memory_enabled', True)
    if 'enabled' in data and isinstance(data['enabled'], bool):
        new_val = data['enabled']
    else:
        new_val = not current_val

    prefs.chat_memory_enabled = new_val
    try:
        db.session.commit()
        return jsonify({
            'status': 'success',
            'memory_enabled': prefs.chat_memory_enabled,
            'message': 'Chatbot memory updated.',
        }), 200
    except Exception as e:
        db.session.rollback()
        logger.error("Error toggling memory for user %s: %s", current_user.id, type(e).__name__)
        return jsonify({
            'status': 'error',
            'message': 'Could not update memory setting.',
        }), 500
