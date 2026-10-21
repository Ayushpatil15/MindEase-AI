"""Facial Expression Check-In Routes.

Orchestrates frontend requests to the ML predictor and service persistence.
Enforces strict 5MB payload limits, validation, and Base64 constraints.
"""

import os
import base64
import logging

from flask import Blueprint, render_template, request, jsonify, current_app
from flask_login import login_required, current_user
from werkzeug.exceptions import RequestEntityTooLarge

from app.extensions import limiter
from app.utils.decorators import student_required
from app.services.emotion_service import save_emotion_scan
from app.ml.predictor import predict

logger = logging.getLogger(__name__)

emotion_bp = Blueprint('emotion', __name__, url_prefix='/emotion')

# Strict 5 MB size limit applied specifically to this endpoint
MAX_PAYLOAD_SIZE = 5 * 1024 * 1024


@emotion_bp.route('/', methods=['GET'])
@login_required
@student_required
def checkin():
    """Render the camera UI for facial expression inference."""
    return render_template('emotion/checkin.html')


@emotion_bp.route('/api/predict', methods=['POST'])
@login_required
@student_required
@limiter.limit("15/minute", methods=["POST"])
def api_predict():
    """Receive Base64 frame, process inference in memory, persist result."""
    # 1. Enforce payload size limit securely
    max_payload = current_app.config.get('MAX_CONTENT_LENGTH', MAX_PAYLOAD_SIZE)
    content_length = request.content_length
    if content_length is not None and content_length > max_payload:
        return jsonify({'status': 'error', 'message': 'Payload exceeds 5MB limit.'}), 413
        
    # Read raw data directly to avoid framework parsing overhead on large files, 
    # and to catch length errors if content-length header was missing/forged
    raw_data = request.get_data(cache=False, as_text=False)
    if len(raw_data) > max_payload:
        return jsonify({'status': 'error', 'message': 'Payload exceeds 5MB limit.'}), 413

    # Now parse JSON safely
    try:
        import json
        data = json.loads(raw_data)
    except json.JSONDecodeError:
        return jsonify({'status': 'error', 'message': 'Invalid JSON format.'}), 400

    if not data or 'image' not in data:
        return jsonify({'status': 'error', 'message': 'Missing image payload.'}), 400
        
    img_b64 = data['image']
    
    # 2. Strict Base64 parsing (strip dataURI prefix if present)
    if isinstance(img_b64, str) and img_b64.startswith('data:image'):
        comma_idx = img_b64.find(',')
        if comma_idx != -1:
            img_b64 = img_b64[comma_idx + 1:]
        else:
            return jsonify({'status': 'error', 'message': 'Malformed Data URI.'}), 400
            
    try:
        frame_bytes = base64.b64decode(img_b64, validate=True)
    except Exception:
        return jsonify({'status': 'error', 'message': 'Malformed Base64 payload.'}), 400
        
    # 3. Model & Cascade path resolution from configuration
    model_path = current_app.config.get(
        'TFLITE_MODEL_PATH',
        os.path.join(current_app.root_path, '..', 'models', 'deployment', 'challenger_fp32.tflite')
    )
    cascade_path = current_app.config.get(
        'HAAR_CASCADE_PATH',
        os.path.join(current_app.root_path, 'static', 'haar', 'haarcascade_frontalface_default.xml')
    )
    
    # 4. Predict via TFLite engine
    result = predict(frame_bytes, model_path, cascade_path)
    
    # 5. Handle Results and Persist
    if result['status'] == 'success':
        min_conf = current_app.config.get('EMOTION_MIN_CONFIDENCE', 0.5)
        if result['confidence'] < min_conf:
            return jsonify({
                'status': 'low_confidence',
                'emotion': 'Not sure',
                'confidence': result['confidence']
            })

        try:
            save_emotion_scan(
                user_id=current_user.id,
                emotion=result['emotion'],
                confidence=result['confidence']
            )
            return jsonify({
                'status': 'success',
                'emotion': result['emotion'],
                'confidence': result['confidence']
            })
        except ValueError as e:
            # Reached if ML somehow returned invalid classes
            logger.error("Service rejected prediction: %s", str(e))
            return jsonify({'status': 'error', 'message': 'Internal service constraint violation.'}), 500
            
    elif result['status'] == 'model_not_initialized':
        return jsonify({'status': 'model_not_initialized'}), 503
        
    elif result['status'] in ('no_face_detected', 'cascade_missing'):
        return jsonify({'status': result['status'], 'message': result['message']}), 400
        
    else:
        # Catch-all for error status
        return jsonify({'status': 'error', 'message': result.get('message', 'Inference failed.')}), 500
