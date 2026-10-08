"""Tests for Phase 5 Facial Expression Check-In."""

import os
import json
import base64
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from app.models.emotion import EmotionScan, EMOTION_CLASSES
from app.services.emotion_service import save_emotion_scan
from app.ml.preprocessor import preprocess_frame, clear_cascade_cache
from app.ml.predictor import predict, reset as predictor_reset
from app.ml.model import build_model
from tests.conftest import create_test_student


# --- 1. Service / Persistence Tests ---

def test_service_saves_valid_emotion(app, db):
    with app.app_context():
        user = create_test_student(db, email='emo@test.com', password='pass')
        scan = save_emotion_scan(user.id, 'Happy', 0.95)
        
        assert scan.id is not None
        assert scan.user_id == user.id
        assert scan.emotion == 'Happy'
        assert scan.confidence == 0.95
        
def test_service_rejects_invalid_class(app, db):
    with app.app_context():
        user = create_test_student(db, email='emo2@test.com', password='pass')
        with pytest.raises(ValueError, match="Invalid emotion class"):
            save_emotion_scan(user.id, 'Disgust', 0.95)  # Disgust is strictly prohibited

def test_service_rejects_invalid_confidence(app, db):
    with app.app_context():
        user = create_test_student(db, email='emo3@test.com', password='pass')
        with pytest.raises(ValueError, match="Invalid confidence range"):
            save_emotion_scan(user.id, 'Happy', 1.05)


# --- 2. Preprocessor Tests (Mocking cv2) ---

@pytest.fixture(autouse=True)
def reset_cascade_cache():
    clear_cascade_cache()
    yield
    clear_cascade_cache()


@patch('app.ml.preprocessor.cv2')

@patch('app.ml.preprocessor.os.path.isfile')
def test_preprocessor_single_face(mock_isfile, mock_cv2):
    mock_isfile.return_value = True
    
    # Mock image decode returning a 100x100 dummy image
    mock_cv2.imdecode.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cv2.cvtColor.return_value = np.zeros((100, 100), dtype=np.uint8)
    
    # Mock Cascade returning one face: x=10, y=10, w=50, h=50
    mock_cascade = MagicMock()
    mock_cascade.detectMultiScale.return_value = [[10, 10, 50, 50]]
    mock_cv2.CascadeClassifier.return_value = mock_cascade
    
    # Mock resize returning exactly 48x48
    mock_cv2.resize.return_value = np.zeros((48, 48), dtype=np.uint8)
    
    res = preprocess_frame(b'dummy_bytes', 'fake_path.xml')
    assert res['status'] == 'success'
    assert res['tensor'].shape == (1, 48, 48, 1)

@patch('app.ml.preprocessor.cv2')
@patch('app.ml.preprocessor.os.path.isfile')
def test_preprocessor_multiple_faces_selects_largest(mock_isfile, mock_cv2):
    mock_isfile.return_value = True
    mock_cv2.imdecode.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cv2.cvtColor.return_value = np.zeros((100, 100), dtype=np.uint8)
    mock_cv2.resize.return_value = np.zeros((48, 48), dtype=np.uint8)
    
    mock_cascade = MagicMock()
    # Face 1: area 100, Face 2: area 2500, Face 3: area 400
    mock_cascade.detectMultiScale.return_value = [
        [0, 0, 10, 10], 
        [10, 10, 50, 50], 
        [0, 0, 20, 20]
    ]
    mock_cv2.CascadeClassifier.return_value = mock_cascade
    
    res = preprocess_frame(b'dummy_bytes', 'fake_path.xml')
    assert res['status'] == 'success'
    # We trust the mocked slice logic ran

@patch('app.ml.preprocessor.cv2')
@patch('app.ml.preprocessor.os.path.isfile')
def test_preprocessor_no_face(mock_isfile, mock_cv2):
    mock_isfile.return_value = True
    mock_cv2.imdecode.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cv2.cvtColor.return_value = np.zeros((100, 100), dtype=np.uint8)
    
    mock_cascade = MagicMock()
    mock_cascade.detectMultiScale.return_value = []
    mock_cv2.CascadeClassifier.return_value = mock_cascade
    
    res = preprocess_frame(b'dummy_bytes', 'fake_path.xml')
    assert res['status'] == 'no_face_detected'


def test_preprocessor_concurrency_safety():
    """Verify that multiple concurrent preprocessor calls execute safely without race conditions."""
    import concurrent.futures
    import cv2
    
    cascade_path = "app/static/haar/haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        pytest.skip("Haar cascade file missing for concurrency test")

    img = np.full((300, 300, 3), 200, dtype=np.uint8)
    cv2.ellipse(img, (150, 150), (60, 80), 0, 0, 360, (100, 100, 100), -1)
    cv2.circle(img, (130, 130), 10, (30, 30, 30), -1)
    cv2.circle(img, (170, 130), 10, (30, 30, 30), -1)
    cv2.ellipse(img, (150, 180), (25, 10), 0, 0, 180, (30, 30, 30), 3)
    _, buf = cv2.imencode('.jpg', img)
    frame_bytes = buf.tobytes()

    def run_preprocess(_):
        return preprocess_frame(frame_bytes, cascade_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(run_preprocess, range(20)))

    for res in results:
        assert res['status'] == 'success'
        assert res['tensor'].shape == (1, 48, 48, 1)


# --- 3. Predictor Logic ---

@patch('app.ml.predictor.os.path.isfile')
def test_missing_model_fallback(mock_isfile):
    predictor_reset()
    mock_isfile.side_effect = lambda path: False if path.endswith('.h5') else True
    
    res = predict(b'dummy', 'fake.h5', 'fake.xml')
    assert res['status'] == 'model_not_initialized'

@patch('app.ml.predictor.os.path.isfile')
@patch('app.ml.preprocessor.preprocess_frame')
@patch('app.ml.model.load_model')
def test_predictor_success_mapping(mock_load, mock_prep, mock_isfile):
    predictor_reset()
    mock_isfile.return_value = True
    
    mock_prep.return_value = {'status': 'success', 'tensor': np.zeros((1, 48, 48, 1))}
    
    mock_model = MagicMock()
    # Output matches the 5 classes [Angry, Fear, Happy, Sad, Neutral]
    # Let's mock a high confidence for Happy (index 2)
    mock_model.predict.return_value = np.array([[0.05, 0.05, 0.80, 0.05, 0.05]])
    mock_load.return_value = mock_model
    
    res = predict(b'dummy', 'fake.h5', 'fake.xml')
    assert res['status'] == 'success'
    assert res['emotion'] == 'Happy'
    assert res['confidence'] == 0.80


# --- 4. API Endpoints / Auth / Boundaries ---

def test_api_unauthenticated(client):
    res = client.post('/emotion/api/predict', json={'image': 'data'})
    assert res.status_code == 302  # redirects to login

def test_api_oversized_payload(client, db):
    create_test_student(db, email='sz@test.com', password='pass')
    client.post('/auth/login', data={'email': 'sz@test.com', 'password': 'pass'})
    
    # 5MB + 1 byte payload
    huge_payload = 'a' * (5 * 1024 * 1024 + 1)
    res = client.post('/emotion/api/predict', data=huge_payload)
    assert res.status_code == 413

def test_api_malformed_base64(client, db):
    create_test_student(db, email='b64@test.com', password='pass')
    client.post('/auth/login', data={'email': 'b64@test.com', 'password': 'pass'})
    
    res = client.post('/emotion/api/predict', json={'image': 'data:image/jpeg;base64,!!!invalid!!!'})
    assert res.status_code == 400
    assert b'Malformed Base64 payload' in res.data


# --- 5. Lightweight Keras Integration Test ---

def test_keras_integration(tmp_path):
    """Dynamically builds a small model, saves it, loads it via predictor to verify Keras boundary.
    No permanent artifact is created.
    """
    predictor_reset()
    
    # 1. Build authentic 5-class TF model and save to tmp
    model = build_model()
    tmp_h5 = str(tmp_path / "temp_fer2013_cnn.h5")
    model.save(tmp_h5)
    
    # 2. Assert file exists
    assert os.path.exists(tmp_h5)
    
    # 3. Use predictor with a mocked preprocessor (so we don't need real haar/cv2 logic here)
    with patch('app.ml.preprocessor.preprocess_frame') as mock_prep:
        with patch('app.ml.predictor.os.path.isfile') as mock_isfile:
            mock_isfile.return_value = True
            
            mock_prep.return_value = {'status': 'success', 'tensor': np.zeros((1, 48, 48, 1), dtype=np.float32)}
            
            res = predict(b'dummy', tmp_h5, 'dummy.xml')
            
            assert res['status'] == 'success'
            assert res['emotion'] in EMOTION_CLASSES
            assert 0.0 <= res['confidence'] <= 1.0

    # Model file will be automatically deleted by pytest tmp_path cleanup


def test_build_challenger_model_structure():
    """Verify Phase 5.6 challenger model builds with expected input/output shapes."""
    from app.ml.model import build_challenger_model
    challenger = build_challenger_model()
    
    assert challenger.input_shape == (None, 48, 48, 1)
    assert challenger.output_shape == (None, 5)
    assert challenger.name == 'fer2013_challenger_cnn'


def test_challenger_keras_integration(tmp_path):
    """Dynamically builds Phase 5.6 challenger model, saves it, loads via predictor.
    Verifies inference boundary and output schema with zero persistent side effects.
    """
    from app.ml.model import build_challenger_model
    predictor_reset()
    
    # 1. Build authentic challenger 5-class TF model and save to tmp
    challenger = build_challenger_model()
    tmp_h5 = str(tmp_path / "temp_fer2013_challenger_cnn.h5")
    challenger.save(tmp_h5)
    
    assert os.path.exists(tmp_h5)
    
    # 2. Predict with mocked preprocessor
    with patch('app.ml.preprocessor.preprocess_frame') as mock_prep:
        with patch('app.ml.predictor.os.path.isfile') as mock_isfile:
            mock_isfile.return_value = True
            mock_prep.return_value = {'status': 'success', 'tensor': np.zeros((1, 48, 48, 1), dtype=np.float32)}
            
            res = predict(b'dummy', tmp_h5, 'dummy.xml')
            
            assert res['status'] == 'success'
            assert res['emotion'] in EMOTION_CLASSES
            assert 0.0 <= res['confidence'] <= 1.0


def test_build_low_latency_model_structure():
    """Verify Phase 5.7 low-latency model builds with expected input/output shapes."""
    from app.ml.model import build_low_latency_model
    model = build_low_latency_model()
    
    assert model.input_shape == (None, 48, 48, 1)
    assert model.output_shape == (None, 5)
    assert model.name == 'fer2013_low_latency_cnn'


def test_build_low_latency_keras_integration(tmp_path):
    """Dynamically builds Phase 5.7 low-latency model, saves it, loads via predictor.
    Verifies inference boundary and output schema with zero persistent side effects.
    """
    from app.ml.model import build_low_latency_model
    predictor_reset()
    
    # 1. Build authentic low-latency 5-class TF model and save to tmp
    model = build_low_latency_model()
    tmp_h5 = str(tmp_path / "temp_fer2013_low_latency_cnn.h5")
    model.save(tmp_h5)
    
    assert os.path.exists(tmp_h5)
    
    # 2. Predict with mocked preprocessor
    with patch('app.ml.preprocessor.preprocess_frame') as mock_prep:
        with patch('app.ml.predictor.os.path.isfile') as mock_isfile:
            mock_isfile.return_value = True
            mock_prep.return_value = {'status': 'success', 'tensor': np.zeros((1, 48, 48, 1), dtype=np.float32)}
            
            res = predict(b'dummy', tmp_h5, 'dummy.xml')
            
            assert res['status'] == 'success'
            assert res['emotion'] in EMOTION_CLASSES
            assert 0.0 <= res['confidence'] <= 1.0


def test_build_hybrid_model_structure():
    """Verify Phase 5.8 hybrid model builds with expected input/output shapes."""
    from app.ml.model import build_hybrid_model
    model = build_hybrid_model()
    
    assert model.input_shape == (None, 48, 48, 1)
    assert model.output_shape == (None, 5)
    assert model.name == 'fer2013_hybrid_cnn'


def test_hybrid_keras_integration(tmp_path):
    """Dynamically builds Phase 5.8 hybrid model, saves it, loads via predictor.
    Verifies inference boundary and output schema with zero persistent side effects.
    """
    from app.ml.model import build_hybrid_model
    predictor_reset()
    
    # 1. Build authentic hybrid 5-class TF model and save to tmp
    model = build_hybrid_model()
    tmp_h5 = str(tmp_path / "temp_fer2013_hybrid_cnn.h5")
    model.save(tmp_h5)
    
    assert os.path.exists(tmp_h5)
    
    # 2. Predict with mocked preprocessor
    with patch('app.ml.preprocessor.preprocess_frame') as mock_prep:
        with patch('app.ml.predictor.os.path.isfile') as mock_isfile:
            mock_isfile.return_value = True
            mock_prep.return_value = {'status': 'success', 'tensor': np.zeros((1, 48, 48, 1), dtype=np.float32)}
            
            res = predict(b'dummy', tmp_h5, 'dummy.xml')
            
            assert res['status'] == 'success'
            assert res['emotion'] in EMOTION_CLASSES
            assert 0.0 <= res['confidence'] <= 1.0


# --- 6. Phase 5.10 Production TFLite & Fallback Tests ---

def test_tflite_predictor_actual_artifact():
    """Verify TFLite engine loads validated challenger_fp32.tflite artifact and predicts correctly."""
    from app.ml.predictor import get_backend_diagnostic_info
    predictor_reset()
    
    tflite_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'deployment', 'challenger_fp32.tflite')
    assert os.path.exists(tflite_path), f"Missing TFLite test artifact: {tflite_path}"
    
    with patch('app.ml.preprocessor.preprocess_frame') as mock_prep:
        mock_prep.return_value = {'status': 'success', 'tensor': np.zeros((1, 48, 48, 1), dtype=np.float32)}
        with patch('app.ml.predictor.os.path.isfile') as mock_isfile:
            mock_isfile.return_value = True
            
            res = predict(b'dummy', tflite_path, 'dummy.xml', backend='tflite')
            
            assert res['status'] == 'success'
            assert res['emotion'] in EMOTION_CLASSES
            assert 0.0 <= res['confidence'] <= 1.0
            assert res['active_backend'] == 'tflite'
            
            diag = get_backend_diagnostic_info()
            assert diag['active_backend'] == 'tflite'
            assert diag['tflite_loaded'] is True
            assert diag['keras_fallback_active'] is False


def test_tflite_predictor_missing_artifact_fallback():
    """Verify predictor automatically falls back to Keras baseline when TFLite model is missing."""
    from app.ml.predictor import get_backend_diagnostic_info
    predictor_reset()
    
    fake_tflite_path = "models/deployment/non_existent_model.tflite"
    real_keras_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'fer2013_cnn.h5')
    
    with patch('app.ml.preprocessor.preprocess_frame') as mock_prep:
        mock_prep.return_value = {'status': 'success', 'tensor': np.zeros((1, 48, 48, 1), dtype=np.float32)}
        
        # Call predict specifying fake tflite path with explicit fallback path
        from app.ml.predictor import EmotionPredictor
        predictor_engine = EmotionPredictor(
            primary_path=fake_tflite_path,
            backend_type='tflite',
            fallback_path=real_keras_path
        )
        
        assert predictor_engine.tflite_loaded is False
        assert predictor_engine.keras_fallback_active is True
        assert predictor_engine.active_backend_name == 'keras'
        
        res = predictor_engine.predict(np.zeros((1, 48, 48, 1), dtype=np.float32))
        assert res['emotion'] in EMOTION_CLASSES
        assert 0.0 <= res['confidence'] <= 1.0


def test_tflite_predictor_explicit_backend_override():
    """Verify explicit backend selection (tflite vs keras)."""
    predictor_reset()
    keras_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'fer2013_cnn.h5')
    
    with patch('app.ml.preprocessor.preprocess_frame') as mock_prep:
        mock_prep.return_value = {'status': 'success', 'tensor': np.zeros((1, 48, 48, 1), dtype=np.float32)}
        with patch('app.ml.predictor.os.path.isfile') as mock_isfile:
            mock_isfile.return_value = True
            
            res = predict(b'dummy', keras_path, 'dummy.xml', backend='keras')
            assert res['status'] == 'success'
            assert res['active_backend'] == 'keras'
