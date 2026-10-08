"""Predictor — runs CNN inference on preprocessed frames with multi-backend TFLite & Keras support."""

import os
import logging
import threading
import numpy as np

logger = logging.getLogger(__name__)

# Mapping from neural network output index to exact class name
from app.models.emotion import EMOTION_CLASS_MAPPING

# Global singleton predictor instance and lock
_predictor_instance = None
_predictor_lock = threading.Lock()


class TFLiteBackend:
    """TensorFlow Lite inference engine with thread-safe invocation."""

    def __init__(self, model_path, num_threads=4):
        self.model_path = model_path
        self.num_threads = num_threads
        self.lock = threading.Lock()

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"TFLite model artifact missing at {model_path}")

        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path,
            num_threads=num_threads
        )
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_idx = self.input_details[0]['index']
        self.output_idx = self.output_details[0]['index']

        self.in_dtype = self.input_details[0]['dtype']
        self.out_dtype = self.output_details[0]['dtype']

        self.input_scale, self.input_zero_point = self.input_details[0].get('quantization', (0.0, 0))
        self.output_scale, self.output_zero_point = self.output_details[0].get('quantization', (0.0, 0))

    def predict(self, tensor):
        """Execute single-sample TFLite inference thread-safely."""
        with self.lock:
            # 1. Quantize or format input
            if self.in_dtype == np.int8:
                sample_quant = np.round(tensor / self.input_scale + self.input_zero_point).astype(np.int8)
                self.interpreter.set_tensor(self.input_idx, sample_quant)
            elif self.in_dtype == np.uint8:
                sample_quant = np.round(tensor / self.input_scale + self.input_zero_point).astype(np.uint8)
                self.interpreter.set_tensor(self.input_idx, sample_quant)
            else:
                self.interpreter.set_tensor(self.input_idx, tensor.astype(np.float32))

            # 2. Execute inference
            self.interpreter.invoke()

            # 3. Retrieve and dequantize output
            out_tensor = self.interpreter.get_tensor(self.output_idx)
            if self.out_dtype == np.int8 or self.out_dtype == np.uint8:
                probs = (out_tensor[0].astype(np.float32) - self.output_zero_point) * self.output_scale
            else:
                probs = out_tensor[0]

            return probs


class KerasBackend:
    """Keras .h5 inference engine for baseline production execution and fallback."""

    def __init__(self, model_path):
        self.model_path = model_path
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Keras model artifact missing at {model_path}")

        from app.ml.model import load_model
        self.model = load_model(model_path)
        if self.model is None:
            raise RuntimeError(f"Failed to load Keras model from {model_path}")

    def predict(self, tensor):
        """Execute Keras model prediction."""
        preds = self.model.predict(tensor, verbose=0)
        return preds[0]


class EmotionPredictor:
    """Unified backend manager supporting TFLite, Keras, and automatic fallback."""

    def __init__(self, primary_path, backend_type='keras', fallback_path=None, num_threads=4):
        self.primary_path = primary_path
        self.backend_type = backend_type.lower() if backend_type else 'keras'
        self.fallback_path = fallback_path or os.path.join(
            os.path.dirname(__file__), '..', '..', 'models', 'fer2013_cnn.h5'
        )
        self.num_threads = num_threads

        self.backend = None
        self.active_backend_name = 'uninitialized'
        self.tflite_loaded = False
        self.keras_fallback_active = False
        self.last_error = None

        self._initialize_backend()

    def _initialize_backend(self):
        """Initialize the requested primary backend or activate Keras fallback."""
        if self.backend_type in ('tflite', 'auto') or (self.primary_path and self.primary_path.endswith('.tflite')):
            try:
                logger.info("Initializing TFLiteBackend from %s (threads=%d)", self.primary_path, self.num_threads)
                self.backend = TFLiteBackend(self.primary_path, num_threads=self.num_threads)
                self.active_backend_name = 'tflite'
                self.tflite_loaded = True
                return
            except Exception as e:
                self.last_error = str(e)
                logger.warning(
                    "TFLite initialization failed (%s); activating Keras baseline fallback from %s",
                    str(e), self.fallback_path
                )
                self.tflite_loaded = False

        # Attempt Keras initialization (either as primary choice or fallback)
        keras_target = self.primary_path if (self.backend_type == 'keras' or not self.primary_path.endswith('.tflite')) else self.fallback_path
        try:
            logger.info("Initializing KerasBackend from %s", keras_target)
            self.backend = KerasBackend(keras_target)
            self.active_backend_name = 'keras'
            if self.primary_path.endswith('.tflite') or self.backend_type == 'tflite':
                self.keras_fallback_active = True
        except Exception as e:
            self.last_error = str(e)
            logger.error("Keras backend initialization failed: %s", str(e))
            self.backend = None
            self.active_backend_name = 'failed'

    def predict(self, tensor):
        """Execute inference using active backend with output validation."""
        if self.backend is None:
            raise RuntimeError("No active ML backend available")

        try:
            probs = self.backend.predict(tensor)
        except Exception as e:
            logger.warning("Primary backend inference failed (%s); attempting Keras fallback...", str(e))
            if self.active_backend_name == 'tflite' and self.fallback_path and os.path.isfile(self.fallback_path):
                try:
                    fallback_engine = KerasBackend(self.fallback_path)
                    probs = fallback_engine.predict(tensor)
                    self.keras_fallback_active = True
                    self.active_backend_name = 'keras_fallback'
                except Exception as fallback_err:
                    logger.error("Fallback inference failed: %s", str(fallback_err))
                    raise RuntimeError("Inference failed on both primary and fallback backends") from fallback_err
            else:
                raise

        # Output Numerical Integrity Validation
        probs = np.asarray(probs)
        if probs.shape != (5,):
            raise ValueError(f"Invalid prediction output shape: {probs.shape}, expected (5,)")

        if np.isnan(probs).any() or np.isinf(probs).any():
            raise ValueError("Model output contains NaN or Inf values")

        prob_sum = float(np.sum(probs))
        if abs(prob_sum - 1.0) > 0.05:
            # Re-normalize if quantization slight drift occurs
            probs = probs / (prob_sum if prob_sum > 0 else 1.0)

        class_idx = int(np.argmax(probs))
        emotion = EMOTION_CLASS_MAPPING[class_idx]
        confidence = float(probs[class_idx])

        return {
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': probs.tolist(),
            'active_backend': self.active_backend_name
        }

    def get_diagnostic_info(self):
        """Return non-sensitive internal diagnostic status for health checks & testing."""
        return {
            'active_backend': self.active_backend_name,
            'tflite_loaded': self.tflite_loaded,
            'keras_fallback_active': self.keras_fallback_active,
            'primary_path_exists': os.path.isfile(self.primary_path) if self.primary_path else False,
            'fallback_path_exists': os.path.isfile(self.fallback_path) if self.fallback_path else False,
            'num_threads': self.num_threads,
            'last_error': self.last_error
        }


def _get_or_create_predictor(model_path, backend=None):
    """Retrieve or instantiate singleton EmotionPredictor."""
    global _predictor_instance
    with _predictor_lock:
        if _predictor_instance is not None:
            # If requested path/backend matches cached instance, reuse
            if _predictor_instance.primary_path == model_path:
                return _predictor_instance
            # Re-initialize if different path requested (e.g. testing different models)

        num_threads = 4
        fallback_path = None
        try:
            from flask import current_app
            if current_app:
                num_threads = current_app.config.get('TFLITE_NUM_THREADS', 4)
                fallback_path = current_app.config.get('KERAS_FALLBACK_MODEL_PATH')
                if not backend:
                    backend = current_app.config.get('MODEL_BACKEND', 'keras')
        except Exception:
            pass

        _predictor_instance = EmotionPredictor(
            primary_path=model_path,
            backend_type=backend,
            fallback_path=fallback_path,
            num_threads=num_threads
        )
        return _predictor_instance


def predict(frame_bytes, model_path, haar_cascade_path, backend=None):
    """Run facial expression inference on a camera frame."""
    if not os.path.isfile(haar_cascade_path):
        return {
            'status': 'cascade_missing',
            'message': 'Face detection classifier not found.',
        }

    predictor = _get_or_create_predictor(model_path, backend=backend)
    if predictor.backend is None:
        return {
            'status': 'model_not_initialized',
            'message': (
                'Facial expression analysis is not available yet because '
                'the expression model has not been initialized.'
            ),
        }

    try:
        from app.ml.preprocessor import preprocess_frame
        prep = preprocess_frame(frame_bytes, haar_cascade_path)

        if prep['status'] != 'success':
            return prep

        res = predictor.predict(prep['tensor'])

        return {
            'status': 'success',
            'emotion': res['emotion'],
            'confidence': res['confidence'],
            'active_backend': res['active_backend']
        }
    except Exception as e:
        logger.error("Prediction error: %s", str(e))
        return {
            'status': 'error',
            'message': 'Internal prediction error.'
        }


def reset():
    """Reset global predictor instance (useful for unit tests)."""
    global _predictor_instance
    with _predictor_lock:
        _predictor_instance = None


def get_backend_diagnostic_info():
    """Return backend status for system diagnostics."""
    global _predictor_instance
    if _predictor_instance:
        return _predictor_instance.get_diagnostic_info()
    return {'active_backend': 'uninitialized', 'tflite_loaded': False}
