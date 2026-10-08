"""Image Preprocessor — face detection, cropping, normalization."""

import os
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

IMG_SIZE = 48
NUM_CLASSES = 5
from app.models.emotion import EMOTION_CLASS_MAPPING

# The original FER-2013 has 7 classes: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
# We keep 0, 2, 3, 4, 6 and map them to our centralized 0-4 indices.
FER_TO_OUR_MAPPING = {
    0: 0, # Angry (FER 0) -> Angry (Our 0)
    2: 1, # Fear (FER 2) -> Fear (Our 1)
    3: 2, # Happy (FER 3) -> Happy (Our 2)
    4: 3, # Sad (FER 4) -> Sad (Our 3)
    6: 4, # Neutral (FER 6) -> Neutral (Our 4)
}
REQUIRED_FER_LABELS = set(FER_TO_OUR_MAPPING.keys())
ROI_MARGIN = 0.10


import threading

# Thread-safe CascadeClassifier cache
_cascade_cache = {}
_cascade_lock = threading.RLock()


def _get_cascade_classifier(haar_cascade_path):
    """Retrieve or instantiate cached CascadeClassifier instance."""
    with _cascade_lock:
        if haar_cascade_path not in _cascade_cache:
            _cascade_cache[haar_cascade_path] = cv2.CascadeClassifier(haar_cascade_path)
        return _cascade_cache[haar_cascade_path]


def clear_cascade_cache():
    """Clear cached CascadeClassifier instances (useful for testing and memory cleanup)."""
    with _cascade_lock:
        _cascade_cache.clear()



def preprocess_frame(frame_bytes, haar_cascade_path):
    """Preprocess a camera frame for CNN inference."""
    if not os.path.isfile(haar_cascade_path):
        return {'status': 'cascade_missing', 'message': 'Face detection classifier not found.'}
        
    try:
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        # Direct grayscale decoding avoids 3-channel color allocation and BGR2GRAY conversion
        gray = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        
        if gray is None:
            # Fallback if image buffer was color-specific format requiring standard BGR decode
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                return {'status': 'error', 'message': 'Failed to decode image buffer.'}
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        with _cascade_lock:
            cascade = _get_cascade_classifier(haar_cascade_path)
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.15,
                minNeighbors=5,
                minSize=(40, 40),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        
        if len(faces) == 0:
            return {'status': 'no_face_detected', 'message': 'No face detected in frame.'}
            
        # Select largest face by area if multiple
        if len(faces) > 1:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            
        x, y, w, h = faces[0]
        
        # 10% safety margin
        pad_w = int(w * ROI_MARGIN)
        pad_h = int(h * ROI_MARGIN)
        
        img_h, img_w = gray.shape[:2]
        x_min = max(0, x - pad_w)
        y_min = max(0, y - pad_h)
        x_max = min(img_w, x + w + pad_w)
        y_max = min(img_h, y + h + pad_h)
        
        roi = gray[y_min:y_max, x_min:x_max]
        
        # Resize to 48x48
        roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        
        # Fast float32 normalization and tensor shaping to (1, 48, 48, 1)
        tensor = (roi_resized.astype(np.float32) / 255.0).reshape(1, 48, 48, 1)
        
        return {'status': 'success', 'tensor': tensor}
        
    except Exception as e:
        logger.error("Preprocessing error: %s", str(e))
        return {'status': 'error', 'message': 'Internal preprocessing error.'}


def parse_fer2013(csv_path):
    """Parse FER-2013 CSV and filter to the 5 required classes."""
    import pandas as pd
    
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"FER-2013 CSV not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # Filter to only the 5 classes
    df = df[df['emotion'].isin(REQUIRED_FER_LABELS)].copy()
    
    # Remap labels to our centralized 0-4 format
    df['mapped_emotion'] = df['emotion'].map(FER_TO_OUR_MAPPING)
    
    from sklearn.model_selection import train_test_split
    
    X_train_raw, y_train_raw = [], []
    X_public, y_public = [], []
    X_private, y_private = [], []
    
    for _, row in df.iterrows():
        pixels = np.fromstring(row['pixels'], sep=' ').reshape(48, 48, 1)
        pixels = pixels.astype('float32') / 255.0
        
        label = np.zeros(NUM_CLASSES, dtype='float32')
        label[int(row['mapped_emotion'])] = 1.0
        
        if row['Usage'] == 'Training':
            X_train_raw.append(pixels)
            y_train_raw.append(label)
        elif row['Usage'] == 'PublicTest':
            X_public.append(pixels)
            y_public.append(label)
        elif row['Usage'] == 'PrivateTest':
            X_private.append(pixels)
            y_private.append(label)
            
    X_train_raw = np.array(X_train_raw)
    y_train_raw = np.array(y_train_raw)
    X_public = np.array(X_public)
    y_public = np.array(y_public)
    X_private = np.array(X_private)
    y_private = np.array(y_private)
    
    # Stratified 90/10 split of the Training set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_raw, y_train_raw,
        test_size=0.10,
        random_state=42,
        stratify=np.argmax(y_train_raw, axis=1)
    )
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'public_test': (X_public, y_public),
        'private_test': (X_private, y_private)
    }
