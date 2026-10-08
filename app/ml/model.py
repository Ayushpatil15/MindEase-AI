"""CNN Model — facial expression recognition.

Architecture (locked specification):
    Input(48, 48, 1)
      → Conv2D(32, 3×3, ReLU) → BatchNorm → MaxPool(2×2) → Dropout(0.25)
      → Conv2D(64, 3×3, ReLU) → BatchNorm → MaxPool(2×2) → Dropout(0.25)
      → Conv2D(128, 3×3, ReLU) → BatchNorm → MaxPool(2×2) → Dropout(0.25)
      → Flatten
      → Dense(256, ReLU) → BatchNorm → Dropout(0.50)
      → Dense(5, Softmax)
"""

import os
import logging
from tensorflow.keras.models import Sequential, Model, load_model as keras_load
from tensorflow.keras.layers import (
    Input, Conv2D, SeparableConv2D, DepthwiseConv2D, BatchNormalization, MaxPooling2D, Dropout,
    Flatten, Dense, GlobalAveragePooling2D, Add, Multiply, Reshape, Activation
)
from tensorflow.keras.optimizers import Adam
try:
    from tensorflow.keras.optimizers import AdamW
except ImportError:
    AdamW = None

logger = logging.getLogger(__name__)

NUM_CLASSES = 5


def build_model():
    """Build and compile the facial expression CNN."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.50),

        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def _se_block(input_tensor, ratio=8):
    """Squeeze-and-Excitation channel attention block."""
    filters = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, filters))(se)
    se = Dense(max(filters // ratio, 8), activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    return Multiply()([input_tensor, se])


def _residual_block(x, filters, stride=1):
    """Residual convolutional block with SE channel attention."""
    shortcut = x
    
    x_conv = Conv2D(filters, (3, 3), strides=stride, padding='same')(x)
    x_conv = BatchNormalization()(x_conv)
    x_conv = Activation('relu')(x_conv)
    
    x_conv = Conv2D(filters, (3, 3), strides=1, padding='same')(x_conv)
    x_conv = BatchNormalization()(x_conv)
    
    x_conv = _se_block(x_conv)
    
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
        
    x_out = Add()([shortcut, x_conv])
    x_out = Activation('relu')(x_out)
    return x_out


def build_challenger_model(input_shape=(48, 48, 1), num_classes=NUM_CLASSES):
    """Build and compile the Phase 5.6 Challenger Residual CNN model."""
    inputs = Input(shape=input_shape)
    
    # Stem
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Residual Stage 1 (32 filters)
    x = _residual_block(x, 32)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Residual Stage 2 (64 filters)
    x = _residual_block(x, 64)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Residual Stage 3 (128 filters)
    x = _residual_block(x, 128)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.30)(x)
    
    # Residual Stage 4 (256 filters)
    x = _residual_block(x, 256)
    x = Dropout(0.35)(x)
    
    # Classification Head with Global Average Pooling
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.40)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='fer2013_challenger_cnn')
    
    opt = AdamW(learning_rate=1e-3, weight_decay=1e-4) if AdamW is not None else Adam(learning_rate=1e-3)
        
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_low_latency_model(input_shape=(48, 48, 1), num_classes=NUM_CLASSES):
    """Build and compile the Phase 5.7 Low-Latency Depthwise-Separable CNN model."""
    inputs = Input(shape=input_shape)
    
    # Stem Conv2D
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Block 1 (SeparableConv 32)
    x = SeparableConv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.20)(x)
    
    # Block 2 (SeparableConv 64)
    x = SeparableConv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Block 3 (SeparableConv 128)
    x = SeparableConv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Block 4 (SeparableConv 128)
    x = SeparableConv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.30)(x)
    
    # Classification Head with Global Average Pooling
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.35)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='fer2013_low_latency_cnn')
    
    opt = AdamW(learning_rate=1e-3, weight_decay=1e-4) if AdamW is not None else Adam(learning_rate=1e-3)
        
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def _inverted_residual_block(x, expansion_factor, out_channels):
    """MobileNetV2-style inverted residual block."""
    in_channels = x.shape[-1]
    expanded_channels = int(in_channels * expansion_factor)
    
    # 1. 1x1 Expansion Conv
    x_conv = Conv2D(expanded_channels, (1, 1), padding='same')(x)
    x_conv = BatchNormalization()(x_conv)
    x_conv = Activation('relu')(x_conv)
    
    # 2. 3x3 Depthwise Conv
    x_conv = DepthwiseConv2D((3, 3), padding='same')(x_conv)
    x_conv = BatchNormalization()(x_conv)
    x_conv = Activation('relu')(x_conv)
    
    # 3. 1x1 Linear Projection Conv (no activation)
    x_conv = Conv2D(out_channels, (1, 1), padding='same')(x_conv)
    x_conv = BatchNormalization()(x_conv)
    
    # 4. Residual shortcut if dimension-compatible
    if in_channels == out_channels:
        return Add()([x, x_conv])
    return x_conv


def build_hybrid_model(input_shape=(48, 48, 1), num_classes=NUM_CLASSES):
    """Build and compile the Phase 5.8 Tiny Residual / Depthwise Hybrid model."""
    inputs = Input(shape=input_shape)
    
    # Stem Conv2D
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Block 1 (Inverted Residual 32, E=2.0)
    x = _inverted_residual_block(x, expansion_factor=2.0, out_channels=32)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.20)(x)
    
    # Block 2 (Inverted Residual 32, E=2.0, Residual Add)
    x = _inverted_residual_block(x, expansion_factor=2.0, out_channels=32)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Block 3 (Inverted Residual 64, E=2.5)
    x = _inverted_residual_block(x, expansion_factor=2.5, out_channels=64)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Block 4 (Inverted Residual 64, E=2.0, Residual Add)
    x = _inverted_residual_block(x, expansion_factor=2.0, out_channels=64)
    x = Dropout(0.30)(x)
    
    # Classification Head with Global Average Pooling
    x = GlobalAveragePooling2D()(x)
    x = Dense(48, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.30)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='fer2013_hybrid_cnn')
    
    opt = AdamW(learning_rate=1e-3, weight_decay=1e-4) if AdamW is not None else Adam(learning_rate=1e-3)
        
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def load_model(model_path):
    """Load a trained model from an .h5 file."""
    if not os.path.isfile(model_path):
        logger.warning('Model file not found at %s', model_path)
        return None

    try:
        return keras_load(model_path)
    except Exception as e:
        logger.error('Failed to load model from %s: %s', model_path, e)
        return None





