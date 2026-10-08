# MindEase AI Model Artifacts

This directory holds the trained facial expression recognition CNN models.
Model binary files (`.h5`) are gitignored as per repository policy.

## Baseline Model (`fer2013_cnn.h5`)
- **Phase**: 5.5 Baseline
- **Architecture**: 3-stage Sequential CNN + Flatten + Dense 256
- **Input shape**: `(48, 48, 1)` Grayscale
- **Classes**: 5 (`Angry`, `Fear`, `Happy`, `Sad`, `Neutral`)
- **Params**: ~620k
- **Status**: Immutable production baseline

## Challenger Model (`fer2013_challenger_cnn.h5`)
- **Phase**: 5.6 Challenger
- **Architecture**: 4-stage Residual CNN + SE Attention + Global Average Pooling
- **Input shape**: `(48, 48, 1)` Grayscale (100% inference contract compatible)
- **Classes**: 5 (`Angry`, `Fear`, `Happy`, `Sad`, `Neutral`)
- **Params**: ~450k
- **Training**: Real-time Data Augmentation, `ReduceLROnPlateau`, `EarlyStopping`
- **Training command**:
  ```bash
  python -m app.ml.train --data path/to/fer2013.csv --model-type challenger --augment --epochs 50 --output models/fer2013_challenger_cnn.h5
  ```

## Low-Latency Model (`fer2013_low_latency_cnn.h5`)
- **Phase**: 5.7 Low-Latency Optimization
- **Architecture**: 4-stage Depthwise-Separable CNN + Global Average Pooling + Dense 64
- **Input shape**: `(48, 48, 1)` Grayscale (100% inference contract compatible)
- **Classes**: 5 (`Angry`, `Fear`, `Happy`, `Sad`, `Neutral`)
- **Params**: ~41k (0.59 MB file size — 92% smaller than baseline)
- **Training**: Real-time Data Augmentation, `ReduceLROnPlateau`, `EarlyStopping`
- **Training command**:
  ```bash
  python -m app.ml.train --data path/to/fer2013.csv --model-type low_latency --augment --epochs 50 --output models/fer2013_low_latency_cnn.h5
  ```

## Hybrid Model (`fer2013_hybrid_cnn.h5`)
- **Phase**: 5.8 Tiny Residual / Depthwise Hybrid
- **Architecture**: MobileNetV2-style 4-block Inverted Residual Bottleneck (MBConv) + Global Average Pooling + Dense 48
- **Input shape**: `(48, 48, 1)` Grayscale (100% inference contract compatible)
- **Classes**: 5 (`Angry`, `Fear`, `Happy`, `Sad`, `Neutral`)
- **Params**: 43,605 (0.70 MB file size — 90.3% smaller than baseline)
- **Performance**: PrivateTest Accuracy 62.25% (+6.35%), PrivateTest Macro F1 59.47% (+6.18%)
- **Latency**: CPU Single-Sample Mean 56.38 ms (Failed Gate 4 <15.0 ms threshold; NOT promoted)
- **Training command**:
  ```bash
  python scripts/phase58_experiment.py
  ```

## Phase 5.9 Deployment & TFLite Optimization

Deployment artifacts location: `models/deployment/`

### Artifacts Overview
- **Hybrid TFLite Runtimes** (`hybrid_fp32.tflite`, `hybrid_fp16.tflite`, `hybrid_dynamic_range.tflite`, `hybrid_int8.tflite`): Sizes 0.07 MB – 0.16 MB. PrivateTest Accuracy 62.25% (+6.35% vs baseline), CPU Latency ~0.89 ms – 4.31 ms. Eligible for promotion.
- **Low-Latency TFLite Runtimes** (`low_latency_*.tflite`): Sizes 0.06 MB – 0.16 MB. Latency ~0.08 ms – 0.61 ms. Accuracy failed Gate 1 & 2.
- **Challenger TFLite Runtimes** (`challenger_fp32.tflite`, `challenger_fp16.tflite`, `challenger_dynamic_range.tflite`, `challenger_int8.tflite`): Sizes 1.27 MB – 4.90 MB. PrivateTest Accuracy **65.94%** (+10.04% vs baseline), PrivateTest Macro F1 **63.06%** (+9.77% vs baseline). **CPU Single-Sample Latency = 0.89 ms** (threads=4, FP32 TFLite) / 0.70 ms (threads=4, Full INT8 TFLite). **PASSED ALL 5 PROMOTION GATES**.

### Benchmark Command
```bash
python scripts/phase59_experiment.py
```
