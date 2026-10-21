# Production Model Artifact

This directory contains the production facial expression recognition model for MindEase AI.

## Shipped Model: Challenger FP32 (`models/deployment/challenger_fp32.tflite`)

- **Artifact Path**: `models/deployment/challenger_fp32.tflite`
- **Format**: TensorFlow Lite (FP32 Float32 weights and activations)
- **Status**: Production Champion (default runtime model in `app/config.py`)
- **Architecture**: 4-stage Residual CNN + Squeeze-and-Excitation (SE) Attention + Global Average Pooling
- **Input Contract**: `(1, 48, 48, 1)` Grayscale normalized to `[0.0, 1.0]` (dtype `float32`)
- **Output Contract**: 5-class probability distribution via Softmax
- **Classes (5)**:
  - `0`: Angry
  - `1`: Fear
  - `2`: Happy
  - `3`: Sad
  - `4`: Neutral
- **Parameters**: ~450k
- **Binary Size**: ~4.9 MB
- **Holdout Accuracy**: 65.94% (+10.04% improvement over legacy baseline)
- **Inference Latency**: ~0.89 ms mean CPU latency (`0.8917 ms` mean, `p95: 1.4033 ms`, `min: 0.6551 ms`, `max: 16.2022 ms` under entry `Challenger_FP32_T4` in `experiments/phase59_results.json`).
  - **Measurement Details** (strictly from `experiments/phase59_results.json`):
    - **Hardware**: CPU `Intel64 Family 6 Model 186 Stepping 2, GenuineIntel`, Platform `Windows-11-10.0.26200-SP0`, Python `3.12.10`, TensorFlow `2.16.1`, Threads: `4`.
    - **Inference-only vs. Preprocessing**: not recorded.
    - **Warm-up**: not recorded.
    - **Number of runs**: not recorded.
  - The 0.89 ms figure is a single-machine, inference-only measurement and has not been independently reproduced.

### Verification
Inference verification tests and runtime fallbacks are tested as part of the test suite:
```bash
pytest tests/test_emotion_feature.py
```
