# Facial Emotion Recognition (FER) Model & Pipeline Audit Report

**Date:** October 21, 2026  
**Artifacts Evaluated:**
- Shipped Model: `models/deployment/challenger_fp32.tflite` (Phase 5.6 Challenger Residual CNN)
- Training Benchmark: `experiments/phase59_results.json`
- Dataset Source: FER-2013 (`C:\Users\TUSHAR\Downloads\archive (1)\fer2013.csv`, N=35,887 raw; 31,338 across 5 target classes)
- Script: [experiments/emotion_audit.py](file:///e:/MindEase%20AI/experiments/emotion_audit.py)

---

## 1. Label Mapping Audit

### Training vs. Runtime Mapping

- **Training Mapping** ([app/ml/preprocessor.py:L16-L23](file:///e:/MindEase%20AI/app/ml/preprocessor.py#L16-L23)):
  ```python
  FER_TO_OUR_MAPPING = {
      0: 0,  # Angry   (FER 0) -> Our index 0
      2: 1,  # Fear    (FER 2) -> Our index 1
      3: 2,  # Happy   (FER 3) -> Our index 2
      4: 3,  # Sad     (FER 4) -> Our index 3
      6: 4,  # Neutral (FER 6) -> Our index 4
  }
  # FER classes 1 (Disgust) and 5 (Surprise) are explicitly filtered out.
  ```

- **Runtime Mapping** ([app/models/emotion.py:L14-L20](file:///e:/MindEase%20AI/app/models/emotion.py#L14-L20) & [app/ml/predictor.py:L11,L121-L123](file:///e:/MindEase%20AI/app/ml/predictor.py#L11)):
  ```python
  EMOTION_CLASS_MAPPING = {
      0: 'Angry',
      1: 'Fear',
      2: 'Happy',
      3: 'Sad',
      4: 'Neutral'
  }
  ```

### Comparison Matrix

| Model Output Index | FER-2013 Raw Label | FER Class Name | Runtime Class Name | Status |
| :---: | :---: | :---: | :---: | :---: |
| **0** | 0 | Angry | `Angry` | **MATCH** |
| **1** | 2 | Fear | `Fear` | **MATCH** |
| **2** | 3 | Happy | `Happy` | **MATCH** |
| **3** | 4 | Sad | `Sad` | **MATCH** |
| **4** | 6 | Neutral | `Neutral` | **MATCH** |

**Conclusion:** The label mappings in training and runtime are an **exact 1:1 match**. There is zero indexing shift.

---

## 2. Output Type Verification (Probabilities vs. Logits)

1. **Architecture Output Layer** ([app/ml/model.py:L133](file:///e:/MindEase%20AI/app/ml/model.py#L133)):
   ```python
   outputs = Dense(num_classes, activation='softmax')(x)
   ```
   The final dense layer specifies `activation='softmax'`.

2. **TFLite Model Output Inspection**:
   - Model Path: `models/deployment/challenger_fp32.tflite`
   - Output Tensor: Shape `(1, 5)`, `dtype=float32`, `quantization=(0.0, 0)`
   - On Zeros Input: Sum = `1.000000`, Min = `0.014223`, Max = `0.842777`
   - On Uniform Random Input: Sum = `1.000000`, Min = `0.022143`, Max = `0.389154`
   - All output values reside strictly within `[0.0, 1.0]` and sum to `1.0`.

3. **Predictor Transformations** ([app/ml/predictor.py:L65-L71,L107-L123](file:///e:/MindEase%20AI/app/ml/predictor.py#L65-L71)):
   - `TFLiteBackend.predict` extracts `out_tensor[0]` directly without modifications.
   - `EmotionPredictor.predict` performs only sanity checks (NaN/Inf validation and drift normalization if $|sum - 1.0| > 0.05$).
   - **No extra transformation** (no sigmoid, no logit-to-softmax re-application, no temperature scaling) is applied.

---

## 3. Preprocessing Parity & Tensor Statistics

### Side-by-Side Pipeline Comparison

| Step | Training (`parse_fer2013`) | Runtime (`preprocess_frame`) | Parity Status | Accuracy Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Color Conversion** | Raw 48x48 grayscale pixel strings from CSV | Decoded via `cv2.IMREAD_GRAYSCALE` (fallback to `COLOR_BGR2GRAY`) | **Identical** | None |
| **Face Detection & Crop** | Pre-cropped faces from Google search results (no detector used) | Haar Cascade (`haarcascade_frontalface_default.xml`) with **10% margin** (`ROI_MARGIN = 0.10`) | **MISMATCH** | **Moderate**: 10% padding introduces hair, neck, and room background into 48x48 input, altering perimeter features. |
| **Resize Interpolation** | Native 48x48 (no resizing needed) | `cv2.resize(..., interpolation=cv2.INTER_AREA)` | **Different** | Minor: standard area downsampling from webcam resolutions. |
| **Normalization Range** | `pixels.astype('float32') / 255.0` $\rightarrow [0.0, 1.0]$ | `(roi.astype(np.float32) / 255.0)` $\rightarrow [0.0, 1.0]$ | **Identical** | None |
| **Equalization** | None (no CLAHE or histogram equalization) | None | **Identical** | **Moderate**: Both pipelines are susceptible to uneven illumination, webcam gain, and backlight. |
| **Channel & Layout** | Single channel, `(batch, 48, 48, 1)`, `float32` | Single channel, `(1, 48, 48, 1)`, `float32` | **Identical** | None |

### Tensor Statistics Comparison

| Metric | FER-2013 Test Set ($N=6,236$) | Runtime Fixture Crop ($N=1$) |
| :--- | :---: | :---: |
| **Tensor Shape** | `(6236, 48, 48, 1)` | `(1, 48, 48, 1)` |
| **Data Type** | `float32` | `float32` |
| **Mean** | **0.4966** | **0.6274** |
| **Std Dev** | **0.2545** | **0.1839** |
| **Min Value** | 0.0000 | 0.0824 |
| **Max Value** | 1.0000 | 0.8235 |

*Note: The runtime crop demonstrates higher mean luminance and lower variance due to background margin inclusion around the detected face contour.*

---

## 4. Threshold Sweep (PrivateTest, $N=3,118$)

Evaluated using `models/deployment/challenger_fp32.tflite` on the holdout `PrivateTest` partition:

### Overall Coverage vs. Accuracy

| Threshold | Answered Samples | Coverage (%) | Correct Predictions | Accuracy on Answered (%) |
| :---: | :---: | :---: | :---: | :---: |
| **0.3** | 3,090 | 99.10% | 2,045 | 66.18% |
| **0.4** | 2,812 | 90.19% | 1,952 | 69.42% |
| **0.5** *(current)* | **2,283** | **73.22%** | **1,728** | **75.69%** |
| **0.6** | 1,867 | 59.88% | 1,514 | 81.09% |
| **0.7** | 1,514 | 48.56% | 1,309 | 86.46% |

### Per-Class Breakdown (Coverage / Accuracy on Answered)

| Class | Support | T = 0.3 | T = 0.4 | T = 0.5 *(current)* | T = 0.6 | T = 0.7 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Angry** | 491 | 99.4% / 59.0% | 89.2% / 62.8% | 69.5% / 71.6% | 54.0% / 76.6% | 40.9% / 82.1% |
| **Fear** | 528 | 98.9% / 38.3% | 85.8% / 41.5% | 62.7% / 46.8% | 46.0% / 50.2% | 32.4% / 56.7% |
| **Happy** | 879 | 99.4% / 85.5% | 96.4% / 87.4% | **89.2% / 92.0%** | 83.8% / 94.4% | 78.5% / 97.2% |
| **Sad** | 594 | 99.3% / 57.8% | 86.0% / 57.7% | 58.4% / 55.3% | 38.4% / 56.6% | 23.6% / 57.1% |
| **Neutral** | 626 | 98.4% / 76.1% | 89.9% / 80.6% | **76.7% / 86.7%** | 62.9% / 92.4% | 49.8% / 94.9% |

---

## 5. Recorded Benchmark Metrics (`phase59_results.json`)

Exact metrics recorded for `Challenger FP32` across evaluation splits:

### Summary Performance
- **Validation** ($N=2,511$): Accuracy = 65.99%, Macro F1 = 0.6289, Weighted F1 = 0.6582
- **PublicTest** ($N=3,118$): Accuracy = 62.51%, Macro F1 = 0.5942, Weighted F1 = 0.6237
- **PrivateTest** ($N=3,118$): Accuracy = 65.94%, Macro F1 = 0.6306, Weighted F1 = 0.6576

### Confusion Matrix on PrivateTest ($N=3,118$)
*(Rows = True Label, Columns = Predicted Label)*

| True Class | Total | Angry | Fear | Happy | Sad | Neutral | Recall | Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Angry** | 491 | **289** | 36 | 9 | 72 | 85 | 58.86% | 63.10% |
| **Fear** | 528 | 77 | **201** | 19 | 143 | 88 | **38.07%** | 66.78% |
| **Happy** | 879 | 25 | 24 | **750** | 22 | 58 | **85.32%** | **91.69%** |
| **Sad** | 594 | 52 | 33 | 17 | **344** | 148 | 57.91% | 49.86% |
| **Neutral** | 626 | 15 | 7 | 23 | 109 | **472** | **75.40%** | 55.46% |

### Key Confusion Insights
1. **Fear Confusion:** Fear is the weakest class (recall 38.07%). Out of 528 Fear samples, 143 (27.1%) are misclassified as Sad, 88 (16.7%) as Neutral, and 77 (14.6%) as Angry.
2. **Sad Confusion:** 148 of 594 Sad samples (24.9%) are misclassified as Neutral.
3. **Happy Superiority:** Happy has both high recall (85.32%) and high precision (91.69%), rarely confused with any other category.

---

## 6. Diagnosis & Recommendations

### Root Cause Analysis
1. **No Code / Index Bug:** The pipeline exhibits no software bugs, label inversions, or missing transforms. Output shapes, activations, and label orders match completely.
2. **Intrinsic Dataset Ambiguity (FER-2013 Limit):**
   - Human inter-annotator agreement on FER-2013 is documented in research at approximately $65\% \pm 5\%$.
   - Micro-expressions for Fear and Sadness are notoriously difficult to distinguish at 48x48 pixel resolution, explaining the heavy cross-confusion between Fear, Sad, and Neutral.
3. **Haar Crop Margin Discrepancy:**
   - FER-2013 images are tightly cropped around facial features without background margins.
   - Runtime `preprocess_frame` applies a 10% safety margin (`ROI_MARGIN = 0.10`), introducing hair, ears, and background pixels into the 48x48 frame.

### Recommended Next Steps
1. **Immediate (No Code Changes Needed):**
   - **Retain `EMOTION_MIN_CONFIDENCE = 0.5`**: As shown in the threshold sweep, at $T=0.5$, accuracy on answered samples rises from $65.94\%$ to **$75.69\%$** while preserving **$73.22\%$** coverage.
   - Continue treating the feature as experimental and decoupled from wellness scores and daily plans (already completed in Phase 3).
2. **Future Enhancement Candidate (When Model Updates are Scheduled):**
   - **Tighten Face Crop**: Test reducing `ROI_MARGIN` from `0.10` to `0.0` or `0.02` in `app/ml/preprocessor.py` to match FER-2013's tight facial bounding box.
   - **Illumination Normalization**: Evaluate CLAHE (Contrast Limited Adaptive Histogram Equalization) before resizing, which would require retraining the CNN with the same CLAHE step.
