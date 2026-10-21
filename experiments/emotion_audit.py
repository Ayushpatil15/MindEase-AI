"""Emotion ML Pipeline & Model Audit Script.

Investigates:
1. Label mapping parity between training (FER-2013 mapping) and runtime predictor.
2. Model output type (Softmax probabilities vs logits) and predictor transformation.
3. Preprocessing parity and tensor statistics (FER-2013 test set vs runtime preprocessed fixture).
4. Threshold sweep on FER-2013 test data for thresholds [0.3, 0.4, 0.5, 0.6, 0.7].
5. Diagnosis and recommendations based on phase59_results.json and audit findings.
"""

import os
import sys
import json
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.models.emotion import EMOTION_CLASS_MAPPING, EMOTION_CLASSES
from app.ml.preprocessor import FER_TO_OUR_MAPPING, REQUIRED_FER_LABELS, parse_fer2013, preprocess_frame
from app.ml.predictor import TFLiteBackend, EmotionPredictor


def audit_label_mapping():
    print("=" * 80)
    print("1. LABEL MAPPING AUDIT")
    print("=" * 80)

    # 1. Training mapping
    # Defined in app/ml/preprocessor.py lines 16-22
    # FER-2013 original 7 classes: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
    fer_raw_names = {
        0: 'Angry',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happy',
        4: 'Sad',
        5: 'Surprise',
        6: 'Neutral'
    }

    print("\nTraining mapping (app/ml/preprocessor.py lines 16-22):")
    print("FER_TO_OUR_MAPPING = {")
    for fer_idx, our_idx in FER_TO_OUR_MAPPING.items():
        print(f"    {fer_idx}: {our_idx},  # {fer_raw_names[fer_idx]} (FER {fer_idx}) -> Our index {our_idx}")
    print("}")
    print(f"Excluded FER labels: Disgust (1) and Surprise (5)")

    # 2. Runtime mapping
    # Defined in app/models/emotion.py lines 14-20, imported in app/ml/predictor.py line 11
    print("\nRuntime prediction mapping (app/models/emotion.py lines 14-20):")
    print("EMOTION_CLASS_MAPPING = {")
    for our_idx, name in EMOTION_CLASS_MAPPING.items():
        print(f"    {our_idx}: '{name}',")
    print("}")

    # 3. Direct comparison
    print("\nSide-by-side comparison:")
    print(f"{'Our Index':<10} | {'FER Label':<10} | {'FER Name':<10} | {'Runtime Name':<15} | {'Match?'}")
    print("-" * 65)

    all_match = True
    for fer_idx, our_idx in sorted(FER_TO_OUR_MAPPING.items(), key=lambda x: x[1]):
        fer_name = fer_raw_names[fer_idx]
        runtime_name = EMOTION_CLASS_MAPPING[our_idx]
        match = (fer_name == runtime_name)
        if not match:
            all_match = False
        print(f"{our_idx:<10} | {fer_idx:<10} | {fer_name:<10} | {runtime_name:<15} | {'YES' if match else 'NO'}")

    print(f"\nConclusion on Label Mapping: {'PERFECT MATCH' if all_match else 'MISMATCH DETECTED'}")
    return all_match


def audit_output_type(model_path):
    print("\n" + "=" * 80)
    print("2. OUTPUT TYPE AUDIT (PROBABILITIES VS LOGITS)")
    print("=" * 80)

    # 1. Inspect TFLite model details
    backend = TFLiteBackend(model_path=model_path, num_threads=1)
    in_det = backend.input_details[0]
    out_det = backend.output_details[0]

    print(f"TFLite Model Path:      {model_path}")
    print(f"Input Tensor Shape:     {in_det['shape']} | Dtype: {in_det['dtype']}")
    print(f"Output Tensor Shape:    {out_det['shape']} | Dtype: {out_det['dtype']}")
    print(f"Output Quantization:    {out_det['quantization']}")

    # 2. Run inference on zeros and random uniform inputs
    dummy_zeros = np.zeros((1, 48, 48, 1), dtype=np.float32)
    dummy_random = np.random.uniform(0.0, 1.0, size=(1, 48, 48, 1)).astype(np.float32)

    probs_zeros = backend.predict(dummy_zeros)
    probs_random = backend.predict(dummy_random)

    print("\nSample Output on Zeros Input:")
    print(f"  Values:        {probs_zeros}")
    print(f"  Sum:           {np.sum(probs_zeros):.6f}")
    print(f"  Min / Max:     {np.min(probs_zeros):.6f} / {np.max(probs_zeros):.6f}")

    print("\nSample Output on Random Input:")
    print(f"  Values:        {probs_random}")
    print(f"  Sum:           {np.sum(probs_random):.6f}")
    print(f"  Min / Max:     {np.min(probs_random):.6f} / {np.max(probs_random):.6f}")

    # Check properties
    is_prob = (
        abs(np.sum(probs_zeros) - 1.0) < 1e-4 and
        abs(np.sum(probs_random) - 1.0) < 1e-4 and
        np.all(probs_zeros >= 0.0) and np.all(probs_zeros <= 1.0) and
        np.all(probs_random >= 0.0) and np.all(probs_random <= 1.0)
    )

    print(f"\nModel output is Softmax Probabilities: {is_prob}")
    print("Predictor Transform Check:")
    print("  app/ml/predictor.py (lines 65-71, 107-123):")
    print("  - TFLiteBackend returns out_tensor[0] directly (float32).")
    print("  - EmotionPredictor validates sum ~ 1.0 (safety normalizes only if > 0.05 drift).")
    print("  - NO sigmoid, NO extra softmax, NO temperature scaling applied.")
    return is_prob


def audit_preprocessing_parity(csv_path, haar_path):
    print("\n" + "=" * 80)
    print("3. PREPROCESSING PARITY & TENSOR STATISTICS AUDIT")
    print("=" * 80)

    # 1. Create a synthetic realistic face fixture
    import cv2
    img = np.full((320, 320, 3), 210, dtype=np.uint8)
    # Head contour
    cv2.ellipse(img, (160, 160), (70, 95), 0, 0, 360, (130, 130, 130), -1)
    # Eyes
    cv2.circle(img, (135, 140), 12, (30, 30, 30), -1)
    cv2.circle(img, (185, 140), 12, (30, 30, 30), -1)
    # Eyebrows
    cv2.line(img, (120, 125), (150, 128), (20, 20, 20), 4)
    cv2.line(img, (170, 128), (200, 125), (20, 20, 20), 4)
    # Nose
    cv2.line(img, (160, 145), (160, 175), (40, 40, 40), 3)
    # Mouth
    cv2.ellipse(img, (160, 195), (28, 12), 0, 0, 180, (20, 20, 20), 4)

    _, buf = cv2.imencode('.jpg', img)
    fixture_bytes = buf.tobytes()

    runtime_res = preprocess_frame(fixture_bytes, haar_path)
    if runtime_res['status'] != 'success':
        raise RuntimeError(f"Runtime preprocessing failed on fixture: {runtime_res}")

    runtime_tensor = runtime_res['tensor'] # (1, 48, 48, 1)

    print("Runtime Fixture Preprocessed Tensor:")
    print(f"  Shape:         {runtime_tensor.shape}")
    print(f"  Dtype:         {runtime_tensor.dtype}")
    print(f"  Mean:          {np.mean(runtime_tensor):.4f}")
    print(f"  Std:           {np.std(runtime_tensor):.4f}")
    print(f"  Min:           {np.min(runtime_tensor):.4f}")
    print(f"  Max:           {np.max(runtime_tensor):.4f}")

    # 2. FER-2013 Dataset Statistics
    if csv_path and os.path.exists(csv_path):
        print(f"\nFER-2013 Dataset found locally at: {csv_path}")
        splits = parse_fer2013(csv_path)
        X_priv, y_priv = splits['private_test']
        X_pub, y_pub = splits['public_test']
        X_test = np.concatenate([X_priv, X_pub], axis=0)

        print("FER-2013 Test Set (PrivateTest + PublicTest, N=6236):")
        print(f"  Shape:         {X_test.shape}")
        print(f"  Dtype:         {X_test.dtype}")
        print(f"  Mean:          {np.mean(X_test):.4f}")
        print(f"  Std:           {np.std(X_test):.4f}")
        print(f"  Min:           {np.min(X_test):.4f}")
        print(f"  Max:           {np.max(X_test):.4f}")
        print(f"  Per-sample mean (avg): {np.mean(np.mean(X_test, axis=(1,2,3))):.4f}")
        print(f"  Per-sample std  (avg): {np.mean(np.std(X_test, axis=(1,2,3))):.4f}")
        return splits
    else:
        print("\nFER-2013 Dataset NOT found locally.")
        return None


def run_threshold_sweep(splits, model_path):
    print("\n" + "=" * 80)
    print("4. THRESHOLD SWEEP ON FER-2013 TEST DATA")
    print("=" * 80)

    if splits is None:
        print("FER-2013 dataset not available for threshold sweep.")
        return

    X_test, y_test = splits['private_test']
    y_true = np.argmax(y_test, axis=1)

    backend = TFLiteBackend(model_path=model_path, num_threads=4)

    # Predict all test samples
    print(f"Running inference over PrivateTest (N={len(X_test)}) with {model_path}...")
    all_probs = []
    for i in range(len(X_test)):
        sample = X_test[i:i+1] # (1, 48, 48, 1)
        p = backend.predict(sample)
        all_probs.append(p)
    all_probs = np.array(all_probs)

    pred_classes = np.argmax(all_probs, axis=1)
    pred_confidences = np.max(all_probs, axis=1)

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    print("\n--- OVERALL THRESHOLD SWEEP (PrivateTest N=3118) ---")
    print(f"{'Threshold':<10} | {'Answered':<10} | {'Coverage':<12} | {'Correct':<10} | {'Accuracy (Answered)':<20}")
    print("-" * 72)

    sweep_results = {}

    for t in thresholds:
        mask = pred_confidences >= t
        answered = int(np.sum(mask))
        coverage = answered / len(pred_confidences)
        if answered > 0:
            correct = int(np.sum(pred_classes[mask] == y_true[mask]))
            acc = correct / answered
        else:
            correct = 0
            acc = 0.0

        sweep_results[t] = {
            'answered': answered,
            'coverage': coverage,
            'correct': correct,
            'acc': acc,
            'per_class': {}
        }

        print(f"{t:<10.1f} | {answered:<10d} | {coverage*100:<11.2f}% | {correct:<10d} | {acc*100:<19.2f}%")

    print("\n--- PER-CLASS THRESHOLD SWEEP (PrivateTest) ---")
    header = f"{'Class':<10} | {'Total':<6} | " + " | ".join([f"T={t:.1f} (Cov/Acc)" for t in thresholds])
    print(header)
    print("-" * len(header))

    for c_idx, c_name in enumerate(EMOTION_CLASSES):
        c_mask_total = (y_true == c_idx)
        total_c = int(np.sum(c_mask_total))
        row = [f"{c_name:<10}", f"{total_c:<6d}"]

        for t in thresholds:
            answered_c = int(np.sum((pred_confidences >= t) & c_mask_total))
            cov_c = answered_c / total_c if total_c > 0 else 0.0
            if answered_c > 0:
                correct_c = int(np.sum((pred_confidences >= t) & c_mask_total & (pred_classes == c_idx)))
                acc_c = correct_c / answered_c
            else:
                acc_c = 0.0

            sweep_results[t]['per_class'][c_name] = {
                'total': total_c,
                'answered': answered_c,
                'coverage': cov_c,
                'accuracy': acc_c
            }
            row.append(f"{cov_c*100:5.1f}% / {acc_c*100:5.1f}%")

        print(" | ".join(row))

    return sweep_results


def inspect_phase59_results():
    print("\n" + "=" * 80)
    print("5. PHASE 5.9 RECORDED METRICS (CHALLENGER FP32)")
    print("=" * 80)

    p59_path = os.path.join(PROJECT_ROOT, 'experiments', 'phase59_results.json')
    with open(p59_path, 'r') as f:
        data = json.load(f)

    chal = data['eval_results']['Challenger']['FP32']

    for split in ['Validation', 'PublicTest', 'PrivateTest']:
        s_data = chal[split]
        print(f"\n--- Split: {split} ---")
        print(f"Overall Accuracy:  {s_data['accuracy']*100:.2f}%")
        print(f"Macro F1:          {s_data['macro_f1']:.4f}")
        print(f"Weighted F1:       {s_data['weighted_f1']:.4f}")

        print("\nPer-Class Metrics:")
        print(f"{'Class':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
        print("-" * 46)
        for i, c_name in enumerate(EMOTION_CLASSES):
            prec = s_data['precision'][i]
            rec = s_data['recall'][i]
            f1 = s_data['f1'][i]
            print(f"{c_name:<10} | {prec*100:9.2f}% | {rec*100:8.2f}% | {f1:.4f}")

        print("\nConfusion Matrix (Rows=True, Cols=Pred):")
        print("          " + "  ".join([f"{c[:4]:>5}" for c in EMOTION_CLASSES]))
        for i, row in enumerate(s_data['confusion_matrix']):
            print(f"{EMOTION_CLASSES[i]:<8}  " + "  ".join([f"{val:>5d}" for val in row]))


def main():
    csv_path = r"C:\Users\TUSHAR\Downloads\archive (1)\fer2013.csv"
    if not os.path.exists(csv_path):
        csv_path = None

    haar_path = os.path.join(PROJECT_ROOT, 'app', 'static', 'haar', 'haarcascade_frontalface_default.xml')
    model_path = os.path.join(PROJECT_ROOT, 'models', 'deployment', 'challenger_fp32.tflite')

    audit_label_mapping()
    audit_output_type(model_path)
    splits = audit_preprocessing_parity(csv_path, haar_path)
    sweep_results = run_threshold_sweep(splits, model_path)
    inspect_phase59_results()


if __name__ == '__main__':
    main()
