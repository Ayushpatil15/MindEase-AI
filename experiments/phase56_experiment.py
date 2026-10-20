"""Phase 5.6 — Challenger Training, Evaluation, and Benchmarking.

This is a standalone experiment script. It does NOT modify app/config.py,
does NOT overwrite models/fer2013_cnn.h5, and does NOT promote automatically.
"""

import os
import sys
import time
import json
import hashlib
import argparse

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)


def md5_file(path):
    """Return MD5 hex digest for a file."""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Phase 5.6 Challenger Experiment')
    parser.add_argument('--data', required=True, help='Path to fer2013.csv')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=12)
    args = parser.parse_args()

    import numpy as np
    import tensorflow as tf

    # ================================================================
    # 0. ENVIRONMENT & PRE-FLIGHT CHECKS
    # ================================================================
    print("=" * 70)
    print("PHASE 5.6 — CHALLENGER EXPERIMENT")
    print("=" * 70)

    print(f"\nTensorFlow version : {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Detected GPU(s)   : {gpus}")
    if gpus:
        for gpu in gpus:
            print(f"  GPU detail      : {gpu}")
    else:
        print("  WARNING: No GPU detected — training will use CPU only.")

    # Seed
    tf.keras.utils.set_random_seed(42)
    np.random.seed(42)
    print(f"Random seed       : 42")

    baseline_path = os.path.join(PROJECT_ROOT, 'models', 'fer2013_cnn.h5')
    challenger_path = os.path.join(PROJECT_ROOT, 'models', 'fer2013_challenger_cnn.h5')
    config_path = os.path.join(PROJECT_ROOT, 'app', 'config.py')

    # Record baseline MD5 BEFORE experiment
    baseline_md5_before = md5_file(baseline_path) if os.path.exists(baseline_path) else 'NOT FOUND'
    config_md5_before = md5_file(config_path) if os.path.exists(config_path) else 'NOT FOUND'

    print(f"\nBaseline artifact : {baseline_path}")
    print(f"Baseline MD5      : {baseline_md5_before}")
    print(f"Config MD5        : {config_md5_before}")
    print(f"Challenger output : {challenger_path}")

    # ================================================================
    # 1. PARSE DATASET (using existing parse_fer2013)
    # ================================================================
    print("\n" + "=" * 70)
    print("1. DATASET PARSING")
    print("=" * 70)

    from app.ml.preprocessor import parse_fer2013
    from app.models.emotion import EMOTION_CLASSES

    splits = parse_fer2013(args.data)
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_public, y_public = splits['public_test']
    X_private, y_private = splits['private_test']

    print(f"Train      : {X_train.shape}")
    print(f"Validation : {X_val.shape}")
    print(f"PublicTest : {X_public.shape}")
    print(f"PrivateTest: {X_private.shape}")

    for name, y in [('Train', y_train), ('Validation', y_val), ('PublicTest', y_public), ('PrivateTest', y_private)]:
        indices = np.argmax(y, axis=1)
        unique, counts = np.unique(indices, return_counts=True)
        dist = {EMOTION_CLASSES[i]: int(c) for i, c in zip(unique, counts)}
        print(f"  {name} class distribution: {dist}")

    # ================================================================
    # 2. TRAIN CHALLENGER
    # ================================================================
    print("\n" + "=" * 70)
    print("2. CHALLENGER TRAINING")
    print("=" * 70)

    from app.ml.model import build_challenger_model
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    model = build_challenger_model()
    model.summary()

    param_count = model.count_params()
    print(f"\nChallenger parameter count: {param_count:,}")

    # Class weights from training data ONLY
    y_train_indices = np.argmax(y_train, axis=1)
    weights = compute_class_weight('balanced', classes=np.unique(y_train_indices), y=y_train_indices)
    class_weight_dict = {i: float(w) for i, w in enumerate(weights)}
    print(f"Class weights: {class_weight_dict}")

    # Callbacks — all monitor VALIDATION only
    callbacks = [
        ModelCheckpoint(
            filepath=challenger_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # Augmentation on training data ONLY
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.10,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    train_flow = datagen.flow(X_train, y_train, batch_size=args.batch_size)

    train_start = time.time()
    history = model.fit(
        train_flow,
        steps_per_epoch=len(X_train) // args.batch_size,
        epochs=args.epochs,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        callbacks=callbacks
    )
    train_time = time.time() - train_start
    print(f"\nTraining completed in {train_time:.1f} seconds ({train_time/60:.1f} minutes)")
    print(f"Epochs actually run: {len(history.history['loss'])}")

    # ================================================================
    # 3. VERIFY CHALLENGER ARTIFACT
    # ================================================================
    print("\n" + "=" * 70)
    print("3. CHALLENGER ARTIFACT VERIFICATION")
    print("=" * 70)

    assert os.path.exists(challenger_path), f"Challenger not found at {challenger_path}"
    challenger_size_mb = os.path.getsize(challenger_path) / (1024 * 1024)
    challenger_md5 = md5_file(challenger_path)
    print(f"File exists       : True")
    print(f"File size         : {challenger_size_mb:.2f} MB")
    print(f"MD5               : {challenger_md5}")

    # Load and verify
    from app.ml.model import load_model
    loaded = load_model(challenger_path)
    assert loaded is not None, "Failed to load challenger model"
    print(f"Input shape       : {loaded.input_shape}")
    print(f"Output shape      : {loaded.output_shape}")
    assert loaded.input_shape == (None, 48, 48, 1), f"Bad input shape: {loaded.input_shape}"
    assert loaded.output_shape == (None, 5), f"Bad output shape: {loaded.output_shape}"
    print("Architecture verification: PASSED")

    # ================================================================
    # 4. EVALUATE BOTH MODELS
    # ================================================================
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        accuracy_score, f1_score, precision_score, recall_score
    )

    def full_evaluate(model_obj, X, y, split_name):
        """Evaluate model on a split and return dict of metrics."""
        preds = model_obj.predict(X, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        y_true = np.argmax(y, axis=1)

        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        per_class_p = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_r = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f = f1_score(y_true, y_pred, average=None, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=EMOTION_CLASSES, digits=4)

        result = {
            'split': split_name,
            'accuracy': float(acc),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'per_class_precision': {EMOTION_CLASSES[i]: float(per_class_p[i]) for i in range(5)},
            'per_class_recall': {EMOTION_CLASSES[i]: float(per_class_r[i]) for i in range(5)},
            'per_class_f1': {EMOTION_CLASSES[i]: float(per_class_f[i]) for i in range(5)},
            'confusion_matrix': cm.tolist(),
            'report': report
        }

        print(f"\n--- {split_name} ---")
        print(f"Accuracy    : {acc:.4f}")
        print(f"Macro F1    : {macro_f1:.4f}")
        print(f"Weighted F1 : {weighted_f1:.4f}")
        print(report)
        print("Confusion Matrix:")
        print(cm)
        return result

    # --- Baseline Evaluation ---
    print("\n" + "=" * 70)
    print("4A. BASELINE EVALUATION")
    print("=" * 70)
    baseline_model = load_model(baseline_path)
    assert baseline_model is not None, "Failed to load baseline model"
    print(f"Baseline params   : {baseline_model.count_params():,}")

    baseline_metrics = {}
    for name, X, y in [('Validation', X_val, y_val), ('PublicTest', X_public, y_public), ('PrivateTest', X_private, y_private)]:
        baseline_metrics[name] = full_evaluate(baseline_model, X, y, f"BASELINE {name}")

    # --- Challenger Evaluation ---
    print("\n" + "=" * 70)
    print("4B. CHALLENGER EVALUATION")
    print("=" * 70)
    challenger_model = load_model(challenger_path)
    assert challenger_model is not None, "Failed to load challenger model"
    print(f"Challenger params : {challenger_model.count_params():,}")

    challenger_metrics = {}
    for name, X, y in [('Validation', X_val, y_val), ('PublicTest', X_public, y_public), ('PrivateTest', X_private, y_private)]:
        challenger_metrics[name] = full_evaluate(challenger_model, X, y, f"CHALLENGER {name}")

    # ================================================================
    # 5. LATENCY BENCHMARK
    # ================================================================
    print("\n" + "=" * 70)
    print("5. LATENCY BENCHMARK")
    print("=" * 70)

    dummy_input = np.random.rand(1, 48, 48, 1).astype(np.float32)
    WARMUP = 10
    TIMED = 100

    def benchmark_latency(model_obj, model_name, device):
        """Benchmark inference latency on a specific device."""
        with tf.device(device):
            # Warm-up
            for _ in range(WARMUP):
                model_obj.predict(dummy_input, verbose=0)

            # Timed runs
            times = []
            for _ in range(TIMED):
                t0 = time.perf_counter()
                model_obj.predict(dummy_input, verbose=0)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)  # ms

        mean_ms = np.mean(times)
        p95_ms = np.percentile(times, 95)
        print(f"  {model_name} [{device}]: mean={mean_ms:.2f}ms  p95={p95_ms:.2f}ms")
        return {'mean_ms': float(mean_ms), 'p95_ms': float(p95_ms)}

    latency = {'baseline': {}, 'challenger': {}}

    # CPU benchmark
    print("\nCPU Latency (batch_size=1):")
    latency['baseline']['cpu'] = benchmark_latency(baseline_model, 'Baseline', '/CPU:0')
    latency['challenger']['cpu'] = benchmark_latency(challenger_model, 'Challenger', '/CPU:0')

    # GPU benchmark (if available)
    if gpus:
        print("\nGPU Latency (batch_size=1):")
        latency['baseline']['gpu'] = benchmark_latency(baseline_model, 'Baseline', '/GPU:0')
        latency['challenger']['gpu'] = benchmark_latency(challenger_model, 'Challenger', '/GPU:0')
    else:
        print("\nGPU Latency: SKIPPED (no GPU detected)")

    # ================================================================
    # 6. SIDE-BY-SIDE COMPARISON
    # ================================================================
    print("\n" + "=" * 70)
    print("6. SIDE-BY-SIDE COMPARISON")
    print("=" * 70)

    baseline_size_mb = os.path.getsize(baseline_path) / (1024 * 1024)
    print(f"\n{'Metric':<40} {'Baseline':>12} {'Challenger':>12} {'Delta':>12}")
    print("-" * 78)
    print(f"{'Parameter count':<40} {baseline_model.count_params():>12,} {challenger_model.count_params():>12,} {'':>12}")
    print(f"{'Model size (MB)':<40} {baseline_size_mb:>12.2f} {challenger_size_mb:>12.2f} {'':>12}")

    for split in ['Validation', 'PublicTest', 'PrivateTest']:
        bm = baseline_metrics[split]
        cm = challenger_metrics[split]
        d_acc = cm['accuracy'] - bm['accuracy']
        d_mf1 = cm['macro_f1'] - bm['macro_f1']
        d_wf1 = cm['weighted_f1'] - bm['weighted_f1']
        print(f"\n  --- {split} ---")
        print(f"  {'Accuracy':<38} {bm['accuracy']:>12.4f} {cm['accuracy']:>12.4f} {d_acc:>+12.4f}")
        print(f"  {'Macro F1':<38} {bm['macro_f1']:>12.4f} {cm['macro_f1']:>12.4f} {d_mf1:>+12.4f}")
        print(f"  {'Weighted F1':<38} {bm['weighted_f1']:>12.4f} {cm['weighted_f1']:>12.4f} {d_wf1:>+12.4f}")

    # Per-class PrivateTest comparison
    print(f"\n  --- PrivateTest Per-Class F1 ---")
    print(f"  {'Class':<20} {'Baseline F1':>12} {'Challenger F1':>14} {'Delta':>12}")
    print("  " + "-" * 60)
    bm_pt = baseline_metrics['PrivateTest']
    cm_pt = challenger_metrics['PrivateTest']
    for cls in EMOTION_CLASSES:
        bf1 = bm_pt['per_class_f1'][cls]
        cf1 = cm_pt['per_class_f1'][cls]
        d = cf1 - bf1
        print(f"  {cls:<20} {bf1:>12.4f} {cf1:>14.4f} {d:>+12.4f}")

    # Latency comparison
    print(f"\n  --- Latency (ms) ---")
    for device_key in ['cpu', 'gpu']:
        if device_key in latency['baseline']:
            bl = latency['baseline'][device_key]
            cl = latency['challenger'][device_key]
            label = device_key.upper()
            print(f"  {label + ' Mean':<38} {bl['mean_ms']:>12.2f} {cl['mean_ms']:>12.2f}")
            print(f"  {label + ' P95':<38} {bl['p95_ms']:>12.2f} {cl['p95_ms']:>12.2f}")

    # ================================================================
    # 7. PROMOTION GATE EVALUATION
    # ================================================================
    print("\n" + "=" * 70)
    print("7. PROMOTION GATE EVALUATION")
    print("=" * 70)

    gate_results = {}

    # Gate 1: PrivateTest accuracy >= baseline + 0.05
    b_acc = bm_pt['accuracy']
    c_acc = cm_pt['accuracy']
    delta_acc = c_acc - b_acc
    gate1 = delta_acc >= 0.05
    gate_results['privatetest_accuracy'] = {
        'baseline': b_acc, 'challenger': c_acc, 'delta': delta_acc,
        'threshold': 0.05, 'passed': gate1
    }
    print(f"\n  Gate 1 — PrivateTest Accuracy >= baseline + 0.05")
    print(f"    Baseline: {b_acc:.4f}  Challenger: {c_acc:.4f}  Delta: {delta_acc:+.4f}  {'PASS' if gate1 else 'FAIL'}")

    # Gate 2: PrivateTest Macro F1 >= baseline + 0.05
    b_mf1 = bm_pt['macro_f1']
    c_mf1 = cm_pt['macro_f1']
    delta_mf1 = c_mf1 - b_mf1
    gate2 = delta_mf1 >= 0.05
    gate_results['privatetest_macro_f1'] = {
        'baseline': b_mf1, 'challenger': c_mf1, 'delta': delta_mf1,
        'threshold': 0.05, 'passed': gate2
    }
    print(f"\n  Gate 2 — PrivateTest Macro F1 >= baseline + 0.05")
    print(f"    Baseline: {b_mf1:.4f}  Challenger: {c_mf1:.4f}  Delta: {delta_mf1:+.4f}  {'PASS' if gate2 else 'FAIL'}")

    # Gate 3: No per-class F1 decrease > 0.02 absolute
    gate3 = True
    gate3_details = {}
    print(f"\n  Gate 3 — No per-class F1 decrease > 0.02 absolute")
    for cls in EMOTION_CLASSES:
        bf1 = bm_pt['per_class_f1'][cls]
        cf1 = cm_pt['per_class_f1'][cls]
        d = cf1 - bf1
        cls_pass = d >= -0.02
        gate3_details[cls] = {'baseline': bf1, 'challenger': cf1, 'delta': d, 'passed': cls_pass}
        if not cls_pass:
            gate3 = False
        print(f"    {cls:<12}: Baseline={bf1:.4f}  Challenger={cf1:.4f}  Delta={d:+.4f}  {'PASS' if cls_pass else 'FAIL'}")
    gate_results['per_class_f1'] = {'passed': gate3, 'details': gate3_details}

    # Gate 4: Latency and size
    cpu_mean = latency['challenger']['cpu']['mean_ms']
    gate4_latency = cpu_mean < 15.0
    gate4_size = challenger_size_mb < 20.0
    gate4 = gate4_latency and gate4_size
    gate_results['latency_size'] = {
        'cpu_mean_ms': cpu_mean, 'latency_pass': gate4_latency,
        'size_mb': challenger_size_mb, 'size_pass': gate4_size,
        'passed': gate4
    }
    print(f"\n  Gate 4 — Latency & Size")
    print(f"    CPU mean latency: {cpu_mean:.2f}ms (< 15ms: {'PASS' if gate4_latency else 'FAIL'})")
    print(f"    Model size: {challenger_size_mb:.2f}MB (< 20MB: {'PASS' if gate4_size else 'FAIL'})")

    # Gate 5: Regression — deferred to pytest run
    gate5 = None  # Will be determined by pytest
    print(f"\n  Gate 5 — Regression (pytest) : DEFERRED (run separately)")

    # Overall
    all_pass = gate1 and gate2 and gate3 and gate4
    recommendation = "PROMOTE" if all_pass else "DO NOT PROMOTE"
    if gate5 is not None and not gate5:
        recommendation = "DO NOT PROMOTE"

    print(f"\n  ===========================")
    print(f"  RECOMMENDATION: {recommendation}")
    if not all_pass:
        failures = []
        if not gate1: failures.append("PrivateTest accuracy")
        if not gate2: failures.append("PrivateTest Macro F1")
        if not gate3: failures.append("Per-class F1")
        if not gate4: failures.append("Latency/Size")
        print(f"  Failed gates: {', '.join(failures)}")
    print(f"  ===========================")
    print(f"\n  NOTE: This is ONLY a recommendation.")
    print(f"  app/config.py has NOT been modified.")
    print(f"  MODEL_PATH has NOT been changed.")
    print(f"  No automatic promotion has occurred.")

    # ================================================================
    # 8. INTEGRITY VERIFICATION
    # ================================================================
    print("\n" + "=" * 70)
    print("8. POST-EXPERIMENT INTEGRITY VERIFICATION")
    print("=" * 70)

    baseline_md5_after = md5_file(baseline_path)
    config_md5_after = md5_file(config_path)

    baseline_intact = baseline_md5_before == baseline_md5_after
    config_intact = config_md5_before == config_md5_after

    print(f"  Baseline MD5 before : {baseline_md5_before}")
    print(f"  Baseline MD5 after  : {baseline_md5_after}")
    print(f"  Baseline intact     : {'YES' if baseline_intact else '*** VIOLATION ***'}")
    print(f"  Config MD5 before   : {config_md5_before}")
    print(f"  Config MD5 after    : {config_md5_after}")
    print(f"  Config intact       : {'YES' if config_intact else '*** VIOLATION ***'}")

    assert baseline_intact, "CRITICAL: Baseline model was modified!"
    assert config_intact, "CRITICAL: app/config.py was modified!"

    print("\n" + "=" * 70)
    print("PHASE 5.6 EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
