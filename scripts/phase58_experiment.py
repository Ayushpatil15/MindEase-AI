"""Phase 5.8 Experiment Script — Candidate C (Tiny Residual / Depthwise Hybrid CNN).

Trains, evaluates, benchmarks latency, and checks promotion gates for Candidate C.
"""

import os
import sys
import time
import numpy as np

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_experiment():
    print("=" * 80)
    print("PHASE 5.8 — CANDIDATE C (TINY RESIDUAL / DEPTHWISE HYBRID) EXPERIMENT")
    print("=" * 80)

    # 1. Environment & Hardware Verification
    import tensorflow as tf
    print(f"\n[1/7] ENVIRONMENT VERIFICATION")
    print(f"TensorFlow Version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Detected GPU: {gpus[0].name}")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as e:
                print(f"Memory growth set failed: {e}")
    else:
        print("Detected GPU: NONE (CPU mode)")

    from app.ml.preprocessor import parse_fer2013, FER_TO_OUR_MAPPING
    print(f"Parser Source: app/ml/preprocessor.py (parse_fer2013)")
    print(f"Class Mapping (FER -> Our 0-4): {FER_TO_OUR_MAPPING}")

    # 2. Dataset Load & Split Verification
    print(f"\n[2/7] DATASET SPLIT VERIFICATION")
    csv_path = None
    possible_paths = [
        "C:\\Users\\TUSHAR\\Downloads\\archive (1)\\fer2013.csv",
        "/mnt/c/Users/TUSHAR/Downloads/archive (1)/fer2013.csv",
        "fer2013.csv"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            csv_path = p
            break

    if csv_path is None:
        raise FileNotFoundError(f"Could not locate fer2013.csv in any of {possible_paths}")

    print(f"Loading fer2013 from: {csv_path}")
    splits = parse_fer2013(csv_path)

    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_public, y_public = splits['public_test']
    X_private, y_private = splits['private_test']

    print(f"Train set:       {X_train.shape[0]} samples, shape {X_train.shape}")
    print(f"Validation set:  {X_val.shape[0]} samples, shape {X_val.shape}")
    print(f"PublicTest set:  {X_public.shape[0]} samples, shape {X_public.shape}")
    print(f"PrivateTest set: {X_private.shape[0]} samples, shape {X_private.shape}")

    from app.models.emotion import EMOTION_CLASSES
    print("\nClass distribution per split:")
    for split_name, (X_s, y_s) in [('Train', (X_train, y_train)), ('Val', (X_val, y_val)),
                                   ('PublicTest', (X_public, y_public)), ('PrivateTest', (X_private, y_private))]:
        labels = np.argmax(y_s, axis=1)
        counts = [int(np.sum(labels == i)) for i in range(len(EMOTION_CLASSES))]
        counts_str = ", ".join([f"{EMOTION_CLASSES[i]}: {counts[i]}" for i in range(len(EMOTION_CLASSES))])
        print(f"  {split_name:12s} ({len(y_s)}): {counts_str}")

    # Enforce split counts
    assert len(X_train) == 22591, f"Expected 22,591 train samples, got {len(X_train)}"
    assert len(X_val) == 2511, f"Expected 2,511 val samples, got {len(X_val)}"
    assert len(X_public) == 3118, f"Expected 3,118 public samples, got {len(X_public)}"
    assert len(X_private) == 3118, f"Expected 3,118 private samples, got {len(X_private)}"

    # 3. Model Architecture Verification
    print(f"\n[3/7] MODEL ARCHITECTURE VERIFICATION")
    from app.ml.model import build_hybrid_model, AdamW
    if AdamW is not None:
        print("AdamW Optimizer: AVAILABLE")
    else:
        print("AdamW Optimizer: UNAVAILABLE (using Adam fallback)")

    tf.keras.utils.set_random_seed(42)
    np.random.seed(42)

    hybrid_model = build_hybrid_model()
    hybrid_model.summary()

    trainable_params = sum([tf.keras.backend.count_params(w) for w in hybrid_model.trainable_weights])
    non_trainable_params = sum([tf.keras.backend.count_params(w) for w in hybrid_model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    print(f"\nParameter Counts:")
    print(f"  Trainable Params:     {trainable_params:,}")
    print(f"  Non-Trainable Params: {non_trainable_params:,}")
    print(f"  Total Params:         {total_params:,}")

    # 4. Training
    print(f"\n[4/7] TRAINING HYBRID CANDIDATE")
    output_path = "models/fer2013_hybrid_cnn.h5"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    from sklearn.utils.class_weight import compute_class_weight
    y_train_indices = np.argmax(y_train, axis=1)
    weights = compute_class_weight('balanced', classes=np.unique(y_train_indices), y=y_train_indices)
    class_weight_dict = {i: float(w) for i, w in enumerate(weights)}
    print(f"Class Weight Dictionary: {class_weight_dict}")

    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
    callbacks = [
        ModelCheckpoint(
            filepath=output_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=12,
            restore_best_weights=True,
            verbose=1
        )
    ]

    print("Generating pre-augmented training dataset for fast CPU execution...")
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(
        rotation_range=12,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.10,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Generate 1 augmented version of each training sample
    X_train_aug_list = [X_train]
    y_train_aug_list = [y_train]
    
    # Generate batch by batch
    aug_flow = datagen.flow(X_train, y_train, batch_size=len(X_train), shuffle=False, seed=42)
    X_aug_single, y_aug_single = next(aug_flow)
    X_train_aug_list.append(X_aug_single)
    y_train_aug_list.append(y_aug_single)
    
    X_train_combined = np.concatenate(X_train_aug_list, axis=0)
    y_train_combined = np.concatenate(y_train_aug_list, axis=0)
    print(f"Combined pre-augmented training set shape: {X_train_combined.shape}")

    start_time = time.time()
    history = hybrid_model.fit(
        X_train_combined, y_train_combined,
        batch_size=64,
        epochs=50,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    train_duration = time.time() - start_time
    print(f"Training completed in {train_duration:.1f} seconds.")

    # 5. Post-Training Verification
    print(f"\n[5/7] POST-TRAINING ARTIFACT VERIFICATION")
    assert os.path.exists(output_path), f"Artifact missing at {output_path}"
    artifact_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Artifact path: {output_path}")
    print(f"Artifact size: {artifact_size_mb:.2f} MB")

    from app.ml.model import load_model
    reloaded_model = load_model(output_path)
    assert reloaded_model is not None, "Failed to reload hybrid model artifact!"
    print("Successfully reloaded model artifact.")
    print(f"Reloaded Input shape:  {reloaded_model.input_shape}")
    print(f"Reloaded Output shape: {reloaded_model.output_shape}")

    dummy_input = np.zeros((1, 48, 48, 1), dtype=np.float32)
    dummy_pred = reloaded_model.predict(dummy_input, verbose=0)
    print(f"Sample prediction shape: {dummy_pred.shape}")
    print(f"Sample prediction sum:   {np.sum(dummy_pred):.6f} (softmax verified)")

    # 6. Evaluation of Baseline vs Hybrid
    print(f"\n[6/7] EVALUATION OF BASELINE VS HYBRID")
    from app.ml.model import load_model
    baseline_path = "models/fer2013_cnn.h5"
    assert os.path.exists(baseline_path), f"Baseline missing at {baseline_path}"
    baseline_model = load_model(baseline_path)
    baseline_size_mb = os.path.getsize(baseline_path) / (1024 * 1024)

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

    def evaluate_model(model, name, X_s, y_s):
        preds = model.predict(X_s, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        y_true = np.argmax(y_s, axis=1)

        assert len(y_true) == len(y_pred)
        cm = confusion_matrix(y_true, y_pred)
        assert cm.sum() == len(y_true)

        acc = accuracy_score(y_true, y_pred)
        assert abs((np.trace(cm) / len(y_true)) - acc) < 1e-6

        macro_f1 = f1_score(y_true, y_pred, average='macro')
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')

        prec_per_class = precision_score(y_true, y_pred, average=None)
        rec_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)

        assert abs(np.mean(f1_per_class) - macro_f1) < 1e-6

        return {
            'accuracy': float(acc),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'precision': [float(p) for p in prec_per_class],
            'recall': [float(r) for r in rec_per_class],
            'f1': [float(f) for f in f1_per_class],
            'confusion_matrix': cm.tolist()
        }

    splits_data = [('Validation', X_val, y_val), ('PublicTest', X_public, y_public), ('PrivateTest', X_private, y_private)]
    metrics = {'baseline': {}, 'hybrid': {}}

    for split_name, X_s, y_s in splits_data:
        metrics['baseline'][split_name] = evaluate_model(baseline_model, "Baseline", X_s, y_s)
        metrics['hybrid'][split_name] = evaluate_model(reloaded_model, "Hybrid", X_s, y_s)

    # Print Evaluation Summary
    for split_name, _, _ in splits_data:
        b = metrics['baseline'][split_name]
        h = metrics['hybrid'][split_name]
        print(f"\n--- {split_name.upper()} RESULTS ---")
        print(f"Baseline Acc: {b['accuracy']*100:.2f}% | Hybrid Acc: {h['accuracy']*100:.2f}% | Delta: {(h['accuracy']-b['accuracy'])*100:+.2f}%")
        print(f"Baseline F1:  {b['macro_f1']*100:.2f}% | Hybrid F1:  {h['macro_f1']*100:.2f}% | Delta: {(h['macro_f1']-b['macro_f1'])*100:+.2f}%")
        print("Per-Class F1 (Angry, Fear, Happy, Sad, Neutral):")
        for idx, cls in enumerate(EMOTION_CLASSES):
            d = h['f1'][idx] - b['f1'][idx]
            print(f"  {cls:8s}: Baseline={b['f1'][idx]:.4f}, Hybrid={h['f1'][idx]:.4f}, Delta={d:+.4f}")

    # 7. CPU Latency Benchmark
    print(f"\n[7/7] CPU LATENCY BENCHMARK (BATCH-1, DIRECT PREDICTOR PATH)")
    # Force CPU execution for benchmark
    with tf.device('/CPU:0'):
        dummy_tensor = np.zeros((1, 48, 48, 1), dtype=np.float32)

        def benchmark_latency(model):
            # Warmup (100 runs)
            for _ in range(100):
                _ = model(dummy_tensor, training=False).numpy()

            # Timed runs (1000 runs)
            times = []
            for _ in range(1000):
                t0 = time.perf_counter()
                _ = model(dummy_tensor, training=False).numpy()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)  # ms

            mean_lat = float(np.mean(times))
            p95_lat = float(np.percentile(times, 95))
            return mean_lat, p95_lat

        baseline_cpu_mean, baseline_cpu_p95 = benchmark_latency(baseline_model)
        hybrid_cpu_mean, hybrid_cpu_p95 = benchmark_latency(reloaded_model)

    print(f"Baseline CPU Mean: {baseline_cpu_mean:.2f} ms | P95: {baseline_cpu_p95:.2f} ms")
    print(f"Hybrid   CPU Mean: {hybrid_cpu_mean:.2f} ms | P95: {hybrid_cpu_p95:.2f} ms")

    # 8. Promotion Gate Check
    print(f"\n" + "=" * 80)
    print("PROMOTION GATES AUDIT")
    print("=" * 80)

    b_priv = metrics['baseline']['PrivateTest']
    h_priv = metrics['hybrid']['PrivateTest']

    acc_delta = h_priv['accuracy'] - b_priv['accuracy']
    f1_delta = h_priv['macro_f1'] - b_priv['macro_f1']
    class_f1_deltas = [h_priv['f1'][i] - b_priv['f1'][i] for i in range(5)]
    min_class_f1_delta = min(class_f1_deltas)

    gate1_pass = acc_delta >= 0.05
    gate2_pass = f1_delta >= 0.05
    gate3_pass = min_class_f1_delta >= -0.02
    gate4_pass = (hybrid_cpu_mean < 15.0) and (artifact_size_mb < 20.0)

    print(f"Gate 1 (PrivateTest Accuracy Delta >= +0.05): {acc_delta:+.4f} -> {'PASS' if gate1_pass else 'FAIL'}")
    print(f"Gate 2 (PrivateTest Macro F1 Delta >= +0.05):  {f1_delta:+.4f} -> {'PASS' if gate2_pass else 'FAIL'}")
    print(f"Gate 3 (No Class F1 Delta < -0.02):          {min_class_f1_delta:+.4f} -> {'PASS' if gate3_pass else 'FAIL'}")
    print(f"Gate 4 (CPU Mean < 15.0ms & Size < 20MB):     {hybrid_cpu_mean:.2f}ms, {artifact_size_mb:.2f}MB -> {'PASS' if gate4_pass else 'FAIL'}")

    # Output detailed JSON report for parsing
    import json
    report = {
        'tf_version': tf.__version__,
        'gpus': [g.name for g in gpus],
        'train_duration_sec': train_duration,
        'baseline': {
            'parameters': sum([tf.keras.backend.count_params(w) for w in baseline_model.weights]),
            'artifact_size_mb': baseline_size_mb,
            'cpu_mean_ms': baseline_cpu_mean,
            'cpu_p95_ms': baseline_cpu_p95,
            'metrics': metrics['baseline']
        },
        'hybrid': {
            'parameters': total_params,
            'artifact_size_mb': artifact_size_mb,
            'cpu_mean_ms': hybrid_cpu_mean,
            'cpu_p95_ms': hybrid_cpu_p95,
            'metrics': metrics['hybrid']
        },
        'gates': {
            'gate1_acc_delta': float(acc_delta),
            'gate1_pass': bool(gate1_pass),
            'gate2_f1_delta': float(f1_delta),
            'gate2_pass': bool(gate2_pass),
            'gate3_min_class_f1_delta': float(min_class_f1_delta),
            'gate3_pass': bool(gate3_pass),
            'gate4_cpu_mean_ms': float(hybrid_cpu_mean),
            'gate4_size_mb': float(artifact_size_mb),
            'gate4_pass': bool(gate4_pass)
        }
    }

    report_path = "scripts/phase58_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed report written to: {report_path}")

if __name__ == '__main__':
    run_experiment()
