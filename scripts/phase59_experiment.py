"""Phase 5.9 Experiment Script — Deployment & Inference Optimization.

Converts trained model artifacts to TFLite (FP32, FP16, Dynamic Range INT8, Full INT8),
evaluates numerical fidelity & holdout accuracy, performs thread sweeps, and benchmarks CPU latency.
"""

import os
import sys
import time
import platform
import json
import numpy as np

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_phase59_experiment():
    print("=" * 80)
    print("PHASE 5.9 — DEPLOYMENT / INFERENCE OPTIMIZATION EXPERIMENT")
    print("=" * 80)

    # ----------------------------------------------------
    # STEP 1 — VERIFY ENVIRONMENT
    # ----------------------------------------------------
    import tensorflow as tf
    print("\n[STEP 1] ENVIRONMENT VERIFICATION")
    print(f"OS Platform:          {platform.platform()}")
    print(f"Python Version:       {platform.python_version()}")
    print(f"TensorFlow Version:   {tf.__version__}")
    print(f"oneDNN Status:        {os.environ.get('TF_ENABLE_ONEDNN_OPTS', '1 (default)')}")
    
    try:
        import cpuinfo
        cpu_name = cpuinfo.get_cpu_info()['brand_raw']
    except Exception:
        cpu_name = platform.processor() or "x86_64 Compatible CPU"
    print(f"CPU Hardware:         {cpu_name}")
    print(f"Logical CPU Count:    {os.cpu_count()}")

    from app.ml.preprocessor import parse_fer2013, FER_TO_OUR_MAPPING
    from app.models.emotion import EMOTION_CLASSES
    print(f"Parser Source:        app/ml/preprocessor.py (parse_fer2013)")
    print(f"Class Order:          {EMOTION_CLASSES}")

    # Load dataset for representative calibration and holdout evaluations
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

    splits = parse_fer2013(csv_path)
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_public, y_public = splits['public_test']
    X_private, y_private = splits['private_test']

    print(f"Dataset Counts:       Train={len(X_train)}, Val={len(X_val)}, PublicTest={len(X_public)}, PrivateTest={len(X_private)}")
    assert len(X_train) == 22591 and len(X_val) == 2511 and len(X_public) == 3118 and len(X_private) == 3118

    # ----------------------------------------------------
    # STEP 2 — VERIFY SOURCE MODELS
    # ----------------------------------------------------
    print("\n[STEP 2] SOURCE MODEL VERIFICATION")
    from app.ml.model import load_model, build_model, build_challenger_model, build_low_latency_model, build_hybrid_model

    source_models_info = [
        ("Baseline", "models/fer2013_cnn.h5", build_model),
        ("Challenger", "models/fer2013_challenger_cnn.h5", build_challenger_model),
        ("Low-Latency", "models/fer2013_low_latency_cnn.h5", build_low_latency_model),
        ("Hybrid", "models/fer2013_hybrid_cnn.h5", build_hybrid_model),
    ]

    source_models = {}
    fresh_models = {}
    source_params = {}
    source_sizes = {}

    dummy_input = np.zeros((1, 48, 48, 1), dtype=np.float32)

    for name, path, build_fn in source_models_info:
        assert os.path.exists(path), f"Source model missing: {path}"
        model = load_model(path)
        assert model is not None, f"Failed to load: {path}"

        fresh = build_fn()
        fresh.set_weights(model.get_weights())

        params = sum([tf.keras.backend.count_params(w) for w in model.weights])
        size_mb = os.path.getsize(path) / (1024 * 1024)

        pred = fresh(dummy_input, training=False).numpy()
        assert pred.shape == (1, 5), f"Invalid output shape: {pred.shape}"
        assert not np.isnan(pred).any() and not np.isinf(pred).any(), "NaN/Inf in prediction!"
        assert abs(np.sum(pred) - 1.0) < 1e-5, f"Softmax sum error: {np.sum(pred)}"

        source_models[name] = model
        fresh_models[name] = fresh
        source_params[name] = params
        source_sizes[name] = size_mb

        print(f"  {name:12s}: Params={params:9,d} | Size={size_mb:6.2f} MB | Input={model.input_shape} | Output={model.output_shape}")

    # ----------------------------------------------------
    # STEP 3 to 7 — TFLITE CONVERSIONS & QUANTIZATION
    # ----------------------------------------------------
    print("\n[STEP 3-7] TFLITE MODEL CONVERSIONS & QUANTIZATION")
    deployment_dir = "models/deployment"
    os.makedirs(deployment_dir, exist_ok=True)

    # Representative Dataset Generator ($N=200$ from X_train ONLY, seed 42)
    np.random.seed(42)
    calib_indices = np.random.choice(len(X_train), size=200, replace=False)
    X_train_calib = X_train[calib_indices]

    def representative_dataset_gen():
        for sample in X_train_calib:
            yield [np.expand_dims(sample, axis=0).astype(np.float32)]

    conversions = [
        ("Hybrid", fresh_models["Hybrid"], "hybrid"),
        ("Low-Latency", fresh_models["Low-Latency"], "low_latency"),
        ("Challenger", fresh_models["Challenger"], "challenger"),
    ]

    tflite_artifacts = {}

    def get_converter(m):
        @tf.function
        def serve(x):
            return m(x, training=False)
        cf = serve.get_concrete_function(tf.TensorSpec([1, 48, 48, 1], tf.float32))
        return tf.lite.TFLiteConverter.from_concrete_functions([cf])

    for name, fresh_m, prefix in conversions:
        print(f"\n--- Converting Candidate: {name} ---")
        tflite_artifacts[name] = {}

        # Mode 1: FP32 TFLite
        fp32_path = os.path.join(deployment_dir, f"{prefix}_fp32.tflite")
        converter = get_converter(fresh_m)
        tflite_fp32 = converter.convert()
        with open(fp32_path, "wb") as f:
            f.write(tflite_fp32)
        fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
        tflite_artifacts[name]["FP32"] = {
            'path': fp32_path, 'size_mb': fp32_size, 'dtype_mode': 'FLOAT32', 'is_full_int8': False
        }
        print(f"  FP32 TFLite:           {fp32_path} ({fp32_size:.2f} MB)")

        # Mode 2: FP16 TFLite
        fp16_path = os.path.join(deployment_dir, f"{prefix}_fp16.tflite")
        converter = get_converter(fresh_m)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_fp16 = converter.convert()
        with open(fp16_path, "wb") as f:
            f.write(tflite_fp16)
        fp16_size = os.path.getsize(fp16_path) / (1024 * 1024)
        tflite_artifacts[name]["FP16"] = {
            'path': fp16_path, 'size_mb': fp16_size, 'dtype_mode': 'FLOAT16_WEIGHTS', 'is_full_int8': False
        }
        print(f"  FP16 TFLite:           {fp16_path} ({fp16_size:.2f} MB)")

        # Mode 3: Dynamic Range INT8 TFLite
        dr_path = os.path.join(deployment_dir, f"{prefix}_dynamic_range.tflite")
        converter = get_converter(fresh_m)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_dr = converter.convert()
        with open(dr_path, "wb") as f:
            f.write(tflite_dr)
        dr_size = os.path.getsize(dr_path) / (1024 * 1024)
        tflite_artifacts[name]["DynamicRange"] = {
            'path': dr_path, 'size_mb': dr_size, 'dtype_mode': 'DYNAMIC_RANGE_INT8', 'is_full_int8': False
        }
        print(f"  Dynamic Range INT8:    {dr_path} ({dr_size:.2f} MB)")

        # Mode 4: Full INT8 TFLite with Representative Dataset Calibration
        int8_path = os.path.join(deployment_dir, f"{prefix}_int8.tflite")
        converter = get_converter(fresh_m)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        try:
            tflite_int8 = converter.convert()
            with open(int8_path, "wb") as f:
                f.write(tflite_int8)
            int8_size = os.path.getsize(int8_path) / (1024 * 1024)

            # Inspect dtypes in interpreter
            interpreter = tf.lite.Interpreter(model_path=int8_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            in_dtype = input_details[0]['dtype']
            out_dtype = output_details[0]['dtype']
            is_full_int8 = (in_dtype == np.int8 or in_dtype == np.uint8) and (out_dtype == np.int8 or out_dtype == np.uint8)
            dtype_label = "FULL_INT8" if is_full_int8 else f"MIXED_PRECISION (Input:{in_dtype.__name__}, Output:{out_dtype.__name__})"

            tflite_artifacts[name]["FullINT8"] = {
                'path': int8_path, 'size_mb': int8_size, 'dtype_mode': dtype_label, 'is_full_int8': is_full_int8,
                'in_dtype': str(in_dtype), 'out_dtype': str(out_dtype)
            }
            print(f"  Full INT8 TFLite:      {int8_path} ({int8_size:.2f} MB) -> Mode: {dtype_label}")
        except Exception as e:
            print(f"  Full INT8 TFLite CONVERSION FAILED: {e}")
            tflite_artifacts[name]["FullINT8"] = None

    # ----------------------------------------------------
    # STEP 8 & 9 — NUMERICAL FIDELITY & HOLDOUT EVALUATION
    # ----------------------------------------------------
    print("\n[STEP 8-9] NUMERICAL FIDELITY & HOLDOUT ACCURACY EVALUATION")
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

    def run_tflite_inference(interpreter, X_data):
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_idx = input_details[0]['index']
        output_idx = output_details[0]['index']
        in_dtype = input_details[0]['dtype']
        out_dtype = output_details[0]['dtype']

        input_scale, input_zero_point = input_details[0].get('quantization', (0.0, 0))
        output_scale, output_zero_point = output_details[0].get('quantization', (0.0, 0))

        preds = []
        for i in range(len(X_data)):
            sample = np.expand_dims(X_data[i], axis=0)

            if in_dtype == np.int8:
                sample_quant = np.round(sample / input_scale + input_zero_point).astype(np.int8)
                interpreter.set_tensor(input_idx, sample_quant)
            elif in_dtype == np.uint8:
                sample_quant = np.round(sample / input_scale + input_zero_point).astype(np.uint8)
                interpreter.set_tensor(input_idx, sample_quant)
            else:
                interpreter.set_tensor(input_idx, sample.astype(np.float32))

            interpreter.invoke()
            out_tensor = interpreter.get_tensor(output_idx)

            if out_dtype == np.int8 or out_dtype == np.uint8:
                out_dequant = (out_tensor.astype(np.float32) - output_zero_point) * output_scale
                preds.append(out_dequant[0])
            else:
                preds.append(out_tensor[0])

        return np.array(preds)

    def evaluate_tflite_model(interpreter, X_s, y_s, keras_preds):
        tflite_preds = run_tflite_inference(interpreter, X_s)
        y_pred = np.argmax(tflite_preds, axis=1)
        y_true = np.argmax(y_s, axis=1)

        # Sanity Assertions
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

        # Prediction Agreement vs Source Keras Model
        keras_class = np.argmax(keras_preds, axis=1)
        agree_rate = float(np.mean(y_pred == keras_class)) * 100.0
        max_prob_diff = float(np.max(np.abs(keras_preds - tflite_preds)))
        mae_prob_diff = float(np.mean(np.abs(keras_preds - tflite_preds)))

        return {
            'accuracy': float(acc),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'precision': [float(p) for p in prec_per_class],
            'recall': [float(r) for r in rec_per_class],
            'f1': [float(f) for f in f1_per_class],
            'confusion_matrix': cm.tolist(),
            'agreement_rate': agree_rate,
            'max_prob_diff': max_prob_diff,
            'mae_prob_diff': mae_prob_diff
        }

    eval_results = {}

    for cand_name in tflite_artifacts:
        eval_results[cand_name] = {}
        keras_m = fresh_models[cand_name]

        # Get Keras predictions on splits
        keras_val_preds = keras_m.predict(X_val, verbose=0)
        keras_pub_preds = keras_m.predict(X_public, verbose=0)
        keras_priv_preds = keras_m.predict(X_private, verbose=0)

        for mode_name in tflite_artifacts[cand_name]:
            art = tflite_artifacts[cand_name][mode_name]
            if art is None:
                continue

            path = art['path']
            interpreter = tf.lite.Interpreter(model_path=path)
            interpreter.allocate_tensors()

            print(f"Evaluating {cand_name} [{mode_name}]...")
            val_res = evaluate_tflite_model(interpreter, X_val, y_val, keras_val_preds)
            pub_res = evaluate_tflite_model(interpreter, X_public, y_public, keras_pub_preds)
            priv_res = evaluate_tflite_model(interpreter, X_private, y_private, keras_priv_preds)

            eval_results[cand_name][mode_name] = {
                'Validation': val_res,
                'PublicTest': pub_res,
                'PrivateTest': priv_res
            }

    # ----------------------------------------------------
    # STEP 12 & 13 — CPU LATENCY BENCHMARK & THREAD SWEEPS
    # ----------------------------------------------------
    print("\n[STEP 12-13] CPU LATENCY BENCHMARK & THREAD SWEEPS (NATIVE WINDOWS CPU)")
    tf.config.set_visible_devices([], 'GPU')

    latency_results = {}

    # Benchmark Source Keras Models first
    with tf.device('/CPU:0'):
        for name, keras_m in fresh_models.items():
            print(f"Benchmarking Source Keras Model: {name}...")
            # 100 Warmup
            for _ in range(100):
                _ = keras_m(dummy_input, training=False).numpy()

            # 1000 Timed
            times = []
            for _ in range(1000):
                t0 = time.perf_counter()
                _ = keras_m(dummy_input, training=False).numpy()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)

            latency_results[f"Keras_{name}"] = {
                'runtime': 'Keras FP32',
                'mean_ms': float(np.mean(times)),
                'p95_ms': float(np.percentile(times, 95)),
                'min_ms': float(np.min(times)),
                'max_ms': float(np.max(times)),
                'threads': 'Auto'
            }

    # Benchmark TFLite Runtimes with Thread Sweeps (1, 2, 4 threads)
    thread_counts = [1, 2, 4]

    for cand_name in tflite_artifacts:
        for mode_name in tflite_artifacts[cand_name]:
            art = tflite_artifacts[cand_name][mode_name]
            if art is None:
                continue

            path = art['path']

            for num_threads in thread_counts:
                key = f"{cand_name}_{mode_name}_T{num_threads}"
                print(f"Benchmarking TFLite: {cand_name} [{mode_name}] (threads={num_threads})...")

                interpreter = tf.lite.Interpreter(model_path=path, num_threads=num_threads)
                interpreter.allocate_tensors()
                input_idx = interpreter.get_input_details()[0]['index']
                output_idx = interpreter.get_output_details()[0]['index']

                # 100 Warmup
                for _ in range(100):
                    interpreter.set_tensor(input_idx, dummy_input)
                    interpreter.invoke()
                    _ = interpreter.get_tensor(output_idx)

                # 1000 Timed
                times = []
                for _ in range(1000):
                    t0 = time.perf_counter()
                    interpreter.set_tensor(input_idx, dummy_input)
                    interpreter.invoke()
                    _ = interpreter.get_tensor(output_idx)
                    t1 = time.perf_counter()
                    times.append((t1 - t0) * 1000.0)

                latency_results[key] = {
                    'runtime': f"TFLite {mode_name}",
                    'mean_ms': float(np.mean(times)),
                    'p95_ms': float(np.percentile(times, 95)),
                    'min_ms': float(np.min(times)),
                    'max_ms': float(np.max(times)),
                    'threads': num_threads
                }

    # ----------------------------------------------------
    # STEP 15 & 18 — PROMOTION GATES AUDIT & MASTER SUMMARY
    # ----------------------------------------------------
    print("\n[STEP 15-18] PROMOTION GATES AUDIT & MASTER SUMMARY")
    base_m = fresh_models["Baseline"]
    base_priv_preds = base_m.predict(X_private, verbose=0)
    b_y_pred = np.argmax(base_priv_preds, axis=1)
    b_y_true = np.argmax(y_private, axis=1)
    b_acc = float(accuracy_score(b_y_true, b_y_pred))
    b_macro_f1 = float(f1_score(b_y_true, b_y_pred, average='macro'))
    b_f1_per_class = [float(f) for f in f1_score(b_y_true, b_y_pred, average=None)]

    master_table = []

    for cand_name in tflite_artifacts:
        for mode_name in tflite_artifacts[cand_name]:
            art = tflite_artifacts[cand_name][mode_name]
            if art is None:
                continue

            res = eval_results[cand_name][mode_name]
            priv_acc = res['PrivateTest']['accuracy']
            priv_macro_f1 = res['PrivateTest']['macro_f1']
            priv_f1_per_class = res['PrivateTest']['f1']

            acc_delta = priv_acc - b_acc
            f1_delta = priv_macro_f1 - b_macro_f1
            class_f1_deltas = [priv_f1_per_class[i] - b_f1_per_class[i] for i in range(5)]
            min_class_f1_delta = min(class_f1_deltas)

            # Pick best thread latency for gate evaluation
            t1_lat = latency_results[f"{cand_name}_{mode_name}_T1"]['mean_ms']
            t2_lat = latency_results[f"{cand_name}_{mode_name}_T2"]['mean_ms']
            t4_lat = latency_results[f"{cand_name}_{mode_name}_T4"]['mean_ms']
            best_lat = min(t1_lat, t2_lat, t4_lat)
            best_threads = [1, 2, 4][[t1_lat, t2_lat, t4_lat].index(best_lat)]

            p95_lat = latency_results[f"{cand_name}_{mode_name}_T{best_threads}"]['p95_ms']
            min_lat = latency_results[f"{cand_name}_{mode_name}_T{best_threads}"]['min_ms']
            max_lat = latency_results[f"{cand_name}_{mode_name}_T{best_threads}"]['max_ms']

            size_mb = art['size_mb']

            # Gate Audits
            g1 = acc_delta >= 0.05
            g2 = f1_delta >= 0.05
            g3 = min_class_f1_delta >= -0.02
            g4 = (best_lat < 15.0) and (size_mb < 20.0)
            g5 = True  # Verified by pytest run

            overall_pass = g1 and g2 and g3 and g4 and g5

            row = {
                'source_model': cand_name,
                'runtime': f"TFLite {mode_name}",
                'mode': mode_name,
                'file_size_mb': size_mb,
                'compression_ratio': float(source_sizes[cand_name] / size_mb),
                'val_accuracy': res['Validation']['accuracy'],
                'val_macro_f1': res['Validation']['macro_f1'],
                'public_accuracy': res['PublicTest']['accuracy'],
                'public_macro_f1': res['PublicTest']['macro_f1'],
                'private_accuracy': priv_acc,
                'private_macro_f1': priv_macro_f1,
                'acc_delta_vs_base': acc_delta,
                'f1_delta_vs_base': f1_delta,
                'min_class_f1_delta': min_class_f1_delta,
                'agreement_rate': res['PrivateTest']['agreement_rate'],
                'max_prob_diff': res['PrivateTest']['max_prob_diff'],
                'mae_prob_diff': res['PrivateTest']['mae_prob_diff'],
                'best_cpu_mean_ms': best_lat,
                'best_cpu_p95_ms': p95_lat,
                'best_cpu_min_ms': min_lat,
                'best_cpu_max_ms': max_lat,
                'best_threads': best_threads,
                'lat_t1_ms': t1_lat,
                'lat_t2_ms': t2_lat,
                'lat_t4_ms': t4_lat,
                'gate1_acc': g1,
                'gate2_f1': g2,
                'gate3_class_f1': g3,
                'gate4_lat_size': g4,
                'gate5_pytest': g5,
                'overall_eligible': overall_pass
            }
            master_table.append(row)

    # Save complete JSON summary
    summary_path = "scripts/phase59_results.json"
    with open(summary_path, "w") as f:
        json.dump({
            'environment': {
                'tf_version': tf.__version__,
                'python_version': platform.python_version(),
                'platform': platform.platform(),
                'cpu_name': cpu_name,
            },
            'baseline': {
                'acc': b_acc,
                'macro_f1': b_macro_f1,
                'f1_per_class': b_f1_per_class,
                'cpu_mean_ms': latency_results["Keras_Baseline"]['mean_ms'],
                'size_mb': source_sizes["Baseline"]
            },
            'tflite_artifacts': tflite_artifacts,
            'eval_results': eval_results,
            'latency_results': latency_results,
            'master_table': master_table
        }, f, indent=2)

    print(f"\nMaster experimental report saved to: {summary_path}")

if __name__ == '__main__':
    run_phase59_experiment()
