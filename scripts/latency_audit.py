"""Latency Benchmark Audit Script — Phase 5.8 Final Audit.

Re-runs the exact CPU single-sample latency benchmark for all 4 models in native Windows environment:
1. models/fer2013_cnn.h5 (Baseline)
2. models/fer2013_challenger_cnn.h5 (Challenger)
3. models/fer2013_low_latency_cnn.h5 (Low-Latency)
4. models/fer2013_hybrid_cnn.h5 (Hybrid)
"""

import os
import sys
import time
import platform
import numpy as np

# Ensure root directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def audit_latency():
    print("=" * 80)
    print("PHASE 5.8 — FINAL LATENCY BASELINE AUDIT")
    print("=" * 80)

    # 1. Environment Details
    import tensorflow as tf
    print("\n[1/4] BENCHMARK ENVIRONMENT")
    print(f"OS / Platform:         {platform.platform()}")
    print(f"Python Version:        {platform.python_version()}")
    print(f"TensorFlow Version:    {tf.__version__}")
    print(f"oneDNN Enabled:        {os.environ.get('TF_ENABLE_ONEDNN_OPTS', 'Default (1)')}")

    try:
        import cpuinfo
        cpu_name = cpuinfo.get_cpu_info()['brand_raw']
    except Exception:
        cpu_name = platform.processor() or "x86_64 Compatible CPU"
    print(f"CPU Hardware:          {cpu_name}")
    print(f"Logical CPU Count:     {os.cpu_count()}")

    intra_op = tf.config.threading.get_intra_op_parallelism_threads()
    inter_op = tf.config.threading.get_inter_op_parallelism_threads()
    print(f"Intra-op Threads:      {intra_op if intra_op != 0 else 'Auto (Default)'}")
    print(f"Inter-op Threads:      {inter_op if inter_op != 0 else 'Auto (Default)'}")

    # Force CPU execution for all benchmark calls
    tf.config.set_visible_devices([], 'GPU')

    # 2. Model File Verification & Architecture Derivation
    print("\n[2/4] MODEL ARTIFACT VERIFICATION & ARCHITECTURE DERIVATION")
    models_info = [
        ("Baseline", "models/fer2013_cnn.h5"),
        ("Challenger", "models/fer2013_challenger_cnn.h5"),
        ("Low-Latency", "models/fer2013_low_latency_cnn.h5"),
        ("Hybrid", "models/fer2013_hybrid_cnn.h5"),
    ]

    loaded_models = {}

    from app.ml.model import load_model

    for name, path in models_info:
        if not os.path.exists(path):
            print(f"ERROR: Model file missing at {path}")
            sys.exit(1)

        size_mb = os.path.getsize(path) / (1024 * 1024)
        model = load_model(path)
        if model is None:
            print(f"ERROR: Failed to load model from {path}")
            sys.exit(1)

        loaded_models[name] = model

        # Derivation of layer counts and operation counts
        total_layers = len(model.layers)
        total_params = sum([tf.keras.backend.count_params(w) for w in model.weights])
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])

        layer_types = {}
        for l in model.layers:
            t = l.__class__.__name__
            layer_types[t] = layer_types.get(t, 0) + 1

        conv_ops = (
            layer_types.get('Conv2D', 0) +
            layer_types.get('DepthwiseConv2D', 0) +
            layer_types.get('SeparableConv2D', 0)
        )

        print(f"\nModel: {name:12s} | File: {path}")
        print(f"  File Size:           {size_mb:.2f} MB")
        print(f"  Total Parameters:    {total_params:,} (Trainable: {trainable_params:,})")
        print(f"  Total Keras Layers:  {total_layers}")
        print(f"  Convolution Ops:     {conv_ops}")
        print(f"  Layer Breakdown:     {layer_types}")

    # 3. Synchronized CPU Latency Benchmark
    print("\n[3/4] SYNCHRONIZED CPU LATENCY BENCHMARK")
    print("Harness Specification:")
    print("  - Device: CPU")
    print("  - Batch Size: 1")
    print("  - Input Tensor Shape: (1, 48, 48, 1)")
    print("  - Warmup Iterations: 100 (discarded)")
    print("  - Timed Iterations: 1000")
    print("  - Timing Function: time.perf_counter()")
    print("  - Inference Method: model(x, training=False).numpy()")

    dummy_tensor = np.zeros((1, 48, 48, 1), dtype=np.float32)
    benchmark_results = {}

    with tf.device('/CPU:0'):
        for name, _, in models_info:
            model = loaded_models[name]

            # 100 Warmup Runs
            for _ in range(100):
                _ = model(dummy_tensor, training=False).numpy()

            # 1000 Timed Runs
            latencies = []
            for _ in range(1000):
                t0 = time.perf_counter()
                _ = model(dummy_tensor, training=False).numpy()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)  # ms

            mean_lat = float(np.mean(latencies))
            median_lat = float(np.median(latencies))
            min_lat = float(np.min(latencies))
            max_lat = float(np.max(latencies))
            p95_lat = float(np.percentile(latencies, 95))
            std_lat = float(np.std(latencies))

            benchmark_results[name] = {
                'mean_ms': mean_lat,
                'median_ms': median_lat,
                'p95_ms': p95_lat,
                'min_ms': min_lat,
                'max_ms': max_lat,
                'std_ms': std_lat,
                'file_size_mb': os.path.getsize(models_info[[m[0] for m in models_info].index(name)][1]) / (1024 * 1024),
                'total_params': sum([tf.keras.backend.count_params(w) for w in model.weights]),
                'total_layers': len(model.layers)
            }

    print("\n" + "=" * 80)
    print("FOUR-MODEL CPU LATENCY BENCHMARK TABLE")
    print("=" * 80)
    header = f"{'Model Name':15s} | {'Params':8s} | {'Size (MB)':9s} | {'Layers':6s} | {'Mean (ms)':9s} | {'P95 (ms)':9s} | {'Min (ms)':8s} | {'Max (ms)':8s}"
    print(header)
    print("-" * len(header))

    for name in ["Baseline", "Challenger", "Low-Latency", "Hybrid"]:
        r = benchmark_results[name]
        print(f"{name:15s} | {r['total_params']:8d} | {r['file_size_mb']:9.2f} | {r['total_layers']:6d} | {r['mean_ms']:9.2f} | {r['p95_ms']:9.2f} | {r['min_ms']:8.2f} | {r['max_ms']:8.2f}")

    # 4. Investigation of Baseline Latency (18.16 ms vs ~10.4 ms)
    print("\n[4/4] INVESTIGATION OF BASELINE LATENCY DISCREPANCY")
    # Test alternative threading settings (e.g. single-threaded vs multi-threaded) to analyze impact
    latencies_single_thread = []
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    with tf.device('/CPU:0'):
        baseline_m = loaded_models["Baseline"]
        for _ in range(50):
            _ = baseline_m(dummy_tensor, training=False).numpy()
        for _ in range(200):
            t0 = time.perf_counter()
            _ = baseline_m(dummy_tensor, training=False).numpy()
            t1 = time.perf_counter()
            latencies_single_thread.append((t1 - t0) * 1000.0)

    st_mean = float(np.mean(latencies_single_thread))
    st_p95 = float(np.percentile(latencies_single_thread, 95))
    print(f"Baseline Single-Threaded CPU Latency: Mean = {st_mean:.2f} ms | P95 = {st_p95:.2f} ms")

    # Output JSON summary for exact audit reporting
    import json
    audit_summary = {
        'tf_version': tf.__version__,
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'cpu_name': cpu_name,
        'benchmark_results': benchmark_results,
        'baseline_single_thread_mean_ms': st_mean,
        'baseline_single_thread_p95_ms': st_p95
    }
    with open("scripts/latency_audit_results.json", "w") as f:
        json.dump(audit_summary, f, indent=2)

    print("\nAudit results saved to: scripts/latency_audit_results.json")

if __name__ == '__main__':
    audit_latency()
