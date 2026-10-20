"""Phase 5.6 — Latency Benchmark (standalone).

Benchmarks both baseline and challenger models on CPU and GPU separately,
by loading fresh model instances per device to avoid cross-device errors.
"""

import os
import sys
import time
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)


def main():
    import tensorflow as tf

    tf.keras.utils.set_random_seed(42)

    print("=" * 70)
    print("PHASE 5.6 — LATENCY BENCHMARK")
    print("=" * 70)
    print(f"TensorFlow: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs: {gpus}")

    baseline_path = os.path.join(PROJECT_ROOT, 'models', 'fer2013_cnn.h5')
    challenger_path = os.path.join(PROJECT_ROOT, 'models', 'fer2013_challenger_cnn.h5')

    dummy_input = np.random.rand(1, 48, 48, 1).astype(np.float32)
    WARMUP = 10
    TIMED = 100

    def benchmark_on_device(model_path, model_name, device_name):
        """Load a fresh model on the specified device and benchmark."""
        with tf.device(device_name):
            model = tf.keras.models.load_model(model_path)
            # Warm-up
            for _ in range(WARMUP):
                model(dummy_input, training=False)

            # Timed runs
            times = []
            for _ in range(TIMED):
                t0 = time.perf_counter()
                model(dummy_input, training=False)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

        mean_ms = float(np.mean(times))
        p95_ms = float(np.percentile(times, 95))
        print(f"  {model_name:20s} [{device_name}]: mean={mean_ms:.2f}ms  p95={p95_ms:.2f}ms")
        return {'mean_ms': mean_ms, 'p95_ms': p95_ms}

    results = {}

    # CPU benchmarks
    print("\nCPU Latency (batch_size=1, 10 warmup, 100 timed):")
    results['baseline_cpu'] = benchmark_on_device(baseline_path, 'Baseline', '/CPU:0')
    results['challenger_cpu'] = benchmark_on_device(challenger_path, 'Challenger', '/CPU:0')

    # GPU benchmarks
    if gpus:
        print("\nGPU Latency (batch_size=1, 10 warmup, 100 timed):")
        results['baseline_gpu'] = benchmark_on_device(baseline_path, 'Baseline', '/GPU:0')
        results['challenger_gpu'] = benchmark_on_device(challenger_path, 'Challenger', '/GPU:0')
    else:
        print("\nGPU: Not available — skipping GPU benchmark.")

    # Summary
    print("\n" + "=" * 70)
    print("LATENCY SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<35} {'Baseline':>12} {'Challenger':>12}")
    print("-" * 60)
    print(f"{'CPU Mean (ms)':<35} {results['baseline_cpu']['mean_ms']:>12.2f} {results['challenger_cpu']['mean_ms']:>12.2f}")
    print(f"{'CPU P95 (ms)':<35} {results['baseline_cpu']['p95_ms']:>12.2f} {results['challenger_cpu']['p95_ms']:>12.2f}")
    if gpus:
        print(f"{'GPU Mean (ms)':<35} {results['baseline_gpu']['mean_ms']:>12.2f} {results['challenger_gpu']['mean_ms']:>12.2f}")
        print(f"{'GPU P95 (ms)':<35} {results['baseline_gpu']['p95_ms']:>12.2f} {results['challenger_gpu']['p95_ms']:>12.2f}")

    # Gate 4 check
    cpu_mean = results['challenger_cpu']['mean_ms']
    print(f"\nPromotion Gate 4 — CPU mean latency < 15ms: {cpu_mean:.2f}ms {'PASS' if cpu_mean < 15 else 'FAIL'}")

    print("\nLatency benchmark complete.")


if __name__ == '__main__':
    main()
