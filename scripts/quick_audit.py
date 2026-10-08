import os
import sys
import time
import platform
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    models_info = [
        ("Baseline", "models/fer2013_cnn.h5"),
        ("Challenger", "models/fer2013_challenger_cnn.h5"),
        ("Low-Latency", "models/fer2013_low_latency_cnn.h5"),
        ("Hybrid", "models/fer2013_hybrid_cnn.h5"),
    ]

    from app.ml.model import load_model

    loaded_models = {}
    arch_info = {}

    for name, path in models_info:
        model = load_model(path)
        loaded_models[name] = model

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

        arch_info[name] = {
            'file_size_mb': os.path.getsize(path) / (1024 * 1024),
            'total_params': total_params,
            'trainable_params': trainable_params,
            'total_layers': total_layers,
            'conv_ops': conv_ops,
            'layer_breakdown': layer_types
        }

    dummy_tensor = np.zeros((1, 48, 48, 1), dtype=np.float32)
    benchmark_results = {}

    with tf.device('/CPU:0'):
        for name, _ in models_info:
            model = loaded_models[name]

            # 50 Warmup runs
            for _ in range(50):
                _ = model(dummy_tensor, training=False).numpy()

            # 200 Timed runs
            latencies = []
            for _ in range(200):
                t0 = time.perf_counter()
                _ = model(dummy_tensor, training=False).numpy()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)

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
                'arch': arch_info[name]
            }

    try:
        import cpuinfo
        cpu_name = cpuinfo.get_cpu_info()['brand_raw']
    except Exception:
        cpu_name = platform.processor() or "x86_64 Compatible CPU"

    output = {
        'tf_version': tf.__version__,
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'cpu_name': cpu_name,
        'benchmark_results': benchmark_results
    }

    with open("scripts/latency_audit_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("AUDIT_COMPLETE")

if __name__ == '__main__':
    main()
