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

    from app.ml.model import load_model

    models_info = [
        ("Baseline", "models/fer2013_cnn.h5"),
        ("Challenger", "models/fer2013_challenger_cnn.h5"),
        ("Low-Latency", "models/fer2013_low_latency_cnn.h5"),
        ("Hybrid", "models/fer2013_hybrid_cnn.h5"),
    ]

    dummy_tensor = np.zeros((1, 48, 48, 1), dtype=np.float32)
    results = {}

    with tf.device('/CPU:0'):
        for name, path in models_info:
            model = load_model(path)
            
            # Count layers
            total_layers = len(model.layers)
            total_params = sum([tf.keras.backend.count_params(w) for w in model.weights])
            layer_types = {}
            for l in model.layers:
                t = l.__class__.__name__
                layer_types[t] = layer_types.get(t, 0) + 1
                
            conv_ops = (
                layer_types.get('Conv2D', 0) +
                layer_types.get('DepthwiseConv2D', 0) +
                layer_types.get('SeparableConv2D', 0)
            )

            # --- Harness A: direct __call__ model(x, training=False).numpy() ---
            for _ in range(50):
                _ = model(dummy_tensor, training=False).numpy()

            call_times = []
            for _ in range(200):
                t0 = time.perf_counter()
                _ = model(dummy_tensor, training=False).numpy()
                t1 = time.perf_counter()
                call_times.append((t1 - t0) * 1000.0)

            # --- Harness B: compiled / fast predict path model.predict(x, verbose=0) ---
            for _ in range(10):
                _ = model.predict(dummy_tensor, verbose=0)

            pred_times = []
            for _ in range(50):
                t0 = time.perf_counter()
                _ = model.predict(dummy_tensor, verbose=0)
                t1 = time.perf_counter()
                pred_times.append((t1 - t0) * 1000.0)

            # --- Harness C: tf.function compiled path ---
            @tf.function(reduce_retracing=True)
            def fast_infer(x):
                return model(x, training=False)

            for _ in range(50):
                _ = fast_infer(dummy_tensor).numpy()

            tffunc_times = []
            for _ in range(200):
                t0 = time.perf_counter()
                _ = fast_infer(dummy_tensor).numpy()
                t1 = time.perf_counter()
                tffunc_times.append((t1 - t0) * 1000.0)

            results[name] = {
                'params': total_params,
                'file_size_mb': os.path.getsize(path) / (1024 * 1024),
                'total_layers': total_layers,
                'conv_ops': conv_ops,
                'layer_breakdown': layer_types,
                'direct_call_mean_ms': float(np.mean(call_times)),
                'direct_call_p95_ms': float(np.percentile(call_times, 95)),
                'predict_method_mean_ms': float(np.mean(pred_times)),
                'predict_method_p95_ms': float(np.percentile(pred_times, 95)),
                'tffunc_compiled_mean_ms': float(np.mean(tffunc_times)),
                'tffunc_compiled_p95_ms': float(np.percentile(tffunc_times, 95)),
            }

    try:
        import cpuinfo
        cpu_name = cpuinfo.get_cpu_info()['brand_raw']
    except Exception:
        cpu_name = platform.processor() or "x86_64 Compatible CPU"

    summary = {
        'tf_version': tf.__version__,
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'cpu_name': cpu_name,
        'one_dnn': os.environ.get('TF_ENABLE_ONEDNN_OPTS', '1 (default)'),
        'results': results
    }

    with open("scripts/latency_audit_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("COMPARISON_AUDIT_COMPLETE")

if __name__ == '__main__':
    main()
