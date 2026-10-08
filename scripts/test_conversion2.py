import os
import sys
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.ml.model import load_model, build_hybrid_model, build_low_latency_model, build_challenger_model

def main():
    print("Testing clean model instantiation & weight copy conversion...")

    models_info = [
        ("Hybrid", "models/fer2013_hybrid_cnn.h5", build_hybrid_model),
        ("Low-Latency", "models/fer2013_low_latency_cnn.h5", build_low_latency_model),
        ("Challenger", "models/fer2013_challenger_cnn.h5", build_challenger_model),
    ]

    for name, path, build_fn in models_info:
        src = load_model(path)
        fresh = build_fn()
        fresh.set_weights(src.get_weights())
        print(f"Loaded and transferred weights for {name}")

        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(fresh)
            tflite_model = converter.convert()
            print(f"  [{name}] FP32 TFLite Conversion SUCCESS! Size: {len(tflite_model)/1024:.2f} KB")
        except Exception as e:
            print(f"  [{name}] FP32 TFLite Conversion FAILED: {e}")

if __name__ == '__main__':
    main()
