import os
import sys
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.ml.model import load_model

def main():
    model_path = "models/fer2013_hybrid_cnn.h5"
    model = load_model(model_path)
    print("Loaded model successfully:", model.name)

    # Method 1: Concrete function export
    @tf.function
    def serve(x):
        return model(x, training=False)

    concrete_func = serve.get_concrete_function(tf.TensorSpec([1, 48, 48, 1], tf.float32))
    
    try:
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        tflite_model = converter.convert()
        print("Concrete Function Conversion Success! Size:", len(tflite_model) / 1024, "KB")
    except Exception as e:
        print("Concrete Function Conversion Failed:", e)

    # Method 2: SavedModel export
    temp_dir = "models/deployment/temp_savedmodel"
    tf.saved_model.save(model, temp_dir, signatures={'serving_default': concrete_func})
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(temp_dir)
        tflite_model = converter.convert()
        print("SavedModel Conversion Success! Size:", len(tflite_model) / 1024, "KB")
    except Exception as e:
        print("SavedModel Conversion Failed:", e)

    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

if __name__ == '__main__':
    main()
