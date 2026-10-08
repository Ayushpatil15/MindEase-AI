#!/bin/bash
export LD_LIBRARY_PATH=$(python3 -c "import os, nvidia; print(':'.join([os.path.join(os.path.dirname(nvidia.__file__), sub, 'lib') for sub in os.listdir(os.path.dirname(nvidia.__file__)) if os.path.isdir(os.path.join(os.path.dirname(nvidia.__file__), sub, 'lib'))]))"):$LD_LIBRARY_PATH
source /tmp/mindease_venv/bin/activate
python3 -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
