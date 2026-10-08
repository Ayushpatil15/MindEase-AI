#!/bin/bash
set -e
source /tmp/mindease_venv/bin/activate
export LD_LIBRARY_PATH=$(echo /tmp/mindease_venv/lib/python3.12/site-packages/nvidia/*/lib | tr ' ' ':'):$LD_LIBRARY_PATH
cd "/mnt/e/MindEase AI"
export PYTHONPATH="/mnt/e/MindEase AI:$PYTHONPATH"
python3 scripts/phase58_experiment.py
