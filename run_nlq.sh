#!/bin/bash

config="$1"
export PYTHONPATH="."
gpu_id=0

# ---- Make sure conda-provided CUDA/cuDNN are used (TF 2.11 expects CUDA 11.x + cuDNN 8.x) ----
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"

cmd=(python code/model/nlq/trainer.py --config_yaml "$config")
echo "Executing: CUDA_VISIBLE_DEVICES=$gpu_id ${cmd[*]}"

CUDA_VISIBLE_DEVICES=$gpu_id "${cmd[@]}"
