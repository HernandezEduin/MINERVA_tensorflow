#!/bin/bash

config="$1"
export PYTHONPATH="."
gpu_id=1

cmd=(python code/model/nlq/trainer.py --config_yaml "$config")
echo "Executing: CUDA_VISIBLE_DEVICES=$gpu_id ${cmd[*]}"

CUDA_VISIBLE_DEVICES=$gpu_id "${cmd[@]}"
