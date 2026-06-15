#!/bin/bash
set -ex

export HF_HOME=/mnt/data/hf_cache
export HF_DATASETS_CACHE=/mnt/data/hf_cache
export TRANSFORMERS_CACHE=/mnt/data/hf_cache
export TRITON_CACHE_DIR=/mnt/data/hf_cache/triton
mkdir -p /mnt/data/hf_cache
mkdir -p /mnt/data/hf_cache/triton
mkdir -p /home/ubuntu/.triton/autotune

ENV_NAME="alex_training"
CONFIG=$1
GPU_LIST=$2
PORT=$3

if [ -z "$CONFIG" ]; then
  echo "Usage: ./train.sh <yaml_path> [gpu_list] [master_port]"
  exit 1
fi

if [ -z "$GPU_LIST" ]; then
  GPU_LIST=0
fi

if [ -z "$PORT" ]; then
  PORT=29500
fi

export CUDA_VISIBLE_DEVICES=$GPU_LIST
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$PORT
export FORCE_TORCHRUN=1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate || true
conda activate "$ENV_NAME"

cd LLaMA-Factory
mkdir -p logs

llamafactory-cli train "$CONFIG" \
  2>&1 | tee "logs/$(basename "$CONFIG" .yaml)_gpus${GPU_LIST//,/}.log"
