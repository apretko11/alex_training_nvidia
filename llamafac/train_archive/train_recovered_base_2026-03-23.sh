#!/bin/bash
set -ex

export HF_HOME=/mnt/data/hf_cache
export HF_DATASETS_CACHE=/mnt/data/hf_cache
export TRANSFORMERS_CACHE=/mnt/data/hf_cache
mkdir -p /mnt/data/hf_cache
# HUGGINGFACE_HUB_TOKEN is the name of the variable here

ENV_NAME="alex_training"
CONFIG=$1
GPU_ID=$2
PORT=$3

if [ -z "$CONFIG" ]; then
  echo "Usage: ./train.sh <yaml_path> <gpu_id> <master_port>"
  exit 1
fi

if [ -z "$GPU_ID" ]; then
  GPU_ID=0
fi

if [ -z "$PORT" ]; then
  PORT=29500
fi

export NNODES=1
export NPROC_PER_NODE=1
export CUDA_VISIBLE_DEVICES=$GPU_ID
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$PORT

source $(conda info --base)/etc/profile.d/conda.sh
conda deactivate
conda activate $ENV_NAME

cd LLaMA-Factory

export PYTHONPATH="/mnt/data/alex/llamafac_recovered_base/LLaMA-Factory/src:$PYTHONPATH"

FORCE_TORCHRUN=1 python -m llamafactory.cli train $CONFIG 2>&1 | tee logs/$(basename $CONFIG .yaml)_gpu${GPU_ID}.log
