#!/bin/bash
ENV_NAME="alex_inference"
export NNODES=1
export NPROC_PER_NODE=1
export CUDA_VISIBLE_DEVICES=0
export PATH=~/local_cuda/bin:$PATH
export LD_LIBRARY_PATH=~/local_cuda/lib64:$LD_LIBRARY_PATH

source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Use port 8003 for shard0
export API_PORT=8003

llamafactory-cli api examples/inference/llvmir_o2.yaml

