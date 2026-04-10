#!/bin/bash

# Evaluation launch script
# Usage: ./scripts/run_eval.sh <config_file>
# Example: ./scripts/run_eval.sh configs/eval/qwen2_3b_aime2024.yaml

CONFIG_FILE=$1

if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <config_file>"
    echo "Example: $0 configs/eval/qwen2_3b_aime2024.yaml"
    exit 1
fi

# Set GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

# Set PYTHONPATH to ensure src module can be found
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/..${PYTHONPATH:+:$PYTHONPATH}"

# Set HuggingFace cache directory (force override to avoid permission issues)
export HF_HOME=~/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

echo "=========================================="
echo "Evaluation Configuration:"
echo "  Config: $CONFIG_FILE"
echo "  GPUs: $NUM_GPUS"
echo "=========================================="

torchrun --nproc_per_node=$NUM_GPUS \
    --master_port=29501 \
    scripts/eval_latent.py \
    --config $CONFIG_FILE \
    --load_balance_method interleaved
