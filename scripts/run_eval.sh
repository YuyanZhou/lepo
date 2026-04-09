#!/bin/bash

# 评估启动脚本
# 用法: ./scripts/run_eval.sh <config_file>
# 示例: ./scripts/run_eval.sh configs/eval/qwen2_3b_aime2024.yaml

CONFIG_FILE=$1

if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <config_file>"
    echo "Example: $0 configs/eval/qwen2_3b_aime2024.yaml"
    exit 1
fi

# 设置 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

# 设置 PYTHONPATH，确保能找到 src 模块
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/..${PYTHONPATH:+:$PYTHONPATH}"

# 设置 HuggingFace 缓存目录（强制覆盖，避免权限问题）
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
