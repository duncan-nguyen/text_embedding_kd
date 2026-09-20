#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="0,1"
export TOKENIZERS_PARALLELISM="false"

echo "======================================"
echo "Training with RKD method"
echo "======================================"

# Park et al. (2019): lambda_RKD-D = 1, lambda_RKD-A = 2 (RKD-DA).
python3 main.py \
    --method rkd \
    --train_data data/train_set/merged_3_data_5k_each.csv \
    --student_model google-bert/bert-base-uncased \
    --teacher_model Qwen/Qwen3-Embedding-4B \
    --batch_size 32 \
    --epochs 5 \
    --lr 2e-5 \
    --max_length 256 \
    --save_dir checkpoints/rkd \
    --w_task 1.0 \
    --dist_ratio 1.0 \
    --angle_ratio 2.0 \
    --num_workers 2

echo "======================================"
echo "Training completed!"
echo "======================================"
