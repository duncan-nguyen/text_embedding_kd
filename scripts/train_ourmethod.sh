#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="0,1"
export TOKENIZERS_PARALLELISM="false"

echo "======================================"
echo "Training with OurMethod method"
echo "======================================"

python3 main.py \
    --method ourmethod \
    --train_data data/train_set/merged_3_data_5k_each.csv \
    --student_model bert-base-uncased \
    --base_student_model bert-base-uncased \
    --teacher_model Qwen/Qwen3-Embedding-0.6B \
    --batch_size 32 \
    --epochs 5 \
    --lr 2e-5 \
    --max_length 256 \
    --subspace_rank 64 \
    --num_blocks 8 \
    --stability_margin 0.05 \
    --stability_tau 0.05 \
    --w_fusion 1.0 \
    --save_dir checkpoints/ourmethod

echo "======================================"
echo "Training completed!"
echo "======================================"
