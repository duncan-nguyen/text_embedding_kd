#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="0,1"
export TOKENIZERS_PARALLELISM="false"

echo "======================================"
echo "Training with TALAS method"
echo "======================================"

python3 main.py \
    --method talas \
    --train_data data/train_set/merged_3_data_5k_each.csv \
    --student_model jim12345/MiniLMv2-L6-H384-distilled-from-BERT-Base \
    --teacher_model Qwen/Qwen3-Embedding-0.6B \
    --batch_size 32 \
    --epochs 5 \
    --lr 2e-5 \
    --max_length 256 \
    --save_dir checkpoints/talas

echo "======================================"
echo "Training completed!"
echo "======================================"
