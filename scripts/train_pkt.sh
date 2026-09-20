#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="0,1"
export TOKENIZERS_PARALLELISM="false"

echo "======================================"
echo "Training with PKT method"
echo "======================================"

# Passalis & Tefas transfer without a supervised term (their loop defaults to
# supervised_weight=0), so w_task stays at 0 here.
python3 main.py \
    --method pkt \
    --train_data data/train_set/merged_3_data_5k_each.csv \
    --student_model google-bert/bert-base-uncased \
    --teacher_model Qwen/Qwen3-Embedding-4B \
    --batch_size 32 \
    --epochs 5 \
    --lr 2e-5 \
    --max_length 256 \
    --save_dir checkpoints/pkt \
    --w_task 0.0 \
    --w_pkt 1.0 \
    --pkt_kernel cosine \
    --num_workers 2

echo "======================================"
echo "Training completed!"
echo "======================================"
