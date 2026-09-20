$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$env:CUDA_VISIBLE_DEVICES = "0,1"
$env:TOKENIZERS_PARALLELISM = "false"

Write-Host "======================================"
Write-Host "Training with OurMethod method"
Write-Host "======================================"

$METHOD = "ourmethod"
$TRAIN_DATA = "data\train_set\merged_3_data_5k_each.csv"
$STUDENT_MODEL = "bert-base-uncased"
$BASE_STUDENT_MODEL = "bert-base-uncased"
$TEACHER_MODEL = "Qwen/Qwen3-Embedding-0.6B"
$BATCH_SIZE = 32
$EPOCHS = 5
$LR = 2e-5
$MAX_LENGTH = 256
$SUBSPACE_RANK = 64
$NUM_BLOCKS = 8
$STABILITY_MARGIN = 0.05
$STABILITY_TAU = 0.05
$W_FUSION = 1.0
$SAVE_DIR = "checkpoints/ourmethod"

python main.py `
    --method $METHOD `
    --train_data $TRAIN_DATA `
    --student_model $STUDENT_MODEL `
    --base_student_model $BASE_STUDENT_MODEL `
    --teacher_model $TEACHER_MODEL `
    --batch_size $BATCH_SIZE `
    --epochs $EPOCHS `
    --lr $LR `
    --max_length $MAX_LENGTH `
    --subspace_rank $SUBSPACE_RANK `
    --num_blocks $NUM_BLOCKS `
    --stability_margin $STABILITY_MARGIN `
    --stability_tau $STABILITY_TAU `
    --w_fusion $W_FUSION `
    --save_dir $SAVE_DIR

Write-Host "======================================"
Write-Host "Training completed!"
Write-Host "======================================"
