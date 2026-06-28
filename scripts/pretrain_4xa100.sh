#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BATCH_SIZE="${BATCH_SIZE:-128}"
DATA_PATH="${DATA_PATH:-dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/pretrain_4xa100}"

torchrun --standalone --nproc_per_node=4 Pretraining/main_pretrain.py \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --log_dir "$OUTPUT_DIR" \
  --device cuda \
  --amp_dtype bf16 \
  --batch_size "$BATCH_SIZE" \
  --epochs 300 \
  --num_workers 12 \
  --pin_mem \
  --window_size 7 \
  --num_window 4 \
  --mask_ratio 0.8 \
  --save_freq 50 \
  --init_ckpt ''
