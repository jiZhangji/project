#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../Pretraining"

python main_pretrain.py \
  --data_path ../dataset \
  --output_dir ../runs/smoke_pretrain \
  --log_dir ../runs/smoke_pretrain \
  --device cuda \
  --amp_dtype bf16 \
  --epochs 1 \
  --batch_size 2 \
  --num_workers 0 \
  --init_ckpt '' \
  --max_train_steps 1
