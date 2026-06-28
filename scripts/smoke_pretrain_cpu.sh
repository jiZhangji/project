#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../Pretraining"

python main_pretrain.py \
  --data_path ../dataset \
  --output_dir ../runs/smoke_pretrain_cpu \
  --log_dir ../runs/smoke_pretrain_cpu \
  --device cpu \
  --amp_dtype none \
  --epochs 1 \
  --batch_size 1 \
  --num_workers 0 \
  --init_ckpt '' \
  --max_train_steps 1
