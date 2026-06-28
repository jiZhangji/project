#!/usr/bin/env python3
"""Find the largest per-GPU batch that completes one optimizer step."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--lower", type=int, default=1)
    parser.add_argument("--upper", type=int, default=1024)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    return parser.parse_args()


def run_probe(root, gpus, batch_size, amp_dtype):
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        f"--nproc_per_node={gpus}", "Pretraining/main_pretrain.py",
        "--synthetic_data", "--synthetic_length", "65536",
        "--epochs", "1", "--max_train_steps", "1",
        "--batch_size", str(batch_size), "--num_workers", "0",
        "--amp_dtype", amp_dtype, "--output_dir", "", "--log_dir", "",
        "--init_ckpt", "", "--save_freq", "0",
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    with tempfile.TemporaryFile(mode="w+") as output:
        result = subprocess.run(command, cwd=root, env=env, stdout=output,
                                stderr=subprocess.STDOUT, check=False)
        output.seek(0)
        text = output.read()
    if result.returncode != 0:
        tail = "\n".join(text.splitlines()[-20:])
        return False, tail
    return True, ""


def main():
    args = parse_args()
    if args.lower < 1 or args.upper < args.lower:
        raise SystemExit("Require 1 <= lower <= upper")
    root = Path(__file__).resolve().parents[1]
    low, high = args.lower, args.upper
    best = 0
    while low <= high:
        candidate = (low + high) // 2
        print(f"Probing per-GPU batch size {candidate} on {args.gpus} GPU(s)...", flush=True)
        success, error_tail = run_probe(root, args.gpus, candidate, args.amp_dtype)
        if success:
            best = candidate
            low = candidate + 1
            print("  success", flush=True)
        else:
            high = candidate - 1
            print("  out of memory or runtime failure", flush=True)
            if best == 0 and candidate == args.lower:
                print(error_tail, file=sys.stderr)

    if best == 0:
        raise SystemExit("No working batch size found. Inspect the error above.")
    recommended = max(1, int(best * 0.9))
    print(f"MAX_BATCH_SIZE_PER_GPU={best}")
    print(f"RECOMMENDED_BATCH_SIZE_PER_GPU={recommended}")
    print(f"GLOBAL_BATCH_AT_MAX={best * args.gpus}")


if __name__ == "__main__":
    main()
