#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."

mkdir -p logs

run () {
  local name="$1"; shift
  echo ""
  echo "=== $name ==="
  PYTHONPATH=src PYTHONWARNINGS="ignore::FutureWarning" \
  python -m zkash.train \
    --config configs/zkash_10m_impossible.yaml \
    --override "$@" \
    > "logs/${name}.log" 2>&1
  local rc=$?
  echo "exit code: $rc"
  grep -E "best val acc|early stopping|Traceback|Error" "logs/${name}.log" || echo "(no match)"
}

run "baseline" \
    train.patience=25 paths.checkpoint=checkpoints/imp_baseline.pt

run "cosine" \
    train.schedule=cosine train.warmup_steps=500 train.patience=25 \
    paths.checkpoint=checkpoints/imp_cos.pt

run "ls" \
    train.label_smoothing=0.1 train.patience=25 \
    paths.checkpoint=checkpoints/imp_ls.pt

run "both" \
    train.schedule=cosine train.warmup_steps=500 \
    train.label_smoothing=0.1 train.patience=25 \
    paths.checkpoint=checkpoints/imp_both.pt