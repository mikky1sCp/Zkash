#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

for pd in 0.0 0.1; do
  for wd in 1e-4 1e-2; do
    echo "=== p_drop=$pd  wd=$wd ==="
    PYTHONPATH=src PYTHONWARNINGS="ignore::FutureWarning" \
    python -m zkash.train \
      --config configs/zkash_10m_hard.yaml \
      --override model.p_drop=$pd train.weight_decay=$wd \
                 paths.checkpoint=checkpoints/sweep_pd${pd}_wd${wd}.pt \
      2>&1 | tail -n 6
  done
done