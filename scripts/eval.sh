#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
PYTHONWARNINGS="ignore::FutureWarning" \
PYTHONPATH=src python -m zkash.evaluate \
  --config configs/zkash_01m.yaml \
  --ckpt checkpoints/zkash01m.pt
