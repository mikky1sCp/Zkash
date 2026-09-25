#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
PYTHONPATH=src python -m zkash.evaluate \
  --config configs/default.yaml \
  --ckpt checkpoints/zkash10k.pt