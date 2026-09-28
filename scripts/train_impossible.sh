#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
PYTHONWARNINGS="ignore::FutureWarning" \
PYTHONPATH=src python -m zkash.train --config configs/zkash_10m_impossible.yaml