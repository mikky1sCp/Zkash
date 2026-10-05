#!/usr/bin/env bash
# v4.4.0 — multi-seed on `hard`: baseline vs v4.2.0 regularization.
# ~2 minutes total on a GTX 1660 (2,048 samples, early stops at epoch ~4).
set -e
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-0 1 2 3 4}"
mkdir -p logs

run () {
  local name="$1"; shift
  local cfg="$1"; shift
  echo ""
  echo "########## $name ##########"
  PYTHONPATH=src PYTHONWARNINGS="ignore::FutureWarning" \
  python -m zkash.multiseed \
    --config "$cfg" --seeds $SEEDS \
    --out "logs/multiseed_${name}.json" "$@"
}

run hard_baseline        configs/zkash_10m_hard.yaml
run hard_v42_regularized configs/zkash_10m_hard_v42.yaml

echo ""
echo "########## comparison (val_acc) ##########"
PYTHONPATH=src python -m zkash.compare \
  logs/multiseed_hard_baseline.json \
  logs/multiseed_hard_v42_regularized.json