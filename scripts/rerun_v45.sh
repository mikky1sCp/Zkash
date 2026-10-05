#!/usr/bin/env bash
# v4.5.0 — full re-run of all three tasks at 5 seeds with the fixed
# label-noise semantics (q = p). Writes logs/v45_*.json.
#
# Runtime on a GTX 1660 SUPER:
#   easy        ~1 min   (5 seeds, early stops at epoch 1)
#   hard        ~2 min   (5 seeds, early stops at epoch ~4)
#   impossible  ~1 h     (5 seeds, 262,144 samples)
#
# Legacy v4.4.0 logs under logs/multiseed_*.json are NOT overwritten.
set -e
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-0 1 2 3 4}"
mkdir -p logs checkpoints

run () {
  local name="$1"; shift
  local cfg="$1"; shift
  echo ""
  echo "########## $name ##########"
  PYTHONPATH=src PYTHONWARNINGS="ignore::FutureWarning" \
  python -m zkash.multiseed \
    --config "$cfg" --seeds $SEEDS \
    --out "logs/v45_${name}.json" "$@"
}

run easy       configs/zkash_10m.yaml
run hard       configs/zkash_10m_hard.yaml
run impossible configs/zkash_10m_impossible.yaml

echo ""
echo "########## v4.5.0 summary (val_acc) ##########"
PYTHONPATH=src python -m zkash.compare \
  logs/v45_easy.json \
  logs/v45_hard.json \
  logs/v45_impossible.json

echo ""
echo "########## Bayes ceilings (v4.5.0 semantics) ##########"
python scripts/bayes_hard.py
python scripts/bayes_impossible.py

echo ""
echo "done. logs -> logs/v45_*.json"