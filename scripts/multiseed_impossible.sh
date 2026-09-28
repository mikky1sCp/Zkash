#!/usr/bin/env bash
# v4.4.0 — multi-seed on `impossible`: baseline vs all v4.3.0 variants.
# WARNING: ~1 hour total on a GTX 1660 (262,144 samples × 4 configs × 5 seeds).
set -e
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-0 1 2 3 4}"
CFG=configs/zkash_10m_impossible.yaml
mkdir -p logs

run () {
  local name="$1"; shift
  echo ""
  echo "########## $name ##########"
  PYTHONPATH=src PYTHONWARNINGS="ignore::FutureWarning" \
  python -m zkash.multiseed \
    --config "$CFG" --seeds $SEEDS \
    --out "logs/multiseed_${name}.json" "$@"
}

run impossible_baseline
run impossible_cosine   --override train.schedule=cosine train.warmup_steps=500
run impossible_ls       --override train.label_smoothing=0.1
run impossible_both     --override train.schedule=cosine train.warmup_steps=500 \
                                  train.label_smoothing=0.1

echo ""
echo "########## comparison (val_acc) ##########"
PYTHONPATH=src python -m zkash.compare \
  logs/multiseed_impossible_baseline.json \
  logs/multiseed_impossible_cosine.json \
  logs/multiseed_impossible_ls.json \
  logs/multiseed_impossible_both.json