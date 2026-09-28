"""Multi-seed runner for Zkash10M — v4.4.0.

Runs one config under N different training seeds and reports
mean ± std for val accuracy, train accuracy, gap, and best epoch.

Purpose: quantify the seed-variance floor, so that the small effects
reported in v4.2.0 (−0.0116) and v4.3.0 (−0.0005 to −0.0011) can be
classified as above or below it.

What varies across seeds:
    - model initialization
    - train/val split
    - batch shuffling
What is held fixed:
    - the underlying data distribution (`data.seed`)

This measures run-to-run training variance on a fixed task.
"""
import argparse
import copy
import json
import os
import statistics
from pathlib import Path

from .train import train, _apply_override
from .utils import load_config


def _stats(values):
    n = len(values)
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "values": list(values),
        "n": n,
    }


def aggregate(results):
    if not results:
        raise ValueError("no results to aggregate")
    return {
        "val_acc":    _stats([r["best_val_acc"]    for r in results]),
        "train_acc":  _stats([r["best_train_acc"]  for r in results]),
        "train_loss": _stats([r["best_train_loss"] for r in results]),
        "gap":        _stats([r["best_train_acc"] - r["best_val_acc"] for r in results]),
        "epoch":      _stats([r["best_epoch"]      for r in results]),
        "seeds":      [r["seed"] for r in results],
    }


def format_summary(name, agg):
    lines = [f"=== {name} ==="]
    rows = [
        ("val acc",    "val_acc"),
        ("train acc",  "train_acc"),
        ("gap",        "gap"),
        ("best epoch", "epoch"),
    ]
    for label, key in rows:
        s = agg[key]
        lines.append(
            f"  {label:<10s}  "
            f"{s['mean']:>8.4f} ± {s['std']:.4f}   "
            f"[min {s['min']:>8.4f}, max {s['max']:>8.4f}, n={s['n']}]"
        )
    return "\n".join(lines)


def run_seeds(cfg, seeds, verbose=True):
    """Run `train(cfg)` once per seed. Returns the list of metric dicts."""
    results = []
    base_ckpt = cfg["paths"]["checkpoint"]
    stem, ext = os.path.splitext(base_ckpt)

    for seed in seeds:
        run_cfg = copy.deepcopy(cfg)
        run_cfg["train"]["seed"] = seed
        run_cfg["paths"]["checkpoint"] = f"{stem}_seed{seed}{ext}"

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"seed {seed}   ->   {run_cfg['paths']['checkpoint']}")
            print(f"{'=' * 60}")

        metrics = train(run_cfg)
        metrics["seed"] = seed
        results.append(metrics)
    return results


def main():
    p = argparse.ArgumentParser(description="Run one config under N seeds.")
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--out", default=None, help="JSON report path")
    p.add_argument("--override", nargs="*", default=[])
    args = p.parse_args()

    cfg = load_config(args.config)
    for kv in args.override:
        _apply_override(cfg, kv)

    print(f"config: {args.config}")
    print(f"seeds:  {args.seeds}")

    results = run_seeds(cfg, args.seeds)
    agg = aggregate(results)

    print()
    print(format_summary(Path(args.config).stem, agg))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        report = {
            "config": args.config,
            "seeds": args.seeds,
            "results": results,
            "aggregate": agg,
        }
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()