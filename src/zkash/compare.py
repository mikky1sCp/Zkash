"""Compare multi-seed reports produced by zkash.multiseed.

Usage:
    python -m zkash.compare logs/multiseed_A.json logs/multiseed_B.json ...

The first report is the baseline. Each subsequent report is compared
against it on the chosen metric (default: val_acc). Verdict thresholds:

    σ < 1.0        within 1σ   — effect is inside seed noise
    1.0 ≤ σ < 2.0  within 2σ   — weak evidence
    σ ≥ 2.0        SIGNIFICANT — effect survives seed noise
"""
import argparse
import json
import math
from pathlib import Path


def load_report(path):
    with open(path, "r") as f:
        return json.load(f)


def welch_sigma(base, other, key):
    """|Δmean| / sqrt(std_base² + std_other²)."""
    sb = base["aggregate"][key]
    so = other["aggregate"][key]
    pooled = math.sqrt(sb["std"] ** 2 + so["std"] ** 2)
    delta = so["mean"] - sb["mean"]
    if pooled == 0:
        return float("inf") if delta != 0 else 0.0
    return abs(delta) / pooled


def _verdict(sigma, is_baseline):
    if is_baseline:
        return "baseline"
    if sigma < 1.0:
        return "within 1σ"
    if sigma < 2.0:
        return "within 2σ"
    return "SIGNIFICANT"


def format_comparison(reports, key="val_acc"):
    base = reports[0]
    lines = [
        f"=== comparing `{key}` against baseline: {base['config']} ===",
        f"{'config':<40s}  {'mean':>8s}  {'std':>7s}  {'Δ':>8s}  {'σ':>6s}  verdict",
        "-" * 90,
    ]
    sb = base["aggregate"][key]
    for r in reports:
        s = r["aggregate"][key]
        is_base = r is base
        delta = 0.0 if is_base else s["mean"] - sb["mean"]
        sigma = 0.0 if is_base else welch_sigma(base, r, key)
        lines.append(
            f"{Path(r['config']).stem:<40s}  "
            f"{s['mean']:>8.4f}  {s['std']:>7.4f}  "
            f"{delta:>+8.4f}  {sigma:>6.2f}  {_verdict(sigma, is_base)}"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("reports", nargs="+")
    p.add_argument("--key", default="val_acc",
                   choices=["val_acc", "train_acc", "gap", "epoch"])
    args = p.parse_args()

    reports = [load_report(r) for r in args.reports]
    print(format_comparison(reports, key=args.key))


if __name__ == "__main__":
    main()