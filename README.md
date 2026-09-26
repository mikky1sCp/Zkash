# Zkash10M

**A 10-million-parameter residual network, built to be understood.**

*Depth over width. Norm before activation. Bayes before diagnosis.*

![version](https://img.shields.io/badge/series-v4.3.0-blue?style=flat-square)
![params](https://img.shields.io/badge/params-10%2C000%2C896-green?style=flat-square)
![python](https://img.shields.io/badge/python-%3E%3D3.10-blue?style=flat-square)
![pytorch](https://img.shields.io/badge/pytorch-%3E%3D2.2-orange?style=flat-square)
![license](https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square)

---

## Table of contents

- [What is Zkash?](#what-is-zkash)
- [The scaling series](#the-scaling-series)
- [What changed from v3 to v4](#what-changed-from-v3-to-v4)
- [Architecture in full](#architecture-in-full)
- [The three ingredients, explained](#the-three-ingredients-explained)
- [Results across the series](#results-across-the-series)
- [At the ceiling](#at-the-ceiling)
- [The corrected Bayes formula](#the-corrected-bayes-formula)
- [Two falsified predictions (v4.2.0, v4.3.0)](#two-falsified-predictions-v420-v430)
- [The only intervention that ever worked](#the-only-intervention-that-ever-worked)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Training protocol](#training-protocol)
- [Python API](#python-api)
- [Tests](#tests)
- [Project structure](#project-structure)
- [Roadmap](#roadmap)
- [Common questions](#common-questions)
- [Citation](#citation)

---

## What is Zkash?

**Zkash** is a family of neural networks with **exact parameter budgets** — 10K, 100K, 1M, 10M — each introducing one new architectural idea. They share the same interface (64-dim input → 8-class output), so they are directly comparable on the same task.

Zkash10M is the fourth and largest entry. It abandons the wide-MLP paradigm of its predecessors in favor of **depth**: 20 pre-norm residual blocks, RMSNorm normalization, and GELU activations.

Most educational networks are either **too small to be interesting** (XOR MLPs, 3-layer toys) or **too large to reason about** (production transformers). Zkash10M sits in the sweet spot:

- **10 million parameters** — large enough for depth, residual connections, and normalization to matter
- **42 linear layers** — deep enough to exhibit real gradient flow behavior
- **One file, no dependencies** — `model.py` is under 80 lines of readable code
- **Trains in 60 seconds on a GTX 1660** — you can iterate on ideas without a cluster
- **Fails reproducibly, with a Bayes ceiling to prove it** — and publishes its own falsified predictions

---

## The scaling series

Zkash is a lineage. Each generation added one idea and multiplied the parameter count by 10.

| Generation | Model | Params | Linear layers | Key idea | Report |
|---|---|---:|---:|---|---|
| v1 | **Zkash-10K** | 10,000 | 3 | baseline MLP | ZK-2025-01 |
| v2 | **Zkash-0.1M** | 100,000 | 4 | depth + dropout | ZK-2025-02 |
| v3 | **Zkash-1M** | 1,000,000 | 4 | wide MLP | ZK-2025-03 |
| **v4** | **Zkash10M** | **10,000,896** | **42** | **residual + RMSNorm + GELU** | **ZK-2025-04** |

Each model was designed to answer a specific question:

| Model | Question it answers |
|---|---|
| Zkash-10K | What is the smallest network that can classify 8 Gaussian clouds? |
| Zkash-0.1M | Does 10× more parameters help on the same task? |
| Zkash-1M | Does 100× more parameters help? |
| **Zkash10M** | **Does depth + residual + norm help when width has stopped helping?** |

---

## What changed from v3 to v4

The jump from Zkash-1M to Zkash10M is not just "more parameters". It is a **change of paradigm**.

### Zkash-1M (wide MLP)

```
Input(64) → Linear(508) → ReLU → Linear(640) → ReLU → Linear(988) → ReLU → Linear(8)
```

Four layers. Almost a million parameters. **All capacity is in width.** Each layer is enormous; the network has no depth.

Problems with this design:
- Gradients must traverse every layer without shortcuts
- No normalization — each layer must learn its own scale
- ReLU kills half the neurons; no smoothness

### Zkash10M (deep ResNet)

```
Input(64) → Stem(512)
   ↓
20 × [ RMSNorm → Linear(486) → GELU → Linear(512) → +residual ]
   ↓
RMSNorm → Head(8)
```

Twenty identical blocks. **All capacity is in depth.** Each block is narrow; the network is deep.

Three ideas do the heavy lifting:

| Idea | What it does | Why it matters |
|---|---|---|
| **Residual connection** | `x → x + f(x)` | Gradients flow directly to early layers; training 20+ layers becomes trivial |
| **Pre-norm (RMSNorm)** | Normalize *before* each block | Keeps activation scale stable; no warmup needed |
| **GELU** | Smooth activation | Better gradient signal than ReLU; standard in modern models |

### Side-by-side

| Property | Zkash-1M | Zkash10M |
|---|---:|---:|
| Trainable parameters | 1,000,000 | **10,000,896** |
| Linear layers | 4 | **42** |
| Residual connections | 0 | **20** |
| Normalization layers | 0 | **21 (RMSNorm)** |
| Activation | ReLU | **GELU** |
| Bias terms | yes | **none** |
| Effective depth | 4 | **~42** |
| Forward MACs / sample | ~1M | ~10.2M |
| fp32 size | 4 MB | **40 MB** |
| Time per epoch (GTX 1660, batch=128) | ~2 s | ~10 s |

**Takeaway:** Zkash10M trades width for depth. The parameter count grows 10×, but *where* the parameters live changes completely.

---

## Architecture in full

```
Input(64)
   │
   ▼
┌──────────────────────────────────────────────────┐
│  Stem: Linear(64 → 512, no bias)                 │  32,768 params
└──────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────┐
│  ResidualBlock × 20                              │  498,176 × 20
│  ────────────────────────────────                │
│     x ──┬───────────────────────────────┐        │
│         │                               │        │
│         ▼                               │        │
│     RMSNorm(512)                        │        │
│         │                               │        │
│     Linear(512 → 486, no bias)          │        │
│         │                               │        │
│     GELU                                │        │
│         │                               │        │
│     [optional Dropout]                  │        │
│         │                               │        │
│     Linear(486 → 512, no bias)          │        │
│         │                               │        │
│         ▼                               │        │
│         + ◄─────────────────────────────┘        │
│         │                                        │
└─────────┼────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────┐
│  Final RMSNorm(512)                              │  512 params
└──────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────┐
│  Head: Linear(512 → 8, no bias)                  │  4,096 params
└──────────────────────────────────────────────────┘
          │
          ▼
       Logits(8) → softmax
```

### Parameter budget

| Component | Shape | Weights | Biases | Total | % of total |
|---|---|---:|---:|---:|---:|
| `stem` | Linear(64, 512), no bias | 32,768 | 0 | **32,768** | 0.33% |
| `block` × 20 | RMSNorm + Linear + Linear | 498,176 | 0 | **9,963,520** | 99.63% |
| `final_norm` | RMSNorm(512) | 512 | 0 | **512** | 0.005% |
| `head` | Linear(512, 8), no bias | 4,096 | 0 | **4,096** | 0.04% |
| | | | **Total** | **10,000,896** | **100%** |

Per-block breakdown:

| Sub-layer | Shape | Params |
|---|---|---:|
| `RMSNorm.weight` | (512,) | 512 |
| `fc1.weight` | (486, 512) | 248,832 |
| `fc2.weight` | (512, 486) | 248,832 |
| **Per block** | | **498,176** |

**Dropout adds zero parameters. Weight decay adds zero. Cosine LR adds zero. Label smoothing adds zero.** Every intervention tested in v4.2.0 and v4.3.0 leaves `count_params` at exactly 10,000,896.

---

## The three ingredients, explained

### 1. Residual connections — how deep networks train at all

Without shortcuts, a gradient flowing backward through 20 layers is multiplied by 20 Jacobians. If each is slightly below 1, the signal vanishes; above 1, it explodes.

A residual block computes `y = x + f(x)`. Backprop through this gives:

$$\frac{\partial y}{\partial x} = I + \frac{\partial f}{\partial x}$$

The identity matrix `I` is a **direct highway** for gradients. Even if `∂f/∂x` is tiny, the gradient still reaches layer 0 untouched. This is why ResNet (He et al., 2015) made 100+ layer networks trainable.

**Try it yourself:**

```python
from zkash import ResidualBlock
import torch

blk = ResidualBlock(64, 64)
with torch.no_grad():
    blk.fc1.weight.zero_()
    blk.fc2.weight.zero_()

x = torch.randn(4, 64)
assert torch.allclose(blk(x), x)   # ✓ zeroed block = identity map
```

### 2. RMSNorm — cheap, effective normalization

LayerNorm computes mean and variance; RMSNorm only needs the root-mean-square:

$$\mathrm{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\|x\|_2^2 + \epsilon}}$$

- **No mean subtraction** — cheaper, works equally well in practice
- **One parameter per dimension** (`γ`), no bias
- **Pre-norm placement** — normalize before the block, not after — this is what modern transformers do

**Try it yourself:**

```python
from zkash import RMSNorm
import torch

n = RMSNorm(64)
x = torch.randn(4, 64)
out = n(x)
rms = out.pow(2).mean(dim=-1).sqrt()
assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)
```

### 3. GELU — a smooth ReLU

$$\mathrm{GELU}(x) = x \cdot \Phi(x)$$

where `Φ` is the standard Gaussian CDF. Unlike ReLU, GELU is **smooth everywhere** and allows small negative values through. This gives a better-behaved gradient signal and is the default in BERT, GPT, and ViT.

---

## Results across the series

All models were trained on **easy** synthetic data (n=65,536, `center_scale=3.0`, `noise=0.7`, no label noise).

| Model | Params | Train acc | Val acc | Gap | Time/epoch (GTX 1660) |
|---|---:|---:|---:|---:|---:|
| Zkash-10K | 10,000 | 1.000 | 1.000 | 0.00 | <1 s |
| Zkash-0.1M | 100,000 | 1.000 | 1.000 | 0.00 | ~1 s |
| Zkash-1M | 1,000,000 | 1.000 | 1.000 | 0.00 | ~2 s |
| **Zkash10M** | **10,000,896** | **1.000** | **1.000** | **0.00** | **~10 s** |

**Takeaway:** on trivially separable data, parameter count doesn't matter. All four models converge to 100%. The interesting behavior appears only when the task becomes hard.

### On hard and impossible tasks

The complete picture, with **empirical Bayes ceilings** measured via nearest-centroid Monte Carlo (N = 200,000):

| Config | Clean Bayes | Noisy Bayes | Best model | **Gap to Bayes** | Peak epoch |
|---|---:|---:|---:|---:|---:|
| easy | 1.0000 | 1.0000 | 1.0000 | **0.0000** | 1 |
| hard | — | **0.5879** | 0.5458 | **−0.0421** | 2–4 |
| impossible | 0.8606 | **0.7142** | 0.7128 | **−0.0014** | 5 |

**Takeaway:** the train/val gap tells you almost nothing about which task has room to improve. `hard` has a 0.39 gap and is 4.2 points from Bayes. `impossible` has a 0.006 gap and is **0.14 points** from Bayes. Only the ceiling measurement distinguishes them.

---

## At the ceiling

v4.1.0 classified the two hard tasks as "opposite failures": `hard` overfits, `impossible` underfits. **v4.2.0 and v4.3.0 refute that classification.** Both are **memorization-dominated and ceiling-bound**. They differ in distance to Bayes, not in regime.

### hard — memorization-dominated, 4.2 points below ceiling

| Epoch | Train acc | Val acc | Gap |
|---:|---:|---:|---:|
| 1 | 0.3807 | 0.4694 | −0.089 |
| 4 | 0.7407 | **0.5428** ★ | +0.198 |
| 5 | 0.8188 | 0.4988 | +0.320 |
| 10 | 0.9286 | 0.4743 | **+0.454** |

The gap looks like classical overfitting. It isn't. Two facts disqualify that diagnosis:

1. The val peak is **4.2 points below the true Bayes ceiling** (0.5428 vs 0.5879).
2. **Regularization makes val worse, not better** (−0.0116 across two dataset sizes).

**Correct remedy:** none. `hard` is closed.

### impossible — memorization-dominated, 0.14 points below ceiling

| Epoch | Train acc | Val acc | Gap |
|---:|---:|---:|---:|
| 1 | 0.6944 | 0.7069 | −0.013 |
| 5 | 0.7089 | **0.7128** ★ | −0.004 |
| 10 | 0.7135 | 0.7099 | +0.004 |
| 15 | 0.7188 | 0.7093 | +0.010 |

Train and validation stay together at ~0.71. The gap never exceeds 0.01. Under the v4.1.0 diagnosis, this was underfitting. **v4.3.0 falsifies that**: cosine LR + warmup + label smoothing all land within ±0.0011 of baseline, and the corrected Bayes ceiling (0.7142) shows the peak is **0.14 points** from the theoretical maximum.

**Correct remedy:** none. `impossible` is closed.

### Summary

| Task | Gap to Bayes | Peak epoch | Behavior | Fixable? |
|---|---:|---:|---|---|
| `hard` | −0.042 | 2–4 | Ceiling-bound memorization (fast) | **No** |
| `impossible` | −0.001 | 5 | Ceiling-bound memorization (slow) | **No** |

**This is the central lesson of v4.3.0.** A train/val gap cannot tell you whether regularization will help. A 0.39 gap at 0.54 val and a 0.006 gap at 0.71 val can **both** be memorization-dominated at a Bayes ceiling. The gap is not diagnostic. **The ceiling is.**

---

## The corrected Bayes formula

**Versions v1.0.0 through v4.2.0 used an incorrect Bayes accuracy formula.** It assumed perfect clean-label separability:

$$\text{Acc}_{\text{Bayes}}(p) = 1 - p \cdot \frac{C-1}{C} \quad \text{(assumes } A_{\text{clean}} = 1\text{)}$$

This is true for `easy` (classes don't overlap) but **false** for `hard` and `impossible`, where feature noise causes class overlap. The correct formula is:

$$\boxed{\;\text{Acc}_{\text{Bayes}}(p) = A_{\text{clean}} \cdot \left(1 - \frac{p(C-1)}{C}\right) + (1 - A_{\text{clean}}) \cdot \frac{p}{C}\;}$$

At `A_clean = 1` this reduces to the old formula. At `A_clean < 1` it produces a **strictly lower** ceiling.

### The 11-point correction for `impossible`

| `p` | Old formula | **Corrected** | Error |
|---|---:|---:|---:|
| 0.0 | 1.0000 | **0.8606** | −0.1394 |
| 0.1 | 0.9125 | **0.7870** | −0.1255 |
| **0.2** | **0.8250** | **0.7135** | **−0.1115** |
| 0.3 | 0.7375 | **0.6399** | −0.0976 |

Monte Carlo confirms the corrected value: `impossible` noisy Bayes = **0.7142** (vs analytical 0.7135, difference is MC noise).

**Consequence.** WHITEPAPER v4.1.0 diagnosed `impossible` as underfitting — 11.2 points below Bayes — and predicted that LR schedules would close that gap. That diagnosis was an artifact of the incorrect formula. The true gap is **0.14 points**. There is nothing to close.

### The corrected Bayes loss

The old formula used `p` as the wrong-label rate. In `data.py`, re-drawn labels may coincide with the original, so the true wrong-label rate is `q = p·(C−1)/C = 0.175` at `p = 0.2`, not `0.2`.

$$\mathcal{L}_{\text{Bayes}}(q) = -(1-q)\ln(1-q) - q\ln\!\left(\frac{q}{C-1}\right)$$

| `p` | `q` | Old loss | **Corrected loss** |
|---|---:|---:|---:|
| 0.0 | 0.000 | 0.000 | **0.000** |
| 0.1 | 0.0875 | 0.443 | **0.467** |
| **0.2** | **0.175** | 0.888 | **0.804** |
| 0.3 | 0.263 | 1.333 | **1.086** |

For `impossible` at `p = 0.2`, the Bayes loss floor is **0.804**, not 0.888. The v4.1.0 "margin above Bayes = 0.142" figure was inflated; the corrected margin is 0.256 nats, but the **val accuracy is already at Bayes** — the gap is confidence, not ranking.

---

## Two falsified predictions (v4.2.0, v4.3.0)

Two releases tested the two natural "fix the hard task" hypotheses. **Both predictions were on the record before the runs. Both were wrong.**

### v4.2.0 — dropout + weight decay on `hard`

**Predicted (v4.1.0 §9.1):** val 0.60–0.65, gap ~0.10.
**Observed:** val 0.5306, gap 0.2168.

2×2 ablation `{dropout, wd} ∈ {off, on} × n_samples ∈ {2,048, 8,192}`:

| n_samples | regularization | Train acc @ best | Val acc @ best | Best epoch |
|---:|---|---:|---:|---:|
| 2,048 | none (v4.1.0) | 0.7407 | **0.5428** | 4 |
| 2,048 | dropout 0.1 + wd 1e-2 | 0.7474 | **0.5306** | 4 |
| 8,192 | none | 0.5418 | **0.5458** | 2 |
| 8,192 | dropout 0.1 + wd 1e-2 | 0.5383 | **0.5348** | 2 |
| — | **true Bayes** | — | **0.5879** | — |

| Effect | 2,048 | 8,192 | Mean |
|---|---:|---:|---:|
| **Cost of regularization** | −0.0122 | −0.0110 | **−0.0116** |
| **Gain from 4× data** | +0.0030 | +0.0042 | **+0.0036** |

**Regularization cost 1.2 points. 4× more data gained 0.4 points.** Both are inside single-seed noise; both are consistent across the 2×2.

### v4.3.0 — cosine LR + label smoothing on `impossible`

**Predicted (v4.2.0 §9.3):** val 0.76–0.80.
**Observed:** val 0.7117–0.7124.

Four runs on `impossible` (262,144 samples, seed 0, patience 25):

| Run | Schedule | LS | Best val | Best epoch | Δ vs baseline |
|---|---|---|---:|---:|---:|
| baseline | constant | 0.0 | **0.7128** | 5 | — |
| cosine | cosine | 0.0 | **0.7123** | 5 | −0.0005 |
| LS | constant | 0.1 | **0.7124** | 5 | −0.0004 |
| both | cosine | 0.1 | **0.7117** | 5 | −0.0011 |
| **Bayes (corrected)** | — | — | **0.7142** | — | — |

**All four runs within ±0.0011 of baseline.** All peak at epoch 5. All sit 0.19–0.25 points below Bayes.

Additional evidence: baseline train loss @ best epoch **1.0607**; label smoothing train loss **1.2456** — LS raises the loss floor by exactly the predicted ~0.185 nats, but **val accuracy is unchanged to within noise**. The model's argmax was already correct; only its confidence was disturbed.

### Consolidated ledger

| Release | Prediction | Observed | Status |
|---|---|---|---|
| v4.1.0 §9.1 | `hard` val 0.60–0.65 with dropout+wd | 0.5306 | **✗** |
| v4.1.0 §9.1 | `hard` train acc ~0.75 | 0.7474 | ✓ |
| v4.1.0 §9.1 | `hard` gap ~0.10 | 0.2168 | **✗** |
| v4.2.0 §9.3 | `impossible` val 0.76–0.80 with cosine+LS | 0.7117–0.7124 | **✗** |

**2 of 5 predictions correct.** Every prediction that involved raising val accuracy was wrong. That is itself a finding: the model was already at its ceiling in every case, and the ceiling was not measured.

---

## The only intervention that ever worked

| Intervention | Task | Effect on val acc |
|---|---|---:|
| **Early stopping + best checkpoint (v4.1.0)** | **`hard`** | **+0.103** |
| **Early stopping + best checkpoint (v4.1.0)** | **`impossible`** | **+0.099** |
| Dropout + weight decay (v4.2.0) | `hard` | −0.0116 |
| 4× more data (v4.2.0) | `hard` | +0.0036 |
| Cosine LR + warmup (v4.3.0) | `impossible` | −0.0005 |
| Label smoothing 0.1 (v4.3.0) | `impossible` | −0.0004 |
| Both (v4.3.0) | `impossible` | −0.0011 |

**Early stopping is the only mechanism in five releases that reliably improved val accuracy.** It costs nothing: no parameters, no backward passes, no architectural change. Every other intervention is within single-seed noise or negative.

---

## Quickstart

### Install

```bash
git clone https://github.com/mikky1sCp/zkash10m.git
cd zkash10m

python -m venv .venv
source .venv/Scripts/activate     # Windows / Git Bash
# source .venv/bin/activate       # Linux / macOS

# Use CUDA wheel — plain PyPI installs CPU-only torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

### Sanity check

```bash
python -c "
from zkash import Zkash10M, count_params
n = count_params(Zkash10M())
print('params:', n)
assert n == 10_000_896
"
```

**Expected:** `params: 10000896`.

### Train

```bash
bash scripts/train.sh              # easy — ~30 seconds
bash scripts/train_hard.sh         # hard v4.1.0 baseline — ~15 seconds
bash scripts/train_impossible.sh   # impossible — ~3 minutes
bash scripts/sweep_hard.sh         # hard v4.2.0 2×2 ablation — ~3 minutes
```

### v4.3.0 impossible ablation (four runs)

```bash
export PYTHONIOENCODING=utf-8   # Windows only — train.py prints a ★ marker

CONFIG=configs/zkash_10m_impossible.yaml
BASE="PYTHONPATH=src python -m zkash.train --config $CONFIG"

$BASE --override train.patience=25 \
      paths.checkpoint=checkpoints/imp_baseline.pt

$BASE --override train.schedule=cosine train.warmup_steps=500 train.patience=25 \
      paths.checkpoint=checkpoints/imp_cos.pt

$BASE --override train.label_smoothing=0.1 train.patience=25 \
      paths.checkpoint=checkpoints/imp_ls.pt

$BASE --override train.schedule=cosine train.warmup_steps=500 \
                 train.label_smoothing=0.1 train.patience=25 \
      paths.checkpoint=checkpoints/imp_both.pt
```

### Empirical Bayes ceilings

```bash
python scripts/bayes_hard.py         # 0.5879
python scripts/bayes_impossible.py   # clean 0.8606, noisy 0.7142
```

### Evaluate

```bash
bash scripts/eval.sh
```

### Test

```bash
pytest -q
```

---

## Configuration

All knobs live in `configs/*.yaml`. Five configs ship by default.

### `configs/zkash_10m.yaml` — easy

```yaml
model:
  n_in: 64
  n_out: 8
  dim: 512
  hidden: 486
  depth: 20
  p_drop: 0.0

data:
  n_samples: 65536
  n_classes: 8
  noise: 0.7
  center_scale: 3.0
  label_noise: 0.0
  seed: 0

train:
  device: auto
  epochs: 60
  patience: 10
  min_delta: 0.0
  batch_size: 128
  lr: 1.0e-3
  weight_decay: 1.0e-4
  seed: 0

paths:
  checkpoint: checkpoints/zkash10m.pt
```

### `configs/zkash_10m_hard.yaml` — v4.1.0 baseline

```yaml
data:
  n_samples: 2048
  noise: 1.5
  center_scale: 0.3
  label_noise: 0.0
```

### `configs/zkash_10m_hard_v42.yaml` — v4.2.0 (falsified prediction)

```yaml
model:
  p_drop: 0.1

data:
  n_samples: 2048
  noise: 1.5
  center_scale: 0.3
  label_noise: 0.0

train:
  epochs: 100
  patience: 25
  weight_decay: 1.0e-2
```

### `configs/zkash_10m_impossible.yaml` — v4.1.0 baseline

```yaml
data:
  n_samples: 262144
  noise: 1.5
  center_scale: 0.5
  label_noise: 0.2

train:
  batch_size: 256
```

### `configs/zkash_10m_impossible_v43.yaml` — v4.3.0 (falsified prediction)

```yaml
train:
  epochs: 60
  patience: 25
  batch_size: 256
  schedule: cosine
  warmup_steps: 500
  min_lr_ratio: 0.0
  label_smoothing: 0.1
```

---

## Training protocol

| Hyperparameter | v4.1.0 | v4.2.0 | v4.3.0 |
|---|---|---|---|
| Optimizer | AdamW (β = 0.9, 0.999) | same | same |
| Learning rate | 1·10⁻³ (constant) | same | **cosine (impossible only)** |
| Weight decay | 1·10⁻⁴ | **1·10⁻² (hard only)** | same |
| Weight decay groups | — | **matrices / 1-D split** | same |
| Batch size | 128 (256 impossible) | same | same |
| Epochs | 60 max, patience 10 | 100 max, patience 25 | 60 max, patience 25 |
| Loss | Cross-entropy | same | **+ LS 0.1 (impossible only)** |
| Warmup | none | none | **500 steps (impossible only)** |
| Gradient clipping | **none** | none | none |
| Dropout | **none** | **0.1 (hard only)** | same |
| Early stopping | **yes** (patience 10 on `val_acc`) | yes (patience 25) | yes (patience 25) |

**Why no gradient clipping?** Pre-norm residual networks train cleanly from epoch 1. This is a structural property, not a missing feature.

**Why early stopping?** v4.1.0 added it and recovered +10 percentage points on both hard tasks, for free. **It remains the only mechanism in this project that reliably improves val acc.** See §"The only intervention that ever worked".

**Why parameter groups in v4.2.0?** RMSNorm γ must not be decayed — shrinking it fights pre-norm's own stabilization. The optimizer gets two groups; the model's parameter count is unchanged.

**Why cosine + LS in v4.3.0?** To **falsify** the underfitting hypothesis from v4.1.0. They were predicted to raise val acc by +0.05–0.08. They did not. The result is in the falsification ledger.

---

## Python API

### Load and use

```python
import torch
from zkash import Zkash10M, count_params

model = Zkash10M()
print(count_params(model))                # 10000896

x = torch.randn(4, 64)
logits = model(x)
print(logits.shape)                       # torch.Size([4, 8])
```

### Inspect the architecture

```python
from zkash import Zkash10M, ResidualBlock, RMSNorm

m = Zkash10M()
print(len(m.blocks))                      # 20
print(m.blocks[0].fc1.weight.shape)       # torch.Size([486, 512])
print(m.blocks[0].fc2.weight.shape)       # torch.Size([512, 486])
print(m.blocks[0].norm.weight.shape)      # torch.Size([512])
```

### Load a checkpoint

```python
import torch
from zkash import Zkash10M

state = torch.load("checkpoints/zkash10m.pt", map_location="cpu")
model = Zkash10M()
model.load_state_dict(state["model"])
model.eval()

print("best val acc:", state["best_val_acc"])
print("best epoch:  ", state["best_epoch"])
```

### Verify the residual identity

```python
from zkash import ResidualBlock
import torch

blk = ResidualBlock(64, 64)
with torch.no_grad():
    blk.fc1.weight.zero_()
    blk.fc2.weight.zero_()

x = torch.randn(4, 64)
assert torch.allclose(blk(x), x)          # ✓
```

### Build weight decay parameter groups

```python
from zkash.model import Zkash10M
from zkash.train import build_param_groups
import torch

model = Zkash10M()
groups = build_param_groups(model, weight_decay=1e-2)
print(len(groups[0]["params"]))   # 42  — weight matrices
print(len(groups[1]["params"]))   # 21  — RMSNorm γ
```

---

## Tests

```bash
pytest -q
```

| Test | What it checks |
|---|---|
| `test_param_count` | exact 10,000,896 budget |
| `test_rmsnorm_shape` | RMSNorm preserves shape |
| `test_rmsnorm_unit_rms` | output has unit RMS |
| `test_residual_identity_at_zero` | zeroed block = identity |
| `test_forward_shape` | output shape `(N, 8)` |
| `test_forward_batch_sizes` | batch sizes 1, 8, 32, 128 |
| `test_forward_dtype` | output dtype is float32 |
| `test_forward_finite` | no NaNs or infs |
| `test_backward_all_params_get_grad` | no dead parameters |
| `test_state_dict_roundtrip` | serialization correctness |
| `test_early_stopping_saves_best_state` | best-checkpoint behavior |
| `test_patience_config_present` | all configs expose `patience` |
| `test_two_groups_returned` | `build_param_groups` contract |
| `test_no_parameter_appears_twice` | no double-counting in groups |
| `test_group_union_covers_all_trainable_params` | 10,000,896 preserved |
| `test_rmsnorm_params_are_not_decayed` | 21 norm tensors excluded |
| `test_linear_weights_are_decayed` | 42 matrices included |
| `test_decay_group_size_matches_budget` | group sizes exact |
| `test_norm_weight_actually_untouched_by_decay` | semantic check |

**27 tests, ~4 s.**

---

## Project structure

```
zkash10m/
├── README.md
├── WHITEPAPER.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── configs/
│   ├── zkash_10m.yaml
│   ├── zkash_10m_hard.yaml
│   ├── zkash_10m_hard_v42.yaml
│   ├── zkash_10m_impossible.yaml
│   └── zkash_10m_impossible_v43.yaml
├── src/
│   └── zkash/
│       ├── __init__.py
│       ├── model.py        # RMSNorm, ResidualBlock, Zkash10M — one file
│       ├── data.py         # synthetic Gaussian clouds
│       ├── train.py        # training loop, early stopping, param groups, LR, LS
│       ├── evaluate.py     # validation with checkpoint metadata
│       └── utils.py        # seeds, device, config
├── scripts/
│   ├── train.sh
│   ├── train_hard.sh
│   ├── train_impossible.sh
│   ├── sweep_hard.sh       # v4.2.0 2×2 ablation
│   ├── sweep_impossible.sh # v4.3.0 four-run ablation
│   ├── bayes_hard.py       # empirical Bayes for hard
│   ├── bayes_impossible.py # empirical Bayes for impossible
│   └── eval.sh
├── tests/
│   ├── test_model.py
│   └── test_param_groups.py
└── checkpoints/
    └── .gitkeep
```

The entire model — normalization, residual block, and the network itself — lives in **one file under 80 lines**. Read it top to bottom; there is no hidden machinery.

---

## Roadmap

| Version | Status | What it adds | Observed effect |
|---|---|---|---|
| v1.0.0 | ✅ | Baseline, no regularization | documents failure |
| v4.1.0 | ✅ | Early stopping + best checkpoint | **+0.10 on hard & impossible** |
| v4.2.0 | ✅ | dropout + wd, `hard` | **−0.0116 on hard (falsified)** |
| **v4.3.0** | ✅ | **cosine + LS on `impossible`; corrected Bayes formula** | **−0.0005 to −0.0011 (falsified)** |
| v4.4.0 | 🔜 | Multi-seed runs (5 seeds) — mean ± std | quantifies variance floor |
| v4.5.0 | 📅 | `data.py` label-noise fix + full re-run | consistency with corrected §5.1 |
| v5.0.0 | 📅 | Zkash100M — depth 100, exact 100M params | next scale-up |

### v4.4.0 predictions (falsifiable)

**Multi-seed, 5 seeds:**

| Task | Prediction |
|---|---|
| `hard` val acc, 5-seed mean ± std | **0.545 ± 0.008** |
| `impossible` val acc, 5-seed mean ± std | **0.713 ± 0.003** |
| Effect of v4.2.0 regularization on `hard` | inside 1σ of seed noise |

If the std is larger than predicted, the "effects" in v4.2.0/v4.3.0 become even more clearly null. If smaller, some of the small effects may survive.

### v4.5.0 predictions (falsifiable)

**Fix `data.py` label noise to draw from `C-1` other classes**, so `p` becomes the true wrong-label rate. Re-run all three configs.

| Task | Prediction |
|---|---|
| `easy` | unchanged at 1.0000 |
| `hard` | unchanged at 0.5458 (no label noise) |
| `impossible` at `p = 0.2` | **drops by ~1.5 points to ~0.70**, because the true wrong rate rises from 0.175 to 0.20 |

This will confirm the corrected Bayes formula quantitatively.

### v5.0.0 — Zkash100M

Depth 100, exact 100M parameters. Same architecture, 5× deeper. The open question is whether depth 100 reaches the ceiling faster — it cannot exceed the ceiling, which is task-determined.

| Task | Prediction |
|---|---|
| `hard` val acc | **0.586 ± 0.005, peak at epoch 1–2** |
| `impossible` val acc | **0.714 ± 0.002, peak at epoch 3–5** |

**Neither exceeds Bayes. This prediction is on the record.**

---

## Common questions

**Why 512 → 486 → 512 inside each block?**
The bottleneck (`hidden < dim`) forces each block to compress and re-expand, which acts as a mild regularizer. It also makes the block's parameter share ~5% of the budget instead of ballooning.

**Why no bias in any Linear layer?**
After RMSNorm, a bias is redundant — the normalization already re-centers activations. Dropping biases saves ~20,000 parameters and simplifies the accounting.

**Why 20 blocks and not 4 wide ones?**
Depth is the entire point of v4. Four wide blocks would just be a bigger MLP with the same gradient-flow problems. Twenty narrow blocks with residuals are the smallest example of a *deep* network that trains cleanly.

**Why not use LayerNorm?**
RMSNorm does the same job with half the operations and one parameter per dimension. It's the modern default in LLaMA, T5, and Gemma.

**Why does it train without warmup?**
Pre-norm keeps activation scale bounded at every layer, so gradients never spike. Warmup exists to prevent early-training instability — pre-norm prevents that instability structurally.

**Why is val accuracy low on some tasks?**
Because on those tasks, the model is **close to the Bayes ceiling** — and the ceiling is low. For `hard`, empirical Bayes is 0.5879; the best run reaches 0.5458. For `impossible`, corrected Bayes is 0.7142; the best run reaches 0.7128.

**Why did dropout + weight decay make `hard` *worse*?**
Because `hard` is memorization-dominated and ceiling-bound. Regularization suppresses the (already-correct) signal fit along with the noise fit. Net effect: −0.0116 across two dataset sizes.

**Why didn't cosine LR + label smoothing help `impossible`?**
Because `impossible` is also ceiling-bound. All four v4.3.0 runs land within ±0.0011 of baseline, and peak at epoch 5 — the same epoch as baseline. Cosine and LS change *how the model gets there*, not *where it stops*.

**Why did 4× more data not help `hard`?**
The memorization transition happens at a fixed number of gradient steps (64–128), set by the task's signal-to-noise ratio, not by dataset size. More data makes the peak arrive earlier; it doesn't move the peak.

**Why does the weight decay use parameter groups?**
RMSNorm γ should not be decayed — shrinking it fights pre-norm's own stabilization. The 21 norm tensors (10,752 params) are excluded; the 42 weight matrices are decayed. Parameter count is unchanged.

**Why does early stopping help so much?**
Because we save the **best** epoch, not the **last**. On `hard`, val peaks at 0.5428 on epoch 4 and drops to 0.4743 by epoch 10 — but v1.0.0 kept saving the last. Early stopping recovers the peak. It remains the only mechanism in this project that reliably improves val acc.

**Wasn't the v4.1.0 "impossible is underfitting" diagnosis correct?**
No. It was based on an incorrect Bayes formula. The old formula gave 0.825 as the ceiling; the corrected formula gives 0.7135 (MC: 0.7142). The apparent 11.2-point "gap" was an artifact of the formula, not of the model.

**Is this model production-ready?**
No. It's an educational reference. It has documented failure modes on non-trivial tasks, and both v4.2.0 and v4.3.0 interventions produced null or negative results. But it *does* have a complete, correct Bayes analysis — which is more than most production models.

---

## Citation

```bibtex
@techreport{zkash10m,
  title       = {Zkash10M: A 10M-Parameter Residual Reference Network for Education},
  number      = {ZK-2025-04},
  institution = {Zkash Project},
  year        = {2025},
  note        = {Version 4.3.0}
}
```

**References**

- He, K., Zhang, X., Ren, S., Sun, J. (2015). *Deep Residual Learning for Image Recognition.* arXiv:1512.03385.
- Zhang, B., Sennrich, R. (2019). *Root Mean Square Layer Normalization.* arXiv:1910.07467.
- Hendrycks, D., Gimpel, K. (2016). *Gaussian Error Linear Units (GELUs).* arXiv:1606.08415.
- Veit, A., Wilber, M., Belongie, S. (2016). *Residual Networks Behave Like Ensembles of Relatively Shallow Networks.* arXiv:1605.06431.
- Loshchilov, I., Hutter, F. (2019). *Decoupled Weight Decay Regularization.* arXiv:1711.05101.
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting.* JMLR 15(1).
- Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., Wojna, Z. (2016). *Rethinking the Inception Architecture for Computer Vision.* CVPR.
- Prechelt, L. (1998). *Early Stopping — But When?* In: Neural Networks: Tricks of the Trade. Springer.

---

## License

MIT — see [`LICENSE`](LICENSE).

---