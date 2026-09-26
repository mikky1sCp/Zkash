# Zkash10M

**A 10-million-parameter residual network, built to be understood.**

*Depth over width. Norm before activation. Predictions before results.*

![version](https://img.shields.io/badge/series-v4.2.0-blue?style=flat-square)
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
- [Two regimes, one mechanism](#two-regimes-one-mechanism)
- [v4.2.0: a falsified prediction](#v420-a-falsified-prediction)
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
- **Fails in a reproducible, documentable way** — and publishes its falsified predictions

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

**Dropout adds zero parameters.** With `p_drop = 0.1` (v4.2.0), `count_params` still returns exactly 10,000,896.

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

Once we make the data difficult, the models diverge. Below are results for **Zkash10M v4.1.0** (early stopping, no regularization):

| Config | n_samples | label_noise | Train acc | Val acc | Gap | Best epoch |
|---|---:|---:|---:|---:|---:|---:|
| easy | 65,536 | 0.0 | 1.0000 | 1.0000 | 0.000 | 1 |
| hard | 2,048 | 0.0 | 0.9286 | **0.5428** | +0.386 | 4 |
| impossible | 262,144 | 0.2 | 0.7188 | **0.7128** | +0.006 | 5 |

And v4.2.0 added the missing piece — the **empirical Bayes ceiling** for `hard`:

| Config | Bayes ceiling | Best model | Gap to Bayes |
|---|---:|---:|---:|
| easy | 1.0000 | 1.0000 | 0.0000 |
| hard | **0.5879** | 0.5458 | **−0.0421** |
| impossible | 0.8250 | 0.7128 | **−0.1122** |

**Takeaway:** the train/val gap tells you almost nothing about which task has room to improve. `hard` has a 0.39 gap and is 4.2 points from Bayes. `impossible` has a 0.006 gap and is 11.2 points from Bayes. Only the ceiling measurement distinguishes them.

---

## Two regimes, one mechanism

v4.1.0 classified the two hard tasks as "opposite failures": `hard` overfits, `impossible` underfits. **v4.2.0 refutes that classification.** Both are memorization-dominated; they differ in distance to Bayes, not in regime.

### hard — memorization-dominated, ceiling-bound

| Epoch | Train acc | Val acc | Gap |
|---:|---:|---:|---:|
| 1 | 0.3807 | 0.4694 | −0.089 |
| 4 | 0.7407 | **0.5428** ★ | +0.198 |
| 5 | 0.8188 | 0.4988 | +0.320 |
| 10 | 0.9286 | 0.4743 | **+0.454** |

The gap looks like classical overfitting. It isn't. Two facts disqualify that diagnosis:

1. The val peak is **4.2 points below the true Bayes ceiling** (0.5428 vs 0.5879).
2. **Regularization makes val worse, not better** (−0.0116 across two dataset sizes).

The model learns the class signal in the first ~64–128 gradient steps, then spends the rest fitting training noise. Peak val is pinned near the empirical ceiling for whatever sample size is available.

**Correct remedy:** none available within the architecture. `hard` is closed.
**Wrong remedy (now empirically confirmed):** dropout, weight decay, more data, longer training.

### impossible — optimization-bound

| Epoch | Train acc | Val acc | Gap |
|---:|---:|---:|---:|
| 1 | 0.6944 | 0.7069 | −0.013 |
| 5 | 0.7089 | **0.7128** ★ | −0.004 |
| 10 | 0.7135 | 0.7099 | +0.004 |
| 15 | 0.7188 | 0.7093 | +0.010 |

Train and validation stay together at ~0.71. The gap never exceeds 0.01. **Diagnosis:** 20% label noise makes the signal weak; the model is learning slowly. It needs more iterations at a smaller learning rate.

**Supporting evidence.** Bayes loss for `label_noise = 0.2` and 8 classes: **0.888** (per §5.1 formula; ~0.803 with corrected `data.py` semantics). Observed train loss at epoch 15: **1.030**. Margin above Bayes: 0.14–0.23 nats. The model is not far from the loss floor. It is simply **not yet converged**.

**Correct remedy:** cosine LR schedule, warmup, longer patience.
**Wrong remedy:** dropout — it would make an already-slow model slower.

### Summary

| Task | Gap to Bayes | Peak epoch | Behavior | Fixable? |
|---|---:|---:|---|---|
| `hard` | −0.042 | 2–4 | Ceiling-bound memorization | **No** — gap is task-limited |
| `impossible` | −0.112 | 5 | Optimization-bound memorization | **Maybe** — v4.3.0 tests it |

**This is the central lesson of v4.2.0.** A train/val gap cannot tell you whether regularization will help. A 0.39 gap at 0.54 val and a 0.006 gap at 0.71 val can both be memorization-dominated. Only a **Bayes measurement** tells you which one has room to improve.

---

## v4.2.0: a falsified prediction

WHITEPAPER v4.1.0 §9.1 predicted that dropout + weight decay would raise `hard` val accuracy to **0.60–0.65**. This release tests that prediction directly and reports the result side-by-side.

### Setup

- `p_drop = 0.1` inside each residual block (applied to the hidden activation, not the identity path)
- `weight_decay = 1.0e-2` (10× v4.1.0's value), applied via **two parameter groups**: 2-D weight matrices decayed, all 1-D parameters (RMSNorm γ, 21 tensors, 10,752 params) excluded. **Parameter count unchanged: 10,000,896.**
- `patience = 25`, `epochs = 100`
- 2×2 ablation: `{dropout, wd} ∈ {off, on} × n_samples ∈ {2,048, 8,192}`

### Results

| n_samples | regularization | Train acc @ best | Val acc @ best | Best epoch |
|---:|---|---:|---:|---:|
| 2,048 | none (v4.1.0) | 0.7407 | **0.5428** | 4 |
| 2,048 | dropout + wd 1e-2 | 0.7474 | **0.5306** | 4 |
| 8,192 | none | 0.5418 | **0.5458** | 2 |
| 8,192 | dropout + wd 1e-2 | 0.5383 | **0.5348** | 2 |
| — | **true Bayes** | — | **0.5879** | — |

| Effect | 2,048 | 8,192 | Mean |
|---|---:|---:|---:|
| **Cost of regularization** | −0.0122 | −0.0110 | **−0.0116** |
| **Gain from 4× data** | +0.0030 | +0.0042 | **+0.0036** |

| Metric | §9.1 predicted | v4.2.0 observed | Status |
|---|---:|---:|---|
| Train acc | ~0.75 | 0.7474 | ✓ |
| Val acc | 0.60–0.65 | **0.5306** | ✗ |
| Gap | ~0.10 | 0.2168 | ✗ |

**The prediction is falsified.** Regularization cost 1.2 points. 4× more data gained 0.4 points. Both are consistent across the 2×2. The only mechanism that reliably helps is saving the best epoch — the v4.1.0 result.

### Why both measures fail

Both dropout and more data leave the **memorization transition point** unchanged:

- The `hard` runs peak at **64 and 128 gradient steps** (2,048 and 8,192 samples) — same order of magnitude. That transition is set by the task's signal-to-noise ratio, not by capacity.
- 4× data → 4× steps/epoch → the peak just arrives 4× earlier. Val at the peak is unchanged.

**Neither capacity reduction nor dataset expansion can move a transition that is task-determined.** Only the Bayes measurement reveals this.

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

### Empirical Bayes for hard

```bash
python scripts/bayes_hard.py
# Bayes (nearest-centroid): 0.5879
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

All knobs live in `configs/*.yaml`. Four configs ship by default.

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

### `configs/zkash_10m_impossible.yaml` — optimization-bound

```yaml
data:
  n_samples: 262144
  noise: 1.5
  center_scale: 0.5
  label_noise: 0.2

train:
  batch_size: 256
```

---

## Training protocol

| Hyperparameter | v4.1.0 | v4.2.0 |
|---|---|---|
| Optimizer | AdamW (β = 0.9, 0.999) | same |
| Learning rate | 1·10⁻³ (constant) | same |
| Weight decay | 1·10⁻⁴ | **1·10⁻² (hard only)** |
| Weight decay groups | — | **matrices / 1-D split** |
| Batch size | 128 (256 for impossible) | same |
| Epochs | 60 max, patience 10 | **100 max, patience 25** |
| Loss | Cross-entropy | same |
| Initialization | Default PyTorch (uniform) | same |
| Warmup | **none** | none |
| Gradient clipping | **none** | none |
| Dropout | **none** | **0.1 (hard only)** |
| Early stopping | **yes** (patience 10 on `val_acc`) | **yes** (patience 25) |

**Why no warmup and no clipping?** Pre-norm residual networks train cleanly from epoch 1. This is a structural property — not a missing feature.

**Why early stopping?** v4.1.0 added it and recovered +10 percentage points on both hard tasks, for free. **It remains the only mechanism in this project that reliably improves val acc.**

**Why the parameter groups in v4.2.0?** RMSNorm γ must not be decayed — shrinking it fights pre-norm's own stabilization. The optimizer gets two groups; the model's parameter count is unchanged.

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
│   └── zkash_10m_impossible.yaml
├── src/
│   └── zkash/
│       ├── __init__.py
│       ├── model.py        # RMSNorm, ResidualBlock, Zkash10M — one file
│       ├── data.py         # synthetic Gaussian clouds
│       ├── train.py        # training loop, early stopping, param groups
│       ├── evaluate.py     # validation with checkpoint metadata
│       └── utils.py        # seeds, device, config
├── scripts/
│   ├── train.sh
│   ├── train_hard.sh
│   ├── train_impossible.sh
│   ├── sweep_hard.sh       # v4.2.0 2×2 ablation
│   ├── bayes_hard.py       # empirical Bayes for hard
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
| **v4.2.0** | ✅ | **dropout + wd, hard** | **−0.011 on hard (falsified §9.1)** |
| v4.3.0 | 🔜 | cosine LR + warmup + patience 25 (**impossible**) | target: close the 11.2-pt gap to Bayes |
| v4.3.0 | 🔜 | label smoothing 0.1 (**impossible**) | secondary |
| v4.4.0 | 📅 | Multi-seed runs (5 seeds) — mean ± std | quantifies variance floor |
| v5.0.0 | 📅 | Zkash100M — depth 100, exact 100M params | next scale-up |

### v4.3.0 predictions (falsifiable)

**impossible (optimization-bound regime):**

| Metric | v4.2.0 | Predicted v4.3.0 |
|---|---:|---:|
| Train acc | 0.7188 | ~0.78 |
| Val acc | **0.7128** | **0.76–0.80** |
| Gap | +0.006 | ~0.01 |
| Train loss | 1.030 | ~0.92 |

Cosine LR with warmup lets the model finish convergence. Target train loss approaches Bayes.

**This prediction is on the record. v4.3.0 will report actual numbers side-by-side, whichever way they fall.**

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
Because on those tasks, the model is close to the **Bayes ceiling** — and the ceiling is low. For `hard`, the empirical Bayes is 0.5879; the best model run reaches 0.5458. There is nothing left to learn from this data.

**Why did dropout + weight decay make `hard` *worse*?**
Because `hard` is not overfitting — it is memorization-dominated and ceiling-bound. Regularization suppresses the (already-correct) signal fit along with the noise fit. Net effect: −0.0116 across two dataset sizes. See [v4.2.0: a falsified prediction](#v420-a-falsified-prediction).

**Why didn't 4× more data help `hard`?**
Because the memorization transition happens at a fixed number of gradient steps (64–128), which is set by the task's signal-to-noise ratio, not by dataset size. More data makes the peak arrive earlier; it doesn't move the peak.

**Why does the weight decay use parameter groups?**
RMSNorm γ should not be decayed — shrinking it fights pre-norm's own stabilization. The 21 norm tensors (10,752 params) are excluded; the 42 weight matrices are decayed. Parameter count is unchanged.

**Why does early stopping help so much?**
Because we save the **best** epoch, not the **last**. On `hard`, val peaks at 0.5428 on epoch 4 and drops to 0.4743 by epoch 10 — but v1.0.0 kept saving the last. Early stopping recovers the peak. It remains the only mechanism in this project that reliably improves val acc.

**Is this model production-ready?**
No. It's an educational reference. It has documented failure modes on non-trivial tasks, and its most recent experiment produced a negative result. See the roadmap for v4.3.0.

---

## Citation

```bibtex
@techreport{zkash10m,
  title       = {Zkash10M: A 10M-Parameter Residual Reference Network for Education},
  number      = {ZK-2025-04},
  institution = {Zkash Project},
  year        = {2025},
  note        = {Version 4.2.0}
}
```

**References**

- He, K., Zhang, X., Ren, S., Sun, J. (2015). *Deep Residual Learning for Image Recognition.* arXiv:1512.03385.
- Zhang, B., Sennrich, R. (2019). *Root Mean Square Layer Normalization.* arXiv:1910.07467.
- Hendrycks, D., Gimpel, K. (2016). *Gaussian Error Linear Units (GELUs).* arXiv:1606.08415.
- Veit, A., Wilber, M., Belongie, S. (2016). *Residual Networks Behave Like Ensembles of Relatively Shallow Networks.* arXiv:1605.06431.
- Loshchilov, I., Hutter, F. (2019). *Decoupled Weight Decay Regularization.* arXiv:1711.05101.
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting.* JMLR 15(1).

---

## License

MIT — see [`LICENSE`](LICENSE).

---