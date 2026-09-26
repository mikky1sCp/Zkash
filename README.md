# Zkash10M

**A 10-million-parameter residual network, built to be understood.**

*Depth over width. Norm before activation.*

![version](https://img.shields.io/badge/version-1.0.0-blue?style=flat-square)
![params](https://img.shields.io/badge/params-10%2C000%2C896-green?style=flat-square)
![python](https://img.shields.io/badge/python-%3E%3D3.10-blue?style=flat-square)
![pytorch](https://img.shields.io/badge/pytorch-%3E%3D2.2-orange?style=flat-square)
![license](https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square)

</div>

---

## ⚠️ Known limitation — negative result

Out of the box — **no dropout, no early stopping, no LR schedule** — Zkash10M overfits the synthetic task badly. On 262,144 samples with 20% label noise:

| Metric | Value |
|---|---:|
| Train accuracy | **0.971** |
| Val accuracy | **0.613** |
| Bayes optimal (analytical, for `label_noise=0.2`) | 0.825 |
| **Generalization gap** | **0.358** |

Train accuracy exceeds the Bayes ceiling (0.971 > 0.825) — mathematically only possible if the model **memorized flipped labels**. Val accuracy falls below it (0.613 < 0.825) — the model learned the *train* noise, which does not transfer.

**This is a deliberate, documented result.** It demonstrates that residual connections, RMSNorm, and GELU do **not**, by themselves, prevent overfitting at this parameter-to-sample ratio (~38 params/sample). Regularization is not optional.

Reproduce with:

```bash
PYTHONWARNINGS="ignore::FutureWarning" PYTHONPATH=src \
  python -m zkash.train --config configs/zkash_10m_impossible.yaml
```

Future work (v1.1.0) will add early stopping, dropout, and a weight-decay sweep — but the current release intentionally ships **without** them so the failure mode is inspectable.

---

## Why this model exists

Most educational neural networks are either **too small to be interesting** (XOR MLPs, 3-layer toys) or **too large to reason about** (production transformers). Zkash10M sits in the sweet spot:

- **10 million parameters** — large enough for depth, residual connections, and normalization to actually matter
- **41 linear layers** — deep enough to exhibit real gradient flow behavior
- **One file, no dependencies** — `model.py` is under 80 lines of readable code
- **Trains in 60 seconds on a GTX 1660** — you can iterate on ideas without a cluster
- **Fails in a reproducible, documentable way** — a rare property in educational code

It is the natural next step after you've written your first MLP from scratch and want to understand *why* modern architectures look the way they do.

---

## The scaling lineage

Zkash is a family of models with **exact parameter budgets**. Each generation introduced one new architectural idea. Zkash10M is the fourth.

| Generation | Model | Params | Layers | Key idea | Report |
|---|---|---:|---:|---|---|
| v1 | **Zkash-10K** | 10,000 | 3 | baseline MLP | ZK-2025-01 |
| v2 | **Zkash-0.1M** | 100,000 | 4 | depth + dropout | ZK-2025-02 |
| v3 | **Zkash-1M** | 1,000,000 | 4 | wide MLP | ZK-2025-03 |
| **v4** | **Zkash10M** | **10,000,896** | **41** | **residual + RMSNorm + GELU** | **ZK-2025-04** |

Each model kept the same interface: **64-dim input → 8-class output**. So they're directly comparable on the same task.

### Observed behavior across the series

| Model | Task | Train acc | Val acc | Gap |
|---|---|---:|---:|---:|
| Zkash-10K | easy (n=65k, no noise) | 1.000 | 1.000 | 0.00 |
| Zkash-1M | easy (n=65k, no noise) | 1.000 | 1.000 | 0.00 |
| Zkash10M | easy (n=65k, no noise) | 1.000 | 1.000 | 0.00 |
| Zkash10M | hard (n=2k, scale=0.3, noise=1.5) | **0.984** | **0.440** | **0.544** |
| Zkash10M | impossible (n=262k, label_noise=0.2) | **0.971** | **0.613** | **0.358** |

**Reading:** as parameters grow past the point where the task becomes separable, the model stops generalizing. Adding label noise delays but does not prevent the failure.

---

## What changed from v3 to v4

The jump from Zkash-1M to Zkash10M is not just "more parameters". It is a **change of paradigm**.

### Zkash-1M (wide MLP)

```
Input(64) → Linear(508) → ReLU → Linear(640) → ReLU → Linear(988) → ReLU → Linear(8)
```

Four layers. Almost a million parameters. **All capacity is in width.**

### Zkash10M (deep ResNet)

```
Input(64) → Stem(512)
   ↓
20 × [ RMSNorm → Linear(486) → GELU → Linear(512) → +residual ]
   ↓
RMSNorm → Head(8)
```

Twenty identical blocks. **All capacity is in depth.**

Three ideas do the heavy lifting:

| Idea | What it does | Why it matters |
|---|---|---|
| **Residual connection** | `x → x + f(x)` | Gradients flow directly to early layers; training 20+ layers becomes trivial |
| **Pre-norm (RMSNorm)** | Normalize *before* each block | Keeps activation scale stable; no warmup needed |
| **GELU** | Smooth activation | Better gradient signal than ReLU; standard in modern models |

### Side-by-side

| Property | Zkash-1M | Zkash10M |
|---|---:|---:|
| Trainable parameters | 1,000,000 | 10,000,896 |
| Linear layers | 4 | **41** |
| Residual connections | 0 | **20** |
| Normalization layers | 0 | **21 (RMSNorm)** |
| Activation | ReLU | **GELU** |
| Bias terms | yes | **none** |
| Effective depth | 4 | **~41** |
| Forward MACs / sample | ~1M | ~10.2M |
| fp32 size | 4 MB | **40 MB** |
| Time per epoch (GTX 1660, batch=128) | ~2 s | ~10 s |

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
       Logits(8)
```

### Parameter budget

| Component | Shape | Params | % of total |
|---|---|---:|---:|
| `stem` | Linear(64, 512), no bias | 32,768 | 0.33% |
| `block` × 20 | RMSNorm + Linear + GELU + Linear | 498,176 × 20 | 99.63% |
| `final_norm` | RMSNorm(512) | 512 | 0.005% |
| `head` | Linear(512, 8), no bias | 4,096 | 0.04% |
| | | **10,000,896** | **100%** |

---

## The three ingredients, explained

### 1. Residual connections — how deep networks train at all

A residual block computes `y = x + f(x)`. Backprop through this gives:

$$\frac{\partial y}{\partial x} = I + \frac{\partial f}{\partial x}$$

The identity matrix `I` is a **direct highway** for gradients.

### 2. RMSNorm — cheap, effective normalization

$$\mathrm{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\|x\|_2^2 + \epsilon}}$$

One parameter per dimension (`γ`), no bias. Pre-norm placement.

### 3. GELU — a smooth ReLU

$$\mathrm{GELU}(x) = x \cdot \Phi(x)$$

Smooth everywhere, better gradient signal than ReLU.

---

## Quickstart

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

python -c "
from zkash import Zkash10M, count_params
n = count_params(Zkash10M())
print('params:', n)
assert n == 10_000_896
"

bash scripts/train.sh
bash scripts/eval.sh
pytest -q
```

### Easy config output

```
device: cuda
[Zkash10M] trainable params: 10000896
epoch  60 | train loss 0.0000 | train acc 1.000
val acc: 1.0000
saved checkpoint → checkpoints/zkash10m.pt
```

### Impossible config output (documented failure)

```
device: cuda
[Zkash10M] trainable params: 10000896
epoch  10 | train loss 1.0462 | train acc 0.714
epoch  20 | train loss 1.0077 | train acc 0.726
epoch  30 | train loss 0.7686 | train acc 0.787
epoch  40 | train loss 0.3595 | train acc 0.894
epoch  50 | train loss 0.1601 | train acc 0.948
epoch  60 | train loss 0.0850 | train acc 0.971
val acc: 0.6134
```

The gap is the point.

---

## Configuration

### `configs/zkash_10m.yaml` — easy (separates cleanly)

```yaml
model:
  n_in: 64
  n_out: 8
  dim: 512
  hidden: 486
  depth: 20

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
  batch_size: 128
  lr: 1.0e-3
  weight_decay: 1.0e-4
  seed: 0

paths:
  checkpoint: checkpoints/zkash10m.pt
```

### `configs/zkash_10m_hard.yaml` — overfits hard

```yaml
data:
  n_samples: 2048
  noise: 1.5
  center_scale: 0.3
  label_noise: 0.0
```

### `configs/zkash_10m_impossible.yaml` — overfits despite label noise

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

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (β = 0.9, 0.999) |
| Learning rate | 1·10⁻³ |
| Weight decay | 1·10⁻⁴ |
| Batch size | 128 (or 256 for impossible config) |
| Epochs | 60 |
| Loss | Cross-entropy |
| Warmup | **none** |
| Gradient clipping | **none** |
| Dropout | **none** |
| Early stopping | **none** |

The absence of warmup and clipping is by design — residual + pre-norm train cleanly. The absence of dropout and early stopping is the documented limitation above.

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
│   └── zkash_10m_impossible.yaml
├── src/
│   └── zkash/
│       ├── __init__.py
│       ├── model.py
│       ├── data.py
│       ├── train.py
│       ├── evaluate.py
│       └── utils.py
├── scripts/
│   ├── train.sh
│   ├── train_hard.sh
│   ├── train_impossible.sh
│   └── eval.sh
├── tests/
│   └── test_model.py
└── checkpoints/
    └── .gitkeep
```

---

## Python API

```python
import torch
from zkash import Zkash10M, count_params

model = Zkash10M()
print(count_params(model))                # 10000896

x = torch.randn(4, 64)
logits = model(x)
print(logits.shape)                       # torch.Size([4, 8])
```

### Inspecting a block

```python
from zkash import Zkash10M, ResidualBlock

m = Zkash10M()
print(len(m.blocks))                      # 20
print(m.blocks[0].fc1.weight.shape)       # torch.Size([486, 512])
print(m.blocks[0].fc2.weight.shape)       # torch.Size([512, 486])
```

### Verifying the residual identity

```python
from zkash import ResidualBlock

blk = ResidualBlock(64, 64)
with torch.no_grad():
    blk.fc1.weight.zero_()
    blk.fc2.weight.zero_()

x = torch.randn(4, 64)
assert torch.allclose(blk(x), x)          # ✓
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
| `test_forward_shape` / `batch_sizes` / `dtype` / `finite` | forward-pass sanity |
| `test_backward_all_params_get_grad` | no dead parameters |
| `test_state_dict_roundtrip` | serialization correctness |

---

## Common questions

**Why 512 → 486 → 512 inside each block?**
The bottleneck (`hidden < dim`) forces each block to compress and re-expand. Also keeps each block ~5% of the budget.

**Why no bias in any Linear layer?**
After RMSNorm, bias is redundant — normalization already re-centers activations.

**Why 20 blocks and not 4 wide ones?**
Depth is the entire point of v4. Four wide blocks would just be a bigger MLP with the same gradient-flow problems.

**Why not use LayerNorm?**
RMSNorm does the same job with half the operations and one parameter per dimension.

**Why does it train without warmup?**
Pre-norm keeps activation scale bounded at every layer, so gradients never spike.

**Why is val accuracy so low in the impossible config?**
Because of overfitting. See the [Known limitation](#️-known-limitation--negative-result) section at the top. This is the intended behavior for v1.0.0.

---

## Roadmap

- [x] **v1.0.0** — Zkash10M baseline, documented overfitting
- [ ] **v1.1.0** — early stopping + best checkpoint by val_acc
- [ ] **v1.2.0** — dropout p=0.1 + weight_decay=1e-2
- [ ] **v1.3.0** — regularization ablation table (dropout / wd / both)
- [ ] **v1.4.0** — cosine LR schedule with warmup
- [ ] **v2.0.0** — Zkash100M

---

## Citation

```bibtex
@techreport{zkash10m,
  title       = {Zkash10M: A 10M-Parameter Residual Reference Network for Education},
  number      = {ZK-2025-04},
  institution = {Zkash Project},
  year        = {2025},
  note        = {Version 1.0.0}
}
```

---

## License

MIT — see [`LICENSE`](LICENSE).

---