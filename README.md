# Zkash

Minimal feed-forward neural networks with **exact parameter budgets**, built for education and reproducible scale studies.

> *Small enough to understand, large enough to misbehave.*

![version](https://img.shields.io/badge/version-2.0.0-blue)
![params](https://img.shields.io/badge/params-100k-green)
![python](https://img.shields.io/badge/python-%3E%3D3.10-blue)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Models

| Model | Version | Params | Layers | Hidden sizes | Report |
|---|---|---:|---:|---|---|
| **Zkash-10K** | 1.0.0 | 10,000 | 3 | 64 → 80 → 8 | ZK-2025-01 |
| **Zkash-0.1M** | 2.0.0 | 100,000 | 4 | 191 → 256 → 145 → 8 | ZK-2025-02 |

Both models take a 64-dimensional input and produce 8 logits. The parameter counts are **exact**, not rounded.

---

## Quickstart

```bash
git clone https://github.com/mikky1sCp/zkash.git
cd zkash

# (optional) virtual environment
python -m venv .venv
source .venv/Scripts/activate     # Windows / Git Bash
# source .venv/bin/activate       # Linux / macOS

pip install -r requirements.txt
pip install -e .

bash scripts/train.sh
bash scripts/eval.sh
pytest -q
```

### Expected output

```
[Zkash-0.1M] trainable params: 100000
epoch  10 | train loss 0.0000 | train acc 1.000
...
epoch  60 | train loss 0.0000 | train acc 1.000
val acc: 1.0000
saved checkpoint → checkpoints/zkash01m.pt

params : 100000
samples: 3276
acc    : 1.0000

4 passed
```

---

## Architecture — Zkash-0.1M

```
Input(64)
   │
   ▼
Linear(64  → 191) + ReLU     # 12,415
   │
   ▼
Linear(191 → 256) + ReLU     # 49,152
   │
   ▼
Linear(256 → 145) + ReLU     # 37,265
   │
   ▼
Linear(145 → 8)              #  1,168
   │
   ▼
Logits(8) → softmax
```

### Parameter budget

| Layer | Shape | Weights | Biases | Total |
|---|---|---:|---:|---:|
| `fc1` | Linear(64, 191) | 12,224 | 191 | **12,415** |
| `fc2` | Linear(191, 256) | 48,896 | 256 | **49,152** |
| `fc3` | Linear(256, 145) | 37,120 | 145 | **37,265** |
| `fc4` | Linear(145, 8) | 1,160 | 8 | **1,168** |
| | | | **Total** | **100,000** |

Full technical details in [`WHITEPAPER.md`](WHITEPAPER.md).

---

## Project structure

```
zkash/
├── README.md
├── WHITEPAPER.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── configs/
│   └── zkash_01m.yaml
├── docs/
│   └── whitepaper.md
├── src/
│   └── zkash/
│       ├── __init__.py
│       ├── model.py        # Zkash10K, Zkash01M, build_model
│       ├── data.py         # synthetic Gaussian clouds
│       ├── train.py        # training loop
│       ├── evaluate.py     # validation
│       └── utils.py        # seeds, config, param counter
├── scripts/
│   ├── train.sh
│   └── eval.sh
├── tests/
│   └── test_model.py
└── checkpoints/
    └── .gitkeep
```

---

## Usage

### Python API

```python
import torch
from zkash import Zkash01M, Zkash10K, count_params

model = Zkash01M()
print(count_params(model))          # 100000

x = torch.randn(4, 64)
logits = model(x)
print(logits.shape)                 # torch.Size([4, 8])
```

### Model factory

```python
from zkash import build_model

m1 = build_model("zkash_10k")
m2 = build_model("zkash_01m", p_drop=0.1)
```

### Training from CLI

```bash
PYTHONPATH=src python -m zkash.train --config configs/zkash_01m.yaml
PYTHONPATH=src python -m zkash.evaluate \
    --config configs/zkash_01m.yaml \
    --ckpt checkpoints/zkash01m.pt
```

### Configuration

All hyperparameters live in `configs/zkash_01m.yaml`:

```yaml
model:
  n_in: 64
  n_out: 8
  p_drop: 0.1

data:
  n_samples: 16384
  n_classes: 8
  noise: 0.7
  seed: 0

train:
  epochs: 60
  batch_size: 128
  lr: 1.0e-3
  weight_decay: 1.0e-4
  seed: 0

paths:
  checkpoint: checkpoints/zkash01m.pt
```

---

## Training protocol

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (β = 0.9, 0.999) |
| Learning rate | 1·10⁻³ |
| Weight decay | 1·10⁻⁴ |
| Batch size | 128 |
| Epochs | 60 |
| Init | Kaiming (ReLU), Xavier (output) |
| Loss | Cross-entropy |
| Split | 80 / 20 train / val |

---

## Dataset

The default dataset is **synthetic**: 8 Gaussian clouds in ℝ⁶⁴, centers drawn from 𝒩(0, 3²), samples corrupted with σ = 0.7 noise. See `src/zkash/data.py`.

To plug in your own data, return a `torch.utils.data.TensorDataset(X, y)` with `X` of shape `(N, 64)` and `y` in `[0, 8)`.

---

## Tests

```bash
pytest -q
```

Covers:

- exact parameter counts for both models (`10_000`, `100_000`)
- forward-pass output shapes
- (extend as needed)

---

## Roadmap

- [x] **v1.0.0** — Zkash-10K
- [x] **v2.0.0** — Zkash-0.1M
- [ ] **v2.1.0** — early stopping + best checkpoint by val_loss
- [ ] **v2.2.0** — cosine LR schedule with warmup, CSV logging
- [ ] **v2.3.0** — non-trivial datasets (XOR, overlapping clouds, low-N)
- [ ] **v3.0.0** — Zkash-1M (residual, 1,000,000 params exactly)

---

## Citation

```bibtex
@techreport{zkash01m,
  title  = {Zkash-0.1M: A 100,000-Parameter Reference Neural Network for Education},
  number = {ZK-2025-02},
  year   = {2025},
  note   = {Version 2.0.0}
}
```

---

## License

MIT — see `LICENSE`.

---

## Author

[@mikky1sCp](https://github.com/mikky1sCp)
