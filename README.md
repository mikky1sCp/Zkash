```markdown
# Zkash

Minimal feed-forward neural networks with **exact parameter budgets**, built for education and reproducible scale studies.

> *Small enough to understand, large enough to misbehave.*

![version](https://img.shields.io/badge/version-3.0.0-blue)
![params](https://img.shields.io/badge/params-1M-green)
![python](https://img.shields.io/badge/python-%3E%3D3.10-blue)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Models

| Model | Version | Params | Layers | Hidden sizes | Report |
|---|---|---:|---:|---|---|
| **Zkash-10K** | 1.0.0 | 10,000 | 3 | 64 → 80 → 8 | ZK-2025-01 |
| **Zkash-0.1M** | 2.0.0 | 100,000 | 4 | 191 → 256 → 145 → 8 | ZK-2025-02 |
| **Zkash-1M** | 3.0.0 | 1,000,000 | 4 | 508 → 640 → 988 → 8 | ZK-2025-03 |

All models take a 64-dimensional input and produce 8 logits. The parameter counts are **exact**, not rounded.

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
device: cuda
[Zkash-1M] trainable params: 1000000
epoch  10 | train loss 0.0000 | train acc 1.000
...
epoch 100 | train loss 0.0000 | train acc 1.000
val acc: 1.0000
saved checkpoint → checkpoints/zkash1m.pt

device: cuda
params : 1000000
samples: 6553
acc    : 1.0000

26 passed
```

---

## Architecture — Zkash-1M

```
Input(64)
   │
   ▼
Linear(64  → 508) + ReLU     #  33,020
   │
   ▼
Linear(508 → 640) + ReLU     # 325,760
   │
   ▼
Linear(640 → 988) + ReLU     # 633,308
   │
   ▼
Linear(988 → 8)              #   7,912
   │
   ▼
Logits(8) → softmax
```

### Parameter budget

| Layer | Shape | Weights | Biases | Total |
|---|---|---:|---:|---:|
| `fc1` | Linear(64, 508) | 32,512 | 508 | **33,020** |
| `fc2` | Linear(508, 640) | 325,120 | 640 | **325,760** |
| `fc3` | Linear(640, 988) | 632,320 | 988 | **633,308** |
| `fc4` | Linear(988, 8) | 7,904 | 8 | **7,912** |
| | | | **Total** | **1,000,000** |

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
│   ├── zkash_10k.yaml
│   ├── zkash_01m.yaml
│   └── zkash_1m.yaml
├── src/
│   └── zkash/
│       ├── __init__.py
│       ├── model.py        # Zkash10K, Zkash01M, Zkash1M, build_model
│       ├── data.py         # synthetic Gaussian clouds
│       ├── train.py        # training loop (device-aware)
│       ├── evaluate.py     # validation (device-aware)
│       └── utils.py        # seeds, config, param counter, device
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
from zkash import Zkash1M, count_params

model = Zkash1M()
print(count_params(model))          # 1000000

x = torch.randn(4, 64)
logits = model(x)
print(logits.shape)                 # torch.Size([4, 8])
```

### Model factory

```python
from zkash import build_model

m1 = build_model("zkash_10k")
m2 = build_model("zkash_01m", p_drop=0.1)
m3 = build_model("zkash_1m",  p_drop=0.2)
```

### Training from CLI

```bash
PYTHONPATH=src python -m zkash.train --config configs/zkash_1m.yaml
PYTHONPATH=src python -m zkash.evaluate \
    --config configs/zkash_1m.yaml \
    --ckpt checkpoints/zkash1m.pt
```

### Configuration

All hyperparameters live in `configs/zkash_1m.yaml`:

```yaml
model:
  name: zkash_1m
  n_in: 64
  n_out: 8
  p_drop: 0.2

data:
  n_samples: 32768
  n_classes: 8
  noise: 0.7
  seed: 0

train:
  device: auto      # auto | cpu | cuda | cuda:0
  epochs: 100
  batch_size: 256
  lr: 3.0e-4
  weight_decay: 1.0e-3
  seed: 0

paths:
  checkpoint: checkpoints/zkash1m.pt
```

---

## Training protocol — Zkash-1M

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (β = 0.9, 0.999) |
| Learning rate | 3·10⁻⁴ |
| Weight decay | 1·10⁻³ |
| Batch size | 256 |
| Epochs | 100 |
| Init | Kaiming (ReLU), Xavier (output) |
| Loss | Cross-entropy |
| Dropout | p = 0.2 on `fc2` and `fc3` |
| Split | 80 / 20 train / val |
| Device | auto (cuda if available) |

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

- exact parameter counts for all three models (`10_000`, `100_000`, `1_000_000`)
- forward-pass output shapes, batch sizes, dtype, finiteness
- dropout behavior in train/eval modes
- backward pass reaching every parameter
- state_dict round-trip
- model factory (valid names, unknown raises, kwargs forwarding)

---

## Roadmap

- [x] **v1.0.0** — Zkash-10K
- [x] **v2.0.0** — Zkash-0.1M
- [x] **v3.0.0** — Zkash-1M (device-aware training)
- [ ] **v3.1.0** — early stopping + best checkpoint by val_loss
- [ ] **v3.2.0** — cosine LR schedule with warmup, CSV logging
- [ ] **v3.3.0** — non-trivial datasets (XOR, overlapping clouds, low-N)
- [ ] **v4.0.0** — Zkash-10M

---

## Citation

```bibtex
@techreport{zkash1m,
  title  = {Zkash-1M: A 1,000,000-Parameter Reference Neural Network for Education},
  number = {ZK-2025-03},
  year   = {2025},
  note   = {Version 3.0.0}
}
```

---

## License

MIT — see `LICENSE`.

---

## Author

[@mikky1sCp](https://github.com/mikky1sCp)
```
