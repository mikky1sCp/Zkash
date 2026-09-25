# Zkash-10K — Technical Whitepaper

**Report ID:** ZK-2025-01
**Version:** 1.0
**Status:** Reference / Educational
**Params:** exactly 10,000

---

## Abstract

**Zkash-10K** is a compact feed-forward neural network with **exactly 10,000 trainable parameters**, designed as an educational reference model. It is large enough to learn non-linear decision boundaries, yet small enough to train on a CPU in seconds and to reason about analytically. The parameter count is a hard design constraint, not an approximation.

---

## 1. Motivation

- Bridges the gap between toy models (~100 params) and production networks (>1M params).
- Round, reproducible parameter count enables fair comparison of optimizers, initializers, and regularization schemes.
- Small enough to inspect every weight; large enough to exhibit real generalization behavior.

---

## 2. Architecture

```
Input(64)
   │
   ▼
Linear(64 → 64) + ReLU        # 4,160 params
   │
   ▼
Linear(64 → 80) + ReLU        # 5,200 params
   │
   ▼
Linear(80 → 8, bias=False)    #   640 params
   │
   ▼
Logits(8) → softmax
```

### Parameter budget

| Layer | Weights | Biases | Total |
|---|---:|---:|---:|
| `fc1` Linear(64, 64) | 4,096 | 64 | **4,160** |
| `fc2` Linear(64, 80) | 5,120 | 80 | **5,200** |
| `fc3` Linear(80, 8, bias=False) | 640 | 0 | **640** |
| | | **Total** | **10,000** |

**Design notes**
- `fc3` has no bias: constant offsets are redundant under softmax.
- `fc2` is an expansion layer (64 → 80) to increase non-linear expressivity at low cost.
- Output size 8 covers most educational datasets.

---

## 3. Forward Pass

For input $x \in \mathbb{R}^{64}$:

$$
\begin{aligned}
h_1 &= \mathrm{ReLU}(W_1 x + b_1), \quad h_1 \in \mathbb{R}^{64} \\
h_2 &= \mathrm{ReLU}(W_2 h_1 + b_2), \quad h_2 \in \mathbb{R}^{80} \\
z   &= W_3 h_2, \quad z \in \mathbb{R}^{8} \\
\hat{y} &= \mathrm{softmax}(z)
\end{aligned}
$$

Loss (cross-entropy):

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log \hat{y}_{i,\,y_i}
$$

---

## 4. Reference Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Zkash10K(nn.Module):
    """Zkash-10K: exactly 10,000 trainable parameters."""
    def __init__(self, n_in=64, n_out=8):
        super().__init__()
        self.fc1 = nn.Linear(n_in, 64)                 # 4,160
        self.fc2 = nn.Linear(64, 80)                   # 5,200
        self.fc3 = nn.Linear(80, n_out, bias=False)    #   640

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # logits


# sanity check
model = Zkash10K()
assert sum(p.numel() for p in model.parameters()) == 10_000
```

---

## 5. Training Protocol

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (β = 0.9, 0.999) |
| Learning rate | 3·10⁻³ |
| Weight decay | 1·10⁻⁴ |
| Batch size | 64 |
| Epochs | 50–200 |
| Initialization | Kaiming (ReLU), Xavier (output) |
| Loss | Cross-entropy |

---

## 6. Characteristics

| Metric | Value |
|---|---|
| Trainable parameters | **10,000** |
| Forward MACs (per sample) | ≈ 20,000 |
| fp32 size | ≈ 40 KB |
| int8 size | 10 KB |
| Throughput (CPU, batch=64) | > 10,000 samples/s |
| Convergence | seconds to minutes |

---

## 7. Limitations

- Fixed to small output dimensionalities (≤ 8 classes by default).
- Accepts only vectorized inputs; no native sequence or image handling.
- Prone to overfitting when the dataset is small (< 5,000 samples).
- Not intended for production; educational use only.

---

## 8. Extensions

- **Zkash-10K-Residual** — add skip connections within the same 10,000-param budget.
- **Zkash-10K-Conv** — replace `fc1` with a 3×3 convolution for 8×8 inputs.
- **Zkash-10K-LoRA** — low-rank adaptation on top of a frozen base.
- **Zkash-10K-Quantized** — int8 deployment variant (10 KB).

---

## 9. Conclusion

Zkash-10K is a minimal yet expressive network with a **hard parameter budget of 10,000**. Its transparent architecture, symmetric two-hidden-layer design, and bias-free output layer make it an ideal testbed for studying optimization, initialization, and generalization at small scale.

> *Small enough to understand, big enough to learn.*

---

**Citation**

```
Zkash-10K: A Minimal 10,000-Parameter Neural Network for Education.
Technical Report ZK-2025-01, v1.0.
```
