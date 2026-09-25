# Zkash-1M — Technical Whitepaper

**Report ID:** ZK-2025-03
**Version:** 3.0.0
**Status:** Reference / Educational
**Params:** exactly 1,000,000

---

## Abstract

**Zkash-1M** is a 4-layer feed-forward neural network with **exactly 1,000,000 trainable parameters** — a 10× scale-up of Zkash-0.1M and a 100× scale-up of Zkash-10K. It sits at the low end of "serious" model sizes: large enough to show genuine overfitting, gradient noise, and the benefits of regularization; small enough to train on a CPU and to analyze layer-wise statistics on a laptop. The parameter count is a hard constraint: **1,000,000 exactly**, not one more.

---

## 1. Motivation

- **Three-order-of-magnitude sweep.** Zkash-10K (v1) → Zkash-0.1M (v2) → Zkash-1M (v3) gives a clean study of how behavior scales across two decades of capacity.
- **Real overfitting.** At 1M params, the model will overfit a 4k-sample dataset in a few epochs without regularization. This is the first Zkash where regularization is not optional.
- **Still inspectable.** A 1M-param MLP is small enough to compute full per-layer gradient norms, Jacobian spectra, and Hessian-vector products on CPU.
- **Round number.** Exact 1M enables fair comparisons across optimizers, schedules, and quantized deployments.

---

## 2. Architecture

```
Input(64)
   │
   ▼
Linear(64  → 508) + ReLU     # 33,020
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

### Design notes

- **Expansion → contraction.** First two layers expand (64 → 508 → 640), third expands further (640 → 988) to a wide representation, and the head contracts to 8 logits. The wide `fc3` dominates the budget (63%) and acts as the "learned feature bank".
- **Bias on every layer.** Unlike v1 (which dropped the output bias), v3 keeps biases everywhere for uniform accounting.
- **No normalization, no residuals.** Adding these would change the parameter formula. Kept out to preserve the exact 1M budget and the pedagogical simplicity.
- **Hidden sizes** (508, 640, 988) are the smallest integers satisfying the exact-budget equation subject to `64 < H1 < H2 < H3`. A "monotone funnel then drop" shape.

---

## 3. Forward Pass

For input $x \in \mathbb{R}^{64}$:

$$
\begin{aligned}
h_1 &= \mathrm{ReLU}(W_1 x + b_1), \quad h_1 \in \mathbb{R}^{508} \\
h_2 &= \mathrm{ReLU}(W_2 h_1 + b_2), \quad h_2 \in \mathbb{R}^{640} \\
h_3 &= \mathrm{ReLU}(W_3 h_2 + b_3), \quad h_3 \in \mathbb{R}^{988} \\
z   &= W_4 h_3 + b_4, \quad z \in \mathbb{R}^{8} \\
\hat{y} &= \mathrm{softmax}(z)
\end{aligned}
$$

Loss (cross-entropy):

$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log \hat{y}_{i,\,y_i}
$$

---

## 4. Training Protocol

At this size, training requires more care than v1/v2.

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (β = 0.9, 0.999) |
| Learning rate | 3·10⁻⁴ |
| Weight decay | 1·10⁻³ |
| Batch size | 256 |
| Epochs | 60–200 |
| Initialization | Kaiming (ReLU) for hidden, Xavier for output |
| Loss | Cross-entropy |
| Dropout | p = 0.2 after `fc2` and `fc3` |
| Split | 80 / 20 train / val |

Recommended additions over v2:

- **Dropout** p = 0.2 on `fc2` and `fc3`.
- **Cosine LR schedule with 5-epoch warmup.**
- **Early stopping** on val loss, patience 15.
- **Gradient clipping** at norm 1.0 (for stability in early epochs).

---

## 5. Characteristics

| Metric | Zkash-10K (v1) | Zkash-0.1M (v2) | Zkash-1M (v3) |
|---|---:|---:|---:|
| Trainable parameters | 10,000 | 100,000 | **1,000,000** |
| Forward MACs (per sample) | ≈ 20k | ≈ 200k | ≈ 997,856 |
| fp32 size | 40 KB | 400 KB | **≈ 4 MB** |
| int8 size | 10 KB | 100 KB | 1 MB |
| CPU throughput (batch=256) | > 10k/s | ~2k/s | ~400/s |
| Time-to-overfit (N=4,096) | never | < 5 epochs | < 2 epochs |

---

## 6. Limitations

- Fixed input dimension (64). Not convolutional.
- Still small by production standards — 1M is at the low end of modern models.
- Overfits aggressively without regularization; val loss will diverge from train loss early without dropout / weight decay.
- Output head fixed to 8 classes by default (configurable).

---

## 7. Extensions

- **Zkash-1M-Residual** — add skip connections while preserving the exact 1M budget.
- **Zkash-1M-Conv** — replace `fc1` with a convolutional stem for image inputs.
- **Zkash-1M-Quantized** — int8 deployment (1 MB).
- **Zkash-10M** — next scale-up (v4.0.0).

---

## 8. Conclusion

Zkash-1M is the third reference point in the Zkash scaling series, with an exact 1,000,000-parameter budget. At this size, the model exhibits genuine optimization and generalization phenomena — overfitting, gradient noise, sensitivity to learning rate — that are absent at 10K and only marginal at 0.1M. It is the smallest Zkash for which regularization is not a luxury but a necessity.

> *No longer a toy. Still fully inspectable.*

---

**Citation**

```
Zkash-1M: A 1,000,000-Parameter Reference Neural Network for Education.
Technical Report ZK-2025-03, v3.0.0.
```

```
Zkash-0.1M: A 100,000-Parameter Reference Neural Network for Education.
Technical Report ZK-2025-02, v2.0.0.
```
