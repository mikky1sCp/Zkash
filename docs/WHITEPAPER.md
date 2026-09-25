# Zkash-0.1M — Technical Whitepaper

**Report ID:** ZK-2025-02
**Version:** 2.0.0
**Status:** Reference / Educational
**Params:** exactly 100,000

---

## Abstract

**Zkash-0.1M** is a 4-layer feed-forward neural network with **exactly 100,000 trainable parameters** — a 10× scale-up of the Zkash-10K reference model. It is large enough to exhibit real overfitting, gradient pathologies, and regularization effects, yet small enough to train on a CPU in seconds and to reason about analytically. The parameter count is a hard constraint, not a round-off.

---

## 1. Motivation

- **Scale study.** Direct successor to Zkash-10K (v1.0.0, 10,000 params). Same input/output interface, 10× capacity.
- **Real phenomena.** Unlike the 10K model, Zkash-0.1M can memorize small datasets, exhibit double-descent, and respond measurably to weight decay, dropout, and early stopping.
- **Still inspectable.** 100,000 parameters is small enough to analyze per-layer spectra, gradient norms, and Hessian approximations on a laptop.
- **Round number.** An exact budget enables apples-to-apples comparison across optimizers, schedules, and regularization schemes.

---

## 2. Architecture

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

### Design notes

- **Two expansion stages, one contraction.** `fc1` expands 64 → 191, `fc2` expands 191 → 256, then `fc3` contracts back to 145, and `fc4` maps to 8 logits. This hourglass shape is deliberate: it forces the network to build a compact class-discriminative representation.
- **Bias on every layer.** The softmax-redundancy argument (used in v1.0.0) is sacrificed for architectural uniformity and exact parameter accounting.
- **No normalization layers.** BatchNorm/LayerNorm would add parameters and change the budget. Kept out by design.
- **Hidden sizes** (191, 256, 145) are the smallest integers that hit exactly 100,000 with a 4-layer all-bias design, subject to `64 · H1 > H1 · H2 > H2 · H3 > H3 · 8` (a rough "smooth funnel" preference).

---

## 3. Forward Pass

For input $x \in \mathbb{R}^{64}$:

$$
\begin{aligned}
h_1 &= \mathrm{ReLU}(W_1 x + b_1), \quad h_1 \in \mathbb{R}^{191} \\
h_2 &= \mathrm{ReLU}(W_2 h_1 + b_2), \quad h_2 \in \mathbb{R}^{256} \\
h_3 &= \mathrm{ReLU}(W_3 h_2 + b_3), \quad h_3 \in \mathbb{R}^{145} \\
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

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (β = 0.9, 0.999) |
| Learning rate | 1·10⁻³ (v1.0.0 used 3·10⁻³; larger model → smaller LR) |
| Weight decay | 1·10⁻⁴ |
| Batch size | 128 |
| Epochs | 30–100 |
| Initialization | Kaiming (ReLU) for hidden, Xavier for output |
| Loss | Cross-entropy |
| Split | 80 / 20 train / val |

Recommended additions over v1.0.0:

- **Dropout** p = 0.1 after `fc2` and `fc3`.
- **Cosine LR schedule** with warmup.
- **Early stopping** on val loss (patience 10).

---

## 5. Characteristics

| Metric | Zkash-10K (v1) | Zkash-0.1M (v2) |
|---|---:|---:|
| Trainable parameters | 10,000 | **100,000** |
| Forward MACs (per sample) | ≈ 20k | ≈ 200k |
| fp32 size | 40 KB | **≈ 400 KB** |
| int8 size | 10 KB | 100 KB |
| Typical val acc (8 Gaussians) | ~1.00 | ~1.00 |
| Time-to-overfit (N=4,096) | never | < 5 epochs |
| CPU throughput | > 10k/s | ~2k/s |

---

## 6. Limitations

- Fixed input dimension (64). Not convolutional.
- Still small by production standards — 0.1M is the low end of "real" models.
- Can overfit quickly on small datasets; requires regularization to stay useful.
- Output head is fixed to 8 classes by default (configurable).

---

## 7. Extensions

- **Zkash-0.1M-Residual** — add skip connections while preserving the 100,000-param budget.
- **Zkash-0.1M-Conv** — replace `fc1` with a conv stem for image inputs.
- **Zkash-0.1M-Quantized** — int8 deployment (100 KB).
- **Zkash-1M** — next scale-up (v3.0.0).

---

## 8. Conclusion

Zkash-0.1M is a 10× scale-up of the Zkash-10K reference, with an exact 100,000-parameter budget. Its uniform 4-layer design, hourglass hidden geometry, and full bias coverage make it a clean testbed for studying overfitting, regularization, and optimization dynamics at the low end of practical model sizes.

> *Still small enough to understand, now large enough to misbehave.*

---

**Citation**

```
Zkash-0.1M: A 100,000-Parameter Reference Neural Network for Education.
Technical Report ZK-2025-02, v2.0.0.
```
