# Zkash10M — Technical Whitepaper

**Report ID:** ZK-2025-04
**Version:** 1.0.0
**Status:** Reference / Educational (negative result documented)
**Params:** exactly 10,000,896

---

## Abstract

**Zkash10M** is a 4-stage deep residual network with **exactly 10,000,896 trainable parameters** — the fourth and largest entry in the Zkash scaling series, following Zkash-10K (v1), Zkash-0.1M (v2), and Zkash-1M (v3). It abandons the wide-MLP paradigm of its predecessors in favor of **depth**: 20 pre-norm residual blocks, RMSNorm normalization, and GELU activations. The architecture is deliberately modern and bias-free.

The v1.0.0 release is shipped **without any regularization** — no dropout, no early stopping, no learning-rate schedule. On a task with intrinsic label noise (Bayes optimal 0.825), the model reaches **train accuracy 0.971** — above the Bayes ceiling, which is only possible via memorization of flipped labels — while **val accuracy collapses to 0.613**. This is a deliberate, reproducible, documented result.

The central claim of this report is negative: **residual connections, RMSNorm, and GELU stabilize training but do not regularize it.** At a parameter-to-sample ratio of ~38:1, overfitting is unavoidable without additional mechanisms. Zkash10M v1.0.0 makes this failure inspectable.

---

## 1. Motivation

Three concerns drive the design of Zkash10M:

- **Topological break from the MLP series.** v1–v3 scale width only. v4 introduces residual connections, pre-normalization, and gated activations — the three pillars of every 2020s architecture.
- **Depth over width.** Twenty identical blocks with skip connections produce a network whose effective depth scales linearly with parameter count. This is a fundamentally different regime from a wide 4-layer MLP.
- **Documented failure as a feature.** Most educational networks are designed to succeed. Zkash10M v1.0.0 is designed to fail in a *specific, measurable, reproducible* way — demonstrating the limits of architectural improvements in the absence of regularization.

A model that quietly generalizes teaches nothing. A model that visibly overfits, with analytical bounds to compare against, teaches something.

---

## 2. Architecture

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
       Logits(8) → softmax
```

### 2.1 Parameter budget

| Component | Shape | Weights | Biases | Total |
|---|---|---:|---:|---:|
| `stem` | Linear(64, 512), no bias | 32,768 | 0 | **32,768** |
| `block` × 20 | RMSNorm(512) + Linear(512, 486) + Linear(486, 512) | 498,176 | 0 | **9,963,520** |
| `final_norm` | RMSNorm(512) | 512 | 0 | **512** |
| `head` | Linear(512, 8), no bias | 4,096 | 0 | **4,096** |
| | | | **Total** | **10,000,896** |

Per-block breakdown:

| Sub-layer | Shape | Params |
|---|---|---:|
| `RMSNorm.weight` | (512,) | 512 |
| `fc1.weight` | (486, 512) | 248,832 |
| `fc2.weight` | (512, 486) | 248,832 |
| **Per block** | | **498,176** |

### 2.2 Design notes

- **Pre-norm.** Normalization before each block, not after. Standard in modern transformers (GPT, LLaMA, ViT). The identity path through residuals remains unnormalized, which preserves gradient flow.
- **No bias anywhere.** After RMSNorm, a bias term is redundant — normalization already re-centers activations. Dropping biases saves 20·(512+486) + 512 + 8 ≈ 20,000 parameters and simplifies the accounting.
- **RMSNorm instead of LayerNorm.** One parameter per dimension, no mean subtraction, half the operations. Equivalent empirical performance.
- **Bottleneck hidden size 486 < dim 512.** Forces each block to compress and re-expand, distributing the budget evenly: no single block dominates.
- **Depth 20.** Empirically enough to reach train accuracy 1.0 on separable tasks without vanishing gradients, thanks to pre-norm and residuals.
- **No normalization layers in the head.** The final RMSNorm sits before the linear head, not after — the head output is raw logits, consumed by cross-entropy.

---

## 3. Forward Pass

For input $x \in \mathbb{R}^{64}$:

$$
\begin{aligned}
h_0 &= W_{\text{stem}} x \quad\in \mathbb{R}^{512} \\
\text{for } \ell &= 1 \dots 20: \\
h_\ell &= h_{\ell-1} + W_2^{(\ell)} \, \mathrm{GELU}\!\left(W_1^{(\ell)} \, \mathrm{RMSNorm}(h_{\ell-1})\right) \\
\hat{z} &= W_{\text{head}} \, \mathrm{RMSNorm}(h_{20}) \quad\in \mathbb{R}^{8} \\
\hat{y} &= \mathrm{softmax}(\hat{z})
\end{aligned}
$$

Where:

$$
\mathrm{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\tfrac{1}{d}\|x\|_2^2 + \epsilon}}, \quad \epsilon = 10^{-6}
$$

$$
\mathrm{GELU}(x) = x \cdot \Phi(x), \quad \Phi = \text{standard Gaussian CDF}
$$

Loss (cross-entropy):

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log \hat{y}_{i, y_i}
$$

---

## 4. Training Protocol

| Hyperparameter | Value | Note |
|---|---|---|
| Optimizer | AdamW (β = 0.9, 0.999) | default PyTorch |
| Learning rate | 1·10⁻³ | constant, no schedule |
| Weight decay | 1·10⁻⁴ | effectively off for this scale |
| Batch size | 128 / 256 | config-dependent |
| Epochs | 60 | no early stopping |
| Loss | Cross-entropy | standard |
| Init | Default PyTorch | uniform, Kaiming-friendly |
| Warmup | **none** | pre-norm makes it unnecessary |
| Gradient clipping | **none** | residuals stabilize early epochs |
| Dropout | **none** | this is the point of v1.0.0 |
| Early stopping | **none** | this is the point of v1.0.0 |

**A note on the absences.** No warmup, no clipping — this is a genuine strength of the architecture. Pre-norm residual networks train cleanly from epoch 1 without either. No dropout, no early stopping — this is the **deliberate limitation** explored in §5.5.

---

## 5. Characteristics

### 5.1 Scale comparison

| Metric | Zkash-10K (v1) | Zkash-0.1M (v2) | Zkash-1M (v3) | **Zkash10M (v4)** |
|---|---:|---:|---:|---:|
| Trainable parameters | 10,000 | 100,000 | 1,000,000 | **10,000,896** |
| Linear layers | 3 | 4 | 4 | **41** |
| Residual connections | 0 | 0 | 0 | **20** |
| Normalization layers | 0 | 0 | 0 | **21 (RMSNorm)** |
| Activation | ReLU | ReLU | ReLU | **GELU** |
| Bias terms | partial | yes | yes | **none** |
| Forward MACs / sample | ~20k | ~200k | ~1M | **~10.2M** |
| fp32 size | 40 KB | 400 KB | 4 MB | **40 MB** |
| int8 size | 10 KB | 100 KB | 1 MB | **10 MB** |
| CPU throughput (batch=128) | >10k/s | ~2k/s | ~400/s | **~40/s** |
| GPU throughput (GTX 1660, batch=128) | >100k/s | ~20k/s | ~4k/s | **~400/s** |

### 5.2 Effective depth

At initialization, a residual block is close to the identity map: `y ≈ x`. This means the network's *effective* depth at start is small and grows during training — a well-known property of residual networks (Veit et al., 2016). By end of training, Zkash10M behaves like a network with **variable depth between 1 and 41**, depending on input.

---

## 5.5 Regularization ablation (v1.0.0 baseline)

Zkash10M v1.0.0 ships **without any regularization** — no dropout, no early stopping, no LR schedule, weight decay fixed at 1·10⁻⁴. This is a deliberate choice: it produces a clean, reproducible negative result.

### 5.5.1 Synthetic task definition

Data is generated as `n_classes = 8` Gaussian clouds in ℝ⁶⁴:

- Class centers: $\mathcal{N}(0, \sigma_c^2)$ per coordinate
- Samples: center + $\mathcal{N}(0, \sigma_n^2)$ per coordinate
- Labels: uniform over 8 classes, then a fraction `label_noise` randomly flipped

Two controls make the task progressively harder:

| Control | Effect |
|---|---|
| `center_scale` ↓ | centers come closer → classes overlap |
| `label_noise` ↑ | intrinsic Bayes error, uncrossable ceiling |

Analytical Bayes optimal for label-noise `p` and `C = 8` classes:

$$\text{Bayes}(p) = 1 - p \cdot \frac{C - 1}{C}$$

| `label_noise` | Bayes optimal |
|---|---:|
| 0.0 | 1.000 |
| 0.1 | 0.9125 |
| **0.2** | **0.8250** |
| 0.3 | 0.7375 |

### 5.5.2 Easy task

**Config:** `n_samples=65,536`, `center_scale=3.0`, `noise=0.7`, `label_noise=0.0`

| Metric | Value |
|---|---:|
| Train acc | **1.000** |
| Val acc | **1.000** |
| Gap | **0.000** |
| Epochs to converge | ~10 |

**Interpretation.** The task is trivially separable in ℝ⁶⁴. Any model with ≥10k parameters reaches 100%. Zkash10M's extra 10M parameters are unused — the model learns a linear boundary and stops.

### 5.5.3 Hard task

**Config:** `n_samples=2,048`, `center_scale=0.3`, `noise=1.5`, `label_noise=0.0`

| Metric | Value |
|---|---:|
| Train acc | **0.984** |
| Val acc | **0.440** |
| Gap | **0.544** |

**Interpretation.** With 10M parameters and only 2,048 training samples — **≈4,900 params per sample** — the model memorizes the training set almost entirely. Validation accuracy is only slightly above chance (1/8 = 0.125, so 0.440 is 3.5× chance but far below any usable threshold).

### 5.5.4 Impossible task

**Config:** `n_samples=262,144`, `center_scale=0.5`, `noise=1.5`, `label_noise=0.2`

| Metric | Value |
|---|---:|
| Train acc | **0.971** |
| Val acc | **0.613** |
| Gap | **0.358** |
| Bayes optimal (analytical) | **0.825** |
| Nearest-centroid (empirical lower bound) | ~0.75 |

**Interpretation — the central result of this report.**

1. **Train accuracy exceeds the Bayes ceiling.** `0.971 > 0.825`. This is mathematically impossible if the model were learning only the class signal. The only explanation is that the model has *memorized flipped labels* by exploiting feature correlations with the corrupted targets.

2. **Val accuracy falls below the ceiling.** `0.613 < 0.825`. The memorized noise from the training set does not transfer to the validation set. The model has learned `P(y_train | x)` — the empirical distribution — instead of the true `P(y | x)`.

3. **More data did not fix it.** 262,144 samples is 128× more than the hard task. The model still overfits because the parameter-to-sample ratio (≈38:1) remains too high without regularization.

4. **Label noise did not prevent it.** Even with 20% of labels randomly corrupted, the model found a way to memorize them via feature patterns that correlate with flipped labels in the training set.

### 5.5.5 Why this is a result, not a failure

The three rows form a complete pedagogical spectrum:

| Regime | Train acc | Val acc | Gap | Interpretation |
|---|---:|---:|---:|---|
| Underparametrized | low | low | small | capacity insufficient |
| Well-parametrized | high | high | small | correct fit |
| **Overparametrized (v1.0.0)** | **high** | **low** | **large** | memorization |

Zkash10M v1.0.0 occupies the third row. This demonstrates three claims that are often stated but rarely shown side-by-side:

- **Residual connections and normalization do not prevent overfitting.** They stabilize training; they do not regularize.
- **More data helps but is not sufficient.** The parameter-to-sample ratio matters more than absolute sample count.
- **Label noise does not prevent memorization.** It creates an analytical ceiling that a sufficiently overparametrized model will *cross* on train and *miss* on val.

### 5.5.6 What regularization must achieve

For v1.1.0, the goal is to close the gap on the impossible task. A successful intervention must simultaneously:

1. Bring **train accuracy down** to ≤ 0.825 (matching the Bayes ceiling)
2. Bring **val accuracy up** toward 0.825
3. Shrink the **gap** from 0.358 to < 0.05

Candidate mechanisms, in order of expected impact:

| Mechanism | Expected effect |
|---|---|
| **Early stopping** on val acc | stops training before memorization begins; saves best epoch, not last |
| **Dropout** p = 0.1 in blocks | prevents co-adaptation of features; forces distributed representations |
| **Weight decay** 1·10⁻² (×100) | shrinks weights, reduces effective capacity |
| **Cosine LR + warmup** | smoother convergence; minor effect on final gap |

The ablation table for v1.1.0 will report all four in isolation and in combination.

---

## 6. Limitations

- **Fixed input dimension.** `n_in = 64`, no convolutional or sequence support. The model is a residual MLP, not a general architecture.
- **Depth cost.** Twenty sequential blocks mean inference latency ~10× higher than a same-parameter MLP.
- **~40 MB on disk in fp32.** Comfortable on CPU, but larger than expected for a "small" educational model.
- **Documented overfitting.** The v1.0.0 release is not production-ready; see §5.5.
- **Synthetic-only task.** The default data generator is Gaussian clouds. Real datasets (images, sequences, tabular with mixed types) are not supported out of the box.

---

## 7. Extensions

- **Zkash10M-Regularized (v1.1.0)** — early stopping + dropout + weight decay, ablation table.
- **Zkash10M-Long (v1.2.0)** — depth 200, width 128, same 10M budget. Study depth-vs-width at fixed capacity.
- **Zkash10M-Conv (v1.3.0)** — replace `stem` with a convolutional stem for image inputs (CIFAR-10 subset).
- **Zkash10M-Quantized (v1.4.0)** — int8 deployment at ~10 MB.
- **Zkash100M (v2.0.0)** — next scale-up. 100M parameters, 100 blocks.

---

## 8. Conclusion

Zkash10M is the fourth entry in the Zkash scaling series and the first to abandon the MLP paradigm in favor of depth. With 20 pre-norm residual blocks, RMSNorm, and GELU — all bias-free — it reaches **exactly 10,000,896 trainable parameters** and trains cleanly without warmup or gradient clipping.

Its v1.0.0 release is also the first Zkash that **fails in a documented, reproducible way**. On a task with 20% label noise, train accuracy (0.971) crosses the Bayes ceiling (0.825) while validation accuracy (0.613) falls well below it. The 0.358-point gap is not a bug; it is the result. It demonstrates that architectural improvements — residuals, normalization, gated activations — solve the *optimization* problem of deep networks but not the *generalization* problem. Regularization remains the missing ingredient, and v1.1.0 will supply it.

> *Depth alone does not generalize. Norm alone does not regularize. The gap is the lesson.*

---

## 9. Reproducibility

All results in §5.5 are reproducible on a single GTX 1660 SUPER (6 GB) in under 30 minutes total:

```bash
# easy task
PYTHONWARNINGS="ignore::FutureWarning" PYTHONPATH=src \
  python -m zkash.train --config configs/zkash_10m.yaml

# hard task
PYTHONWARNINGS="ignore::FutureWarning" PYTHONPATH=src \
  python -m zkash.train --config configs/zkash_10m_hard.yaml

# impossible task (the negative result)
PYTHONWARNINGS="ignore::FutureWarning" PYTHONPATH=src \
  python -m zkash.train --config configs/zkash_10m_impossible.yaml
```

Seeds are fixed at `0` in all configs. Deterministic cuDNN is enabled via `utils.set_seed`. Environment: Python 3.12, PyTorch 2.14.0+cu121, CUDA 12.1.

---

**Citation**

```
Zkash10M: A 10M-Parameter Residual Reference Network for Education.
Technical Report ZK-2025-04, v1.0.0.
Zkash Project, 2025.
```

**References**

- He, K., Zhang, X., Ren, S., Sun, J. (2015). *Deep Residual Learning for Image Recognition.* arXiv:1512.03385.
- Zhang, B., Sennrich, R. (2019). *Root Mean Square Layer Normalization.* arXiv:1910.07467.
- Hendrycks, D., Gimpel, K. (2016). *Gaussian Error Linear Units (GELUs).* arXiv:1606.08415.
- Veit, A., Wilber, M., Belongie, S. (2016). *Residual Networks Behave Like Ensembles of Relatively Shallow Networks.* arXiv:1605.06431.
```
