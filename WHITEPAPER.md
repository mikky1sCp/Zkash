# Zkash10M — Technical Whitepaper

**Report ID:** ZK-2025-04
**Version:** 4.1.0
**Status:** Reference / Educational
**Params:** exactly 10,000,896
**Last updated:** 2025

---

## Abstract

**Zkash10M** is a deep residual network with **exactly 10,000,896 trainable parameters** — the fourth entry in the Zkash scaling series (10K → 100K → 1M → 10M). It replaces the wide-MLP paradigm of its predecessors with depth: **20 pre-norm residual blocks**, RMSNorm normalization, and GELU activations, all bias-free. The architecture trains cleanly without warmup or gradient clipping.

This report covers two releases. **v1.0.0** established the baseline and documented a sharp negative result: on a task with 20% label noise, train accuracy (0.971) exceeds the analytical Bayes ceiling (0.825) while val accuracy collapses to 0.613 — memorization of flipped labels. **v4.1.0** adds early stopping with best-checkpoint-by-val_acc and recovers **+10 percentage points** of validation accuracy on both hard tasks, without changing the model or adding parameters.

A key finding of v4.1.0 is that the two hard tasks fail in **fundamentally different modes** — one overfits, the other underfits — and require **opposite remedies**. Generic "add regularization" advice is wrong for one of them.

---

## 1. Motivation

Zkash10M exists to answer a specific question left open by Zkash-1M (v3):

> When width has stopped helping, does **depth** help?

The v1–v3 models all scale a 3–4 layer perceptron. Once width reaches ~1000 neurons per layer, additional capacity produces no measurable improvement on any task the series has been tested on. The natural next step is a **change of topology**, not a change of size.

Three concerns drive the design:

- **Topological break.** Introduce the three pillars of every 2020s architecture — residual connections, pre-normalization, gated activations — and measure whether they make a difference at 10M parameters.
- **Depth over width.** Twenty narrow residual blocks, not four wide ones. Effective depth scales with parameter count.
- **Honest failure.** Document the model's limits with analytical bounds, not just accuracy numbers. A model that quietly generalizes teaches nothing. A model that visibly fails, with a Bayes ceiling to compare against, teaches something.

---

## 2. Architecture

### 2.1 Full specification

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

### 2.2 Parameter budget

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

**Note:** dropout, if enabled, adds **zero** parameters — it masks activations but has no learnable weights. The parameter count remains 10,000,896 for any `p_drop ∈ [0, 1)`.

### 2.3 Design decisions

| Decision | Rationale |
|---|---|
| **Pre-norm placement** | Normalization *before* each block, not after. The residual identity path remains unnormalized → gradient flows cleanly. Standard in GPT, LLaMA, ViT. |
| **No bias in any Linear** | After RMSNorm, a bias term is redundant — normalization already re-centers activations. Saves ~20,000 parameters. |
| **RMSNorm over LayerNorm** | One parameter per dim, no mean subtraction, half the operations. Equivalent empirical performance (Zhang & Sennrich, 2019). |
| **Bottleneck `hidden = 486 < dim = 512`** | Forces each block to compress then re-expand. Distributes budget evenly — no single block dominates. |
| **Depth 20** | Empirically enough to reach train accuracy 1.0 on separable tasks without vanishing gradients. |
| **Head without final norm** | Final RMSNorm sits *before* the linear head. Head output is raw logits consumed by cross-entropy. |

---

## 3. Forward pass

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

## 4. Training protocol

### 4.1 Hyperparameters

| Hyperparameter | v1.0.0 | v4.1.0 |
|---|---|---|
| Optimizer | AdamW (β = 0.9, 0.999) | same |
| Learning rate | 1·10⁻³ constant | same |
| Weight decay | 1·10⁻⁴ | same |
| Batch size | 128 / 256 | same |
| Epochs | 60 (fixed) | 60 (max, early stop) |
| Loss | Cross-entropy | same |
| Initialization | Default PyTorch | same |
| Warmup | none | none |
| Gradient clipping | none | none |
| Dropout | none | none |
| **Early stopping** | **none** | **patience 10, min_delta 0** |
| **Best checkpoint** | **none (last)** | **by `val_acc`** |

### 4.2 Early stopping mechanism (v4.1.0)

After every epoch:

1. `val_acc` is measured on the held-out 20% split.
2. If `val_acc > best_val_acc + min_delta`, the current state is copied to a CPU-side buffer; `wait ← 0`.
3. Otherwise, `wait ← wait + 1`.
4. If `wait ≥ patience`, training halts and the best state is written to disk.

**Cost:** one forward pass on the validation set per epoch — which is already required to report `val_acc`. No additional parameters, no additional backward passes, no architectural change.

### 4.3 A note on absent mechanisms

**No warmup, no gradient clipping.** This is a genuine property of the architecture, not a missing feature. Pre-norm residual networks keep activation scale bounded at every layer, so gradients never spike. Warmup exists to prevent early-training instability; pre-norm prevents that instability *structurally*.

**No dropout in v4.1.0.** Deliberate. See §6.

---

## 5. Synthetic task definition

All results are on a controlled synthetic dataset:

- **Input:** $x \in \mathbb{R}^{64}$
- **Classes:** $C = 8$
- **Class centers:** $\mu_c \sim \mathcal{N}(0, \sigma_c^2 I)$ per coordinate
- **Samples:** $x_i = \mu_{y_i} + \mathcal{N}(0, \sigma_n^2 I)$
- **Labels:** uniform over 8 classes, then a fraction `label_noise` randomly flipped

Three parameters control difficulty:

| Parameter | Effect |
|---|---|
| `center_scale` = σ_c ↓ | centers come closer → classes overlap |
| `noise` = σ_n ↑ | feature noise grows |
| `label_noise` ↑ | intrinsic Bayes error — uncrossable ceiling |

### 5.1 Analytical Bayes bounds

For `label_noise = p` and `C = 8` classes:

**Bayes accuracy:**

$$\text{Acc}_{\text{Bayes}}(p) = 1 - p \cdot \frac{C - 1}{C}$$

**Bayes loss** (for optimal probabilistic predictions $p_{\text{true}} = 1-p$, $p_{\text{other}} = p/(C-1)$):

$$\mathcal{L}_{\text{Bayes}}(p) = -(1-p) \ln(1-p) - p \ln\!\left(\frac{p}{C-1}\right)$$

| `label_noise` | Bayes accuracy | Bayes loss |
|---|---:|---:|
| 0.0 | 1.0000 | 0.000 |
| 0.1 | 0.9125 | 0.443 |
| **0.2** | **0.8250** | **0.888** |
| 0.3 | 0.7375 | 1.333 |

These are hard ceilings: no model, however large or well-trained, can exceed them on this data.

### 5.2 Three task configurations

| Config | `n_samples` | `center_scale` | `noise` | `label_noise` | Bayes acc |
|---|---:|---:|---:|---:|---:|
| easy | 65,536 | 3.0 | 0.7 | 0.0 | 1.000 |
| hard | 2,048 | 0.3 | 1.5 | 0.0 | (data-limited) |
| impossible | 262,144 | 0.5 | 1.5 | **0.2** | **0.825** |

---

## 6. Results

### 6.1 v1.0.0 — no early stopping (baseline)

| Config | Train acc | Val acc | Gap |
|---|---:|---:|---:|
| easy | 1.0000 | 1.0000 | 0.000 |
| hard | 0.9840 | 0.4400 | **0.544** |
| impossible | **0.9710** | 0.6134 | **0.358** |

**Central observation for impossible:** train accuracy (0.971) **exceeds** the Bayes ceiling (0.825). This is mathematically impossible if the model learns only the class signal. The model has memorized flipped labels via feature correlations.

### 6.2 v4.1.0 — early stopping + best checkpoint

| Config | v1.0.0 (last) | v4.1.0 (best) | Δ | Best epoch | Stopped at |
|---|---:|---:|---:|---:|---:|
| easy | 1.0000 | 1.0000 | 0.000 | 1 | 11 (was 60) |
| hard | 0.4400 | **0.5428** | **+0.103** | 4 | 14 (was 60) |
| impossible | 0.6134 | **0.7128** | **+0.099** | 5 | 15 (was 60) |

**Average gain on non-trivial tasks: +0.101.** Training time reduced by ~4×.

The model is **unchanged**. The gain comes entirely from saving the best epoch instead of the last.

### 6.3 Why early stopping helps so much

On the `hard` task, validation accuracy peaks at **0.5428 on epoch 4** and decays to 0.4743 by epoch 10. The v1.0.0 release saved the *last* checkpoint (val 0.4400). The peak was discarded.

| Epoch | Train acc | Val acc | Gap |
|---:|---:|---:|---:|
| 1 | 0.3807 | 0.4694 | −0.089 |
| 4 | 0.7407 | **0.5428** ★ | +0.198 |
| 5 | 0.8188 | 0.4988 | +0.320 |
| 10 | 0.9286 | 0.4743 | **+0.454** |

The 0.544 gap in v1.0.0 was not inevitable. It was the *final* gap after 60 epochs of overfitting — not the *minimum* gap the model achieved.

**General principle:** validation accuracy is not monotonic. Saving the last checkpoint is almost never correct.

---

## 7. Two failure modes

Early stopping did more than recover accuracy. It revealed that the two hard tasks fail for **fundamentally different reasons**.

### 7.1 hard — classical overfitting

| Epoch | Train acc | Val acc | Gap |
|---:|---:|---:|---:|
| 1 | 0.3807 | 0.4694 | −0.089 |
| 4 | 0.7407 | **0.5428** ★ | +0.198 |
| 5 | 0.8188 | 0.4988 | +0.320 |
| 10 | 0.9286 | 0.4743 | **+0.454** |

Train accuracy climbs to 0.93 while val stalls at 0.47. **Diagnosis:** with 2,048 samples and 10M params (~4,900 params/sample), the model memorizes the training set.

### 7.2 impossible — underfitting

| Epoch | Train acc | Val acc | Gap |
|---:|---:|---:|---:|
| 1 | 0.6944 | 0.7069 | −0.013 |
| 5 | 0.7089 | **0.7128** ★ | −0.004 |
| 10 | 0.7135 | 0.7099 | +0.004 |
| 15 | 0.7188 | 0.7093 | +0.010 |

Train and val move together at ~0.71. The gap never exceeds 0.01. **Diagnosis:** the model is not memorizing — it is *learning slowly*. The 20% label noise makes the signal weak; the model needs more iterations at a smaller learning rate to reach the loss floor.

**Supporting evidence from Bayes loss.** At epoch 15:

- Train loss: **1.030**
- Bayes loss (label_noise=0.2): **0.888**
- Margin above Bayes: **0.142**

The model is within 0.14 nats of the theoretical floor. It is not far from optimal — it is simply **not yet converged**.

### 7.3 Different remedies for different modes

| Failure mode | Symptom | Correct remedy | Wrong remedy |
|---|---|---|---|
| **Overfitting** (hard) | train ≫ val | dropout, weight decay, more data | LR schedule, longer training |
| **Underfitting** (impossible) | train ≈ val, both low | LR schedule, warmup, longer patience | dropout (hurts further) |

**This is the central lesson of Zkash10M.** Both failures occur in the *same model*, on the *same architecture*, with the *same optimizer*. Only the data difficulty differs. A one-size-fits-all "regularization" prescription would fail one of the two tasks.

### 7.4 The full spectrum

| Regime | Train acc | Val acc | Gap | Present in |
|---|---:|---:|---:|---|
| Underparametrized | low | low | small | — |
| Well-parametrized | high | high | small | easy |
| **Overparametrized + overfitting** | **high** | **low** | **large** | **hard** |
| **Overparametrized + underfitting** | **low** | **low** | **small** | **impossible** |

Zkash10M v4.1.0 exhibits three of the four rows. The fourth requires a task that the model can fit but the training regime cannot — a case not present in this series.

---

## 8. Limitations

- **Fixed input dimension.** `n_in = 64`. No convolutional or sequence support; the model is a residual MLP.
- **Depth cost.** 20 sequential blocks mean ~10× inference latency compared to a same-parameter MLP.
- **~40 MB in fp32.** Larger than expected for an "educational" model.
- **Not production-ready.** `hard` and `impossible` tasks are unsolved. Early stopping improves but does not fix generalization.
- **Synthetic-only task.** No support for images, sequences, or mixed-type tabular data.
- **Single random seed.** All reported results use `seed = 0`. Variance across seeds is unknown.

---

## 9. Extensions and roadmap

| Version | Status | Adds | Expected effect |
|---|---|---|---|
| **v1.0.0** | shipped | Baseline, no regularization | documents failure |
| **v4.1.0** | shipped | Early stopping + best ckpt by val_acc | **+0.10 hard & impossible** |
| v4.2.0 | planned | dropout p=0.1 + weight decay 1e-2 (**hard**) | val +0.05–0.10 |
| v4.2.0 | planned | cosine LR + 500-step warmup + patience 25 (**impossible**) | val +0.05–0.08 |
| v4.3.0 | planned | Full ablation: dropout × wd × LR schedule | — |
| v4.4.0 | planned | Multi-seed runs (5 seeds), report mean ± std | — |
| v5.0.0 | planned | **Zkash100M** — depth 100, exact 100M params | next scale-up |

### 9.1 Concrete predictions for v4.2.0

**hard (overfitting regime):**

| Metric | v4.1.0 | Predicted v4.2.0 |
|---|---:|---:|
| Train acc | 0.9286 | ~0.75 (regularized) |
| Val acc | **0.5428** | **0.60–0.65** |
| Gap | +0.386 | ~0.10 |

Dropout and weight decay should *reduce* train accuracy while *increasing* val — the classic signature of successful regularization.

**impossible (underfitting regime):**

| Metric | v4.1.0 | Predicted v4.2.0 |
|---|---:|---:|
| Train acc | 0.7188 | ~0.78 |
| Val acc | **0.7128** | **0.76–0.80** |
| Gap | +0.006 | ~0.01 |
| Train loss | 1.030 | ~0.92 |

Cosine LR with warmup lets the model finish convergence. Target train loss approaches Bayes (0.888).

Both predictions are falsifiable. The v4.2.0 release will report actual numbers side-by-side.

---

## 10. Conclusion

Zkash10M is the fourth entry in the Zkash scaling series and the first to abandon the MLP paradigm. With 20 pre-norm residual blocks, RMSNorm, and GELU — all bias-free — it reaches **exactly 10,000,896 trainable parameters** and trains cleanly without warmup or gradient clipping.

Its two shipped releases tell complementary stories.

**v1.0.0** documented a sharp negative result: architectural improvements — residuals, normalization, gated activations — solve the *optimization* problem of deep networks but not the *generalization* problem. On a task with 20% label noise, train accuracy crossed the Bayes ceiling (0.971 > 0.825) while val accuracy collapsed to 0.613.

**v4.1.0** showed that a **single change to the training loop** — saving the best epoch rather than the last — recovers **+10 percentage points** of validation accuracy on both hard tasks, at no cost in parameters and with a 4× reduction in training time. No architectural change. No new mechanism beyond per-epoch validation and a state buffer.

The most important discovery was unexpected: the two hard tasks fail in **opposite ways**. The `hard` task overfits (train ≫ val); the `impossible` task underfits (train ≈ val, both low). They require **opposite remedies**. A generic regularization strategy would help one and hurt the other.

> *Depth alone does not generalize. Norm alone does not regularize. Early stopping helps — but the failure mode determines the cure.*

---

## 11. Reproducibility

All results are reproducible on a single NVIDIA GTX 1660 SUPER (6 GB) in under 30 minutes total.

### 11.1 Environment

| Component | Version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.14.0+cu121 |
| CUDA | 12.1 |
| GPU | NVIDIA GeForce GTX 1660 SUPER |

### 11.2 Setup

```bash
git clone https://github.com/mikky1sCp/zkash10m.git
cd zkash10m

python -m venv .venv
source .venv/Scripts/activate     # Windows / Git Bash
# source .venv/bin/activate       # Linux / macOS

pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

### 11.3 Reproduce all results

```bash
bash scripts/train.sh              # easy        — ~30 s
bash scripts/train_hard.sh         # hard        — ~15 s
bash scripts/train_impossible.sh   # impossible  — ~3 min
```

Expected output for `impossible`:

```
device: cuda
[Zkash10M] trainable params: 10000896
epoch   1 | train loss 1.1121 | train acc 0.6944 | val acc 0.7069 ★
epoch   2 | train loss 1.0753 | train acc 0.7057 | val acc 0.7111 ★
epoch   3 | train loss 1.0683 | train acc 0.7070 | val acc 0.7117 ★
epoch   5 | train loss 1.0607 | train acc 0.7089 | val acc 0.7128 ★
epoch  10 | train loss 1.0461 | train acc 0.7135 | val acc 0.7099
epoch  15 | train loss 1.0303 | train acc 0.7188 | val acc 0.7093
early stopping at epoch 15 (no improvement for 10 epochs)

best val acc: 0.7128 (epoch 5)
saved best checkpoint → checkpoints/zkash10m_impossible.pt
```

### 11.4 Determinism

- All configs fix `seed: 0`
- `utils.set_seed` seeds Python, NumPy, PyTorch (CPU + CUDA)
- `torch.backends.cudnn.deterministic = False`, `benchmark = True` — bit-exact reruns across machines are *not* guaranteed. For strict determinism, flip both flags in `utils.set_seed`.

---

## 12. Citation

```bibtex
@techreport{zkash10m,
  title       = {Zkash10M: A 10M-Parameter Residual Reference Network for Education},
  number      = {ZK-2025-04},
  institution = {Zkash Project},
  year        = {2025},
  note        = {Version 4.1.0}
}
```

---

## References

1. He, K., Zhang, X., Ren, S., Sun, J. (2015). *Deep Residual Learning for Image Recognition.* arXiv:1512.03385.
2. Zhang, B., Sennrich, R. (2019). *Root Mean Square Layer Normalization.* arXiv:1910.07467.
3. Hendrycks, D., Gimpel, K. (2016). *Gaussian Error Linear Units (GELUs).* arXiv:1606.08415.
4. Veit, A., Wilber, M., Belongie, S. (2016). *Residual Networks Behave Like Ensembles of Relatively Shallow Networks.* arXiv:1605.06431.
5. Ba, J. L., Kiros, J. R., Hinton, G. E. (2016). *Layer Normalization.* arXiv:1607.06450.
6. Prechelt, L. (1998). *Early Stopping — But When?* In: Neural Networks: Tricks of the Trade. Springer.

---

**Report history**

| Version | Date | Change |
|---|---|---|
| ZK-2025-04 v1.0.0 | 2025 | Initial release, baseline + negative result |
| ZK-2025-04 v4.1.0 | 2025 | Early stopping + best-checkpoint recovery |
```

---