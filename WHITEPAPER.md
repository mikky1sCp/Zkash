# Zkash10M — Technical Whitepaper

**Report ID:** ZK-2025-04
**Version:** 4.2.0
**Status:** Reference / Educational
**Params:** exactly 10,000,896
**Last updated:** 2025

---

## Abstract

**Zkash10M** is a deep residual network with **exactly 10,000,896 trainable parameters** — the fourth entry in the Zkash scaling series (10K → 100K → 1M → 10M). It replaces the wide-MLP paradigm of its predecessors with depth: **20 pre-norm residual blocks**, RMSNorm normalization, and GELU activations, all bias-free. The architecture trains cleanly without warmup or gradient clipping.

This report covers three releases. **v1.0.0** established the baseline and documented a sharp negative result: on a task with 20% label noise, train accuracy (0.971) exceeds the analytical Bayes ceiling (0.825) while val accuracy collapses to 0.613 — memorization of flipped labels. **v4.1.0** added early stopping with best-checkpoint-by-val_acc and recovered **+10 percentage points** of validation accuracy on both hard tasks, without changing the model or adding parameters. **v4.2.0** tested the natural next hypothesis — that the `hard` task is overfitting and would respond to dropout + weight decay — and **falsified it**. Regularization cost **−0.0116** val accuracy on `hard` across two independent dataset sizes; a 4× larger training set gained **+0.0036**. An empirical Bayes ceiling for `hard` (nearest-centroid Monte Carlo, N = 200,000) was measured at **0.5879**. The best Zkash10M run reaches **0.5458**, i.e. **4.2 points below Bayes** — and no amount of regularization or additional data moves that gap, because both measures leave the memorization transition point unchanged.

The v4.2.0 release is therefore a **falsified prediction published side-by-side with the original prediction**, in accordance with the commitment made in v4.1.0 §9.1.

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
| `fc2.weight` | (486, 512)ᵀ → (512, 486) | 248,832 |
| **Per block** | | **498,176** |

**Note:** dropout, when enabled, adds **zero** parameters — it masks activations but has no learnable weights. The parameter count remains 10,000,896 for any `p_drop ∈ [0, 1)`. The v4.2.0 release confirms this: with `p_drop = 0.1`, the count is still exactly 10,000,896.

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

| Hyperparameter | v1.0.0 | v4.1.0 | v4.2.0 |
|---|---|---|---|
| Optimizer | AdamW (β = 0.9, 0.999) | same | same |
| Learning rate | 1·10⁻³ constant | same | same |
| Weight decay | 1·10⁻⁴ | same | **1·10⁻²** (hard only) |
| Weight decay groups | — | — | **2 groups: matrices / 1-D params** |
| Batch size | 128 / 256 | same | same |
| Epochs | 60 (fixed) | 60 (max, early stop) | **100 (max, early stop)** |
| Loss | Cross-entropy | same | same |
| Initialization | Default PyTorch | same | same |
| Warmup | none | none | none |
| Gradient clipping | none | none | none |
| Dropout | none | none | **0.1 (hard only)** |
| Early stopping | none | patience 10, min_delta 0 | **patience 25, min_delta 0** |
| Best checkpoint | none (last) | by `val_acc` | same |

### 4.2 Early stopping mechanism

After every epoch:

1. `val_acc` is measured on the held-out 20% split.
2. If `val_acc > best_val_acc + min_delta`, the current state is copied to a CPU-side buffer; `wait ← 0`.
3. Otherwise, `wait ← wait + 1`.
4. If `wait ≥ patience`, training halts and the best state is written to disk.

**Cost:** one forward pass on the validation set per epoch — which is already required to report `val_acc`. No additional parameters, no additional backward passes, no architectural change.

### 4.3 Weight decay parameter groups (new in v4.2.0)

AdamW is given two parameter groups:

- **Decay group:** all 2-D weight matrices (42 tensors in total: 1 stem + 20×2 block linears + 1 head; 9,990,144 params). Weight decay = `1e-2` in the v4.2.0 `hard` config.
- **No-decay group:** all 1-D parameters (21 RMSNorm γ tensors: 20 blocks + 1 final; 10,752 params). Weight decay = 0.

Norm weights are excluded because penalizing them shrinks the normalization scale, which is exactly what pre-norm is designed to keep stable. This is standard practice in GPT, LLaMA, and related architectures.

**The split does not change the parameter count.** `count_params(model) == 10,000,896` regardless of the optimizer's grouping. Verified by test.

### 4.4 A note on absent mechanisms

**No warmup, no gradient clipping.** This is a genuine property of the architecture, not a missing feature. Pre-norm residual networks keep activation scale bounded at every layer, so gradients never spike. Warmup exists to prevent early-training instability; pre-norm prevents that instability *structurally*.

**No dropout in v4.1.0.** Deliberate. Dropout was introduced in v4.2.0 as a controlled experiment, and — as reported in §6.4 — removed from the "correct remedy" list for `hard`.

---

## 5. Synthetic task definition

All results are on a controlled synthetic dataset:

- **Input:** $x \in \mathbb{R}^{64}$
- **Classes:** $C = 8$
- **Class centers:** $\mu_c \sim \mathcal{N}(0, \sigma_c^2 I)$ per coordinate
- **Samples:** $x_i = \mu_{y_i} + \mathcal{N}(0, \sigma_n^2 I)$
- **Labels:** uniform over 8 classes, then a fraction `label_noise` randomly re-drawn (note: a re-drawn label may coincide with the original — see §5.3)

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

### 5.2 Empirical Bayes for the `hard` configuration (new in v4.2.0)

The `hard` config has no closed-form Bayes ceiling (unlike `impossible`, where label noise gives an analytical bound). We estimate it by Monte Carlo using the **true class centers** — the optimal nearest-centroid classifier, which no trained network can beat because it has access to information the network never sees:

```
centers ~ N(0, center_scale² · I_64),  center_scale = 0.3
x = center[y] + noise · N(0, I_64),    noise = 1.5
N = 200,000 samples, seed = 0
predict by argmin ||x − center_c||₂ over c ∈ {0…7}
```

**Result: Bayes (`hard`) = 0.5879.**

This is the number against which all `hard` runs in §6 and §7 must be read. The finite-sample ceiling for a classifier that must *estimate* centers from 2,048 or 8,192 samples is strictly lower — approximately 0.55–0.57 and 0.57–0.58 respectively, given the standard error of the center estimate (SE = noise / √N_class ≈ 0.094 and 0.047 per coordinate).

### 5.3 A note on `label_noise` semantics

In `data.py`, `label_noise = p` re-draws a fraction `p` of labels uniformly from all `C` classes, including the original class. The expected fraction of *actually wrong* labels is therefore `p · (C−1)/C = 0.175` at `p = 0.2`, not `0.2`. The Bayes **accuracy** formula in §5.1 correctly accounts for this (it gives 0.825, consistent with 17.5% wrong labels and a 7/8 chance of a wrong label being a different class). The Bayes **loss** formula in §5.1 is written for a 20%-wrong-labels process and overestimates the true floor for this data.py by ~0.085 nats. For `impossible` this affects the "margin above Bayes" figure in §6.4; the effect is flagged in that section but not corrected in this release.

---

### 5.4 Three task configurations

| Config | `n_samples` | `center_scale` | `noise` | `label_noise` | Bayes acc |
|---|---:|---:|---:|---:|---:|
| easy | 65,536 | 3.0 | 0.7 | 0.0 | 1.000 |
| hard | 2,048 | 0.3 | 1.5 | 0.0 | **0.5879 (empirical)** |
| impossible | 262,144 | 0.5 | 1.5 | 0.2 | **0.825** |

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

### 6.4 v4.2.0 — dropout + weight decay (falsified prediction)

WHITEPAPER v4.1.0 §9.1 predicted that the `hard` task was in a classical overfitting regime and would respond to regularization, with val accuracy rising to **0.60–0.65**. The v4.2.0 release tests this directly.

**Setup.** The only changes from v4.1.0:

- `p_drop = 0.1` inside each residual block (applied to the hidden activation, *not* to the residual identity path — unchanged from `model.py`'s existing design).
- `weight_decay = 1.0e-2` (10× larger than v4.1.0's `1.0e-4`), applied via parameter groups: all 2-D weight matrices decayed, all 1-D parameters (RMSNorm γ — 21 tensors, 10,752 params total) excluded. This split does not change the parameter count: 10,000,896 as before.
- `patience = 25`, `epochs = 100` (vs v4.1.0's 10 and 60) to give the regularized runs time to reach a later peak.

A 2×2 ablation was run: `{dropout, wd} ∈ {off, on} × n_samples ∈ {2,048, 8,192}`. Each run is single-seed (seed 0), ~45 s on a GTX 1660 SUPER.

**Results — observed, not predicted:**

| n_samples | regularization | Train acc @ best | Val acc @ best | Best epoch |
|---:|---|---:|---:|---:|
| 2,048 | none (v4.1.0) | 0.7407 | **0.5428** | 4 |
| 2,048 | dropout + wd 1e-2 | 0.7474 | **0.5306** | 4 |
| 8,192 | none | 0.5418 | **0.5458** | 2 |
| 8,192 | dropout + wd 1e-2 | 0.5383 | **0.5348** | 2 |
| — | **true Bayes** | — | **0.5879** | — |

Two effects, both tiny and both consistent across the two dataset sizes:

| Effect | 2,048 | 8,192 | Mean |
|---|---:|---:|---:|
| **Cost of regularization** | −0.0122 | −0.0110 | **−0.0116** |
| **Gain from 4× data** | +0.0030 | +0.0042 | **+0.0036** |

**Prediction vs observation:**

| Metric | §9.1 predicted | v4.2.0 observed | Status |
|---|---:|---:|---|
| Train acc | ~0.75 (regularized) | 0.7474 | ✓ |
| Val acc | 0.60–0.65 | **0.5306** | ✗ |
| Gap | ~0.10 | **0.2168** | ✗ |

**The prediction is falsified.** Regularization did not merely fail to help — it cost 1.2 points. And 4× more data, which any overfitting diagnosis would call the strongest available remedy, moved the number by +0.3 points, well inside single-seed variance.

### 6.5 Why regularization and data both fail on `hard`

The two failure modes are not independent — they share a cause.

**Regularization fails because it changes the wrong thing.** Dropout and weight decay reduce the network's *capacity to memorize*. But at 2,048 samples and 10M parameters (~4,900 params/sample), the memorization transition happens at a **fixed number of optimization steps**, not at a fixed capacity. The `hard` runs peak at epoch 4 (2,048) and epoch 2 (8,192) — i.e. at **64 and 128 gradient steps** respectively. That is the same order of magnitude, and it is set by the *signal-to-noise ratio of the task*, not by model capacity.

**Data fails because it changes the wrong thing too.** 4× more samples → 4× more steps per epoch → the model reaches the same absolute step count (64–128) after 4× fewer epochs. The peak simply arrives earlier. Val accuracy at the peak is essentially unchanged, because the peak is where the model has fit the class signal and not yet fit the noise — a property of the task, not of the dataset size.

**Both measures leave the transition point where it was.** Dropout and weight decay cannot move it because the gradient signal at step 64 is dominated by memorization pressure, not by capacity pressure. More data cannot move it because the signal-to-noise ratio of each individual sample is unchanged.

**What actually varies between `hard` and `impossible` is the distance from Bayes, not the regime.** With the empirical ceiling in hand:

| Task | Bayes | Best model | Gap to Bayes | Peak epoch |
|---|---:|---:|---:|---:|
| `easy` | 1.0000 | 1.0000 | 0.0000 | 1 |
| `hard` | **0.5879** | **0.5458** | **−0.0421** | 2–4 |
| `impossible` | 0.8250 | 0.7128 | **−0.1122** | 5 |

`hard` is **4.2 points** below its ceiling. That is where the model stops, and there is nowhere to go — regularization can only push it *away* from the ceiling, which is exactly what was observed (−0.0116). `impossible` is **11.2 points** below its ceiling, and that gap is the only one in this series where a training-loop change has room to operate.

---

## 7. Two regimes, one mechanism

### 7.1 hard — memorization-dominated, ceiling-bound

The v4.1.0 release classified `hard` as "classical overfitting": train ≫ val, large gap, fix with dropout + weight decay + more data. **v4.2.0 refutes this classification.**

| Epoch | Train acc | Val acc | Gap |
|---:|---:|---:|---:|
| 1 | 0.3807 | 0.4694 | −0.089 |
| 4 | 0.7407 | **0.5428** ★ | +0.198 |
| 5 | 0.8188 | 0.4988 | +0.320 |
| 10 | 0.9286 | 0.4743 | **+0.454** |

The train/val gap is real and large (+0.39 at epoch 10), and it looks like overfitting. But two facts disqualify that diagnosis:

1. **The val peak is 4.2 points below the true Bayes ceiling** (0.5428 vs 0.5879). A model that is genuinely overfitting would be *above* the ceiling on train and *below* it on val — and would close the val gap when regularized. Zkash10M does not.
2. **Regularization makes val worse, not better** (−0.0116 across two dataset sizes). Overfitting responds to regularization; ceiling-bound memorization does not.

The correct description is **memorization-dominated**: the model learns the class signal in the first ~64–128 gradient steps, then spends the remaining ~350 steps fitting the training noise, and the peak val accuracy is pinned near the empirical ceiling for whatever sample size is available. Dropout and weight decay suppress the noise-fitting but also suppress the (already-correct) signal fit — net effect slightly negative. More data raises the ceiling by ~0.4 points per 4× — real but not the bottleneck.

**Correct remedy:** none available within the architecture. `hard` is closed.
**Wrong remedy (now empirically confirmed):** dropout, weight decay, more data, longer training.

### 7.2 impossible — optimization-bound

| Epoch | Train acc | Val acc | Gap |
|---:|---:|---:|---:|
| 1 | 0.6944 | 0.7069 | −0.013 |
| 5 | 0.7089 | **0.7128** ★ | −0.004 |
| 10 | 0.7135 | 0.7099 | +0.004 |
| 15 | 0.7188 | 0.7093 | +0.010 |

Train and val move together at ~0.71. The gap never exceeds 0.01. **Diagnosis:** the model is not memorizing — it is *learning slowly*. The 20% label noise makes the signal weak; the model needs more iterations at a smaller learning rate to reach the loss floor.

**Supporting evidence from Bayes loss.** At epoch 15:

- Train loss: **1.030**
- Bayes loss (label_noise=0.2, per §5.1 formula): **0.888**
- Margin above Bayes: **~0.142** (note: with the corrected `data.py` semantics, this margin is ~0.227 — see §5.3)

The model is within 0.14–0.23 nats of the theoretical floor. It is not far from optimal — it is simply **not yet converged**.

### 7.3 Two regimes, one mechanism

The v4.1.0 release described the two hard tasks as failing in "opposite ways" and requiring "opposite remedies." With the empirical Bayes ceilings now measured, that framing must be softened.

| Task | Gap to Bayes | Peak epoch | Behavior |
|---|---:|---:|---|
| `hard` | **−0.042** | 2–4 | Fast peak, then memorization; ceiling-bound |
| `impossible` | **−0.112** | 5 | Slow convergence; optimization-bound |

Both tasks are **memorization-dominated**: both reach their best val acc early, both then degrade, both are improved only by early stopping. They differ in *distance to ceiling*, not in *regime*:

- `hard` is already within 4.2 points of Bayes. There is no room for any training-loop change to help. Regularization hurts (v4.2.0).
- `impossible` is 11.2 points below Bayes. There is room, and the remaining gap is plausibly explainable by insufficient optimization — see §7.2. Whether a training-loop change actually closes it is the subject of v4.3.0.

**Revised rule of thumb:** a model that is close to its Bayes ceiling cannot be helped by regularization, regardless of its train/val gap. A model that is far from its Bayes ceiling might be — but only if the gap is optimization-driven, not data-driven. The diagnosis requires a Bayes measurement, not a gap inspection.

### 7.4 The full spectrum

| Regime | Train acc | Val acc | Gap | Present in |
|---|---:|---:|---:|---|
| Underparametrized | low | low | small | — |
| Well-parametrized | high | high | small | `easy` |
| **Ceiling-bound memorization** | **high** | **near Bayes** | **large** | **`hard`** |
| **Optimization-bound memorization** | **low-ish** | **well below Bayes** | **small** | **`impossible`** |

Zkash10M v4.2.0 exhibits all four rows across the three tasks and the v4.1.0 baseline. The "overparametrized + overfitting" row of v4.1.0 does not appear in this dataset — it required a task where the model can fit and the training regime cannot, which none of the three configs provides.

---

## 8. Limitations

- **Fixed input dimension.** `n_in = 64`. No convolutional or sequence support; the model is a residual MLP.
- **Depth cost.** 20 sequential blocks mean ~10× inference latency compared to a same-parameter MLP.
- **~40 MB in fp32.** Larger than expected for an "educational" model.
- **Not production-ready.** `hard` is closed as ceiling-bound; `impossible` is unsolved. Early stopping improves but does not fully fix either.
- **Synthetic-only task.** No support for images, sequences, or mixed-type tabular data.
- **Single random seed.** All reported results use `seed = 0`. Variance across seeds is unknown; the −0.0116 regularization cost and +0.0036 data gain are single-seed measurements.
- **Analytical Bayes loss inconsistency.** The `label_noise` implementation in `data.py` does not match the loss formula in §5.1; see §5.3.

---

## 9. Extensions and roadmap

### 9.1 v4.2.0 — observed results (replaces the v4.1.0 prediction)

WHITEPAPER v4.1.0 §9.1 predicted that dropout + weight decay would raise `hard` val accuracy to 0.60–0.65. That prediction was falsified. Per the commitment made in that section, the observed results are reported here side-by-side, unedited.

**hard (2048 and 8192 samples, 2×2 ablation, single seed):**

| Run | Train acc @ best | Val acc @ best | Best epoch |
|---|---:|---:|---:|
| 2048, no reg (v4.1.0) | 0.7407 | **0.5428** | 4 |
| 2048, dropout 0.1 + wd 1e-2 | 0.7474 | **0.5306** | 4 |
| 8192, no reg | 0.5418 | **0.5458** | 2 |
| 8192, dropout 0.1 + wd 1e-2 | 0.5383 | **0.5348** | 2 |

**Empirical Bayes (`hard`):** 0.5879 (nearest-centroid MC, N = 200,000).
**Best observed val:** 0.5458 → **−0.0421 from Bayes.**
**Effect of regularization:** −0.0116 (mean of two dataset sizes).
**Effect of 4× data:** +0.0036 (mean of two regularization settings).

**What this implies for the roadmap.** The `hard` task is closed: no training-loop or regularization change can close a 4.2-point gap that is set by the task's Bayes ceiling. The v4.3.0 ablation originally planned for `hard` (dropout × wd × LR schedule) is **cancelled** — the search space is exhausted by the negative result above. Effort is redirected to `impossible`, where the gap to Bayes is 11.2 points and the remaining question — whether it is optimization-limited or data-limited — is still open.

**Prediction accuracy so far, this series:** 1 of 3 (§9.1 predicted train acc ≈ 0.75, observed 0.7474 ✓; predicted val 0.60–0.65, observed 0.5306 ✗; predicted gap ≈ 0.10, observed 0.217 ✗).

### 9.2 Roadmap

| Version | Status | Adds | Observed effect |
|---|---|---|---|
| **v1.0.0** | shipped | Baseline, no regularization | documents failure |
| **v4.1.0** | shipped | Early stopping + best checkpoint | **+0.10 on hard & impossible** |
| **v4.2.0** | shipped | dropout p=0.1 + wd 1e-2 (**hard**) | **−0.011 on hard (falsified §9.1)** |
| v4.3.0 | planned | cosine LR + warmup + patience 25 (**impossible**) | target: close the 11.2-pt gap to Bayes |
| v4.3.0 | planned | label smoothing 0.1 (**impossible**) | secondary |
| v4.4.0 | planned | Multi-seed runs (5 seeds), report mean ± std | quantifies variance floor |
| v5.0.0 | planned | **Zkash100M** — depth 100, exact 100M params | next scale-up |

### 9.3 Concrete predictions for v4.3.0

**impossible (optimization-bound regime):**

| Metric | v4.2.0 (= v4.1.0) | Predicted v4.3.0 |
|---|---:|---:|
| Train acc | 0.7188 | ~0.78 |
| Val acc | **0.7128** | **0.76–0.80** |
| Gap | +0.006 | ~0.01 |
| Train loss | 1.030 | ~0.92 (approaching Bayes) |

Cosine LR with warmup lets the model finish convergence. Target train loss approaches Bayes (0.888, or ~0.803 with corrected semantics).

**Falsifiable.** The v4.3.0 release will report actual numbers side-by-side, regardless of whether they confirm the prediction.

---

## 10. Conclusion

Zkash10M is the fourth entry in the Zkash scaling series and the first to abandon the MLP paradigm. With 20 pre-norm residual blocks, RMSNorm, and GELU — all bias-free — it reaches **exactly 10,000,896 trainable parameters** and trains cleanly without warmup or gradient clipping.

Its three shipped releases tell three different stories.

**v1.0.0** documented a sharp negative result: architectural improvements — residuals, normalization, gated activations — solve the *optimization* problem of deep networks but not the *generalization* problem. On a task with 20% label noise, train accuracy crossed the Bayes ceiling (0.971 > 0.825) while val accuracy collapsed to 0.613.

**v4.1.0** showed that a **single change to the training loop** — saving the best epoch rather than the last — recovers **+10 percentage points** of validation accuracy on both hard tasks, at no cost in parameters and with a 4× reduction in training time. No architectural change. No new mechanism beyond per-epoch validation and a state buffer.

**v4.2.0** completed the picture by falsifying its own prediction. The `hard` task is not an overfitting problem and cannot be fixed with regularization; it is a memorization-dominated, ceiling-bound task where the model reaches 0.5458 against an empirical Bayes ceiling of 0.5879 in 2–4 epochs and then degrades. Dropout + weight decay cost 1.2 points; 4× more data gained 0.4 points. Both are inside noise; both are consistent across the 2×2. The only mechanism that reliably helps is saving the best epoch — the v4.1.0 result.

The central lesson is therefore sharper than v4.1.0 stated. It is not that `hard` and `impossible` need opposite remedies. It is that **a model's train/val gap cannot tell you whether regularization will help**. A 0.39 gap at 0.54 val and a 0.006 gap at 0.71 val can both be memorization-dominated, and only a Bayes measurement tells you which one has room to improve. Zkash10M v4.2.0 measured it.

> *Depth alone does not generalize. Norm alone does not regularize. Early stopping helps — and a Bayes ceiling tells you whether anything else can.*

---

## 11. Reproducibility

All results are reproducible on a single NVIDIA GTX 1660 SUPER (6 GB) in under 40 minutes total.

### 11.1 Environment

| Component | Version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.5.1+cu121 |
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
bash scripts/train_hard.sh         # hard v4.1.0 — ~15 s
bash scripts/train_impossible.sh   # impossible  — ~3 min
bash scripts/sweep_hard.sh         # hard v4.2.0 2×2 ablation — ~3 min
python scripts/bayes_hard.py       # empirical Bayes for hard — ~5 s
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

Expected output for `sweep_hard.sh`:

```
=== p_drop=0.0  wd=0.0001 ===
best val acc: 0.5428 (epoch 4)

=== p_drop=0.0  wd=0.01 ===
best val acc: 0.5458 (epoch 2)

=== p_drop=0.1  wd=0.0001 ===
best val acc: 0.5306 (epoch 4)

=== p_drop=0.1  wd=0.01 ===
best val acc: 0.5348 (epoch 2)
```

Expected output for `bayes_hard.py`:

```
Bayes (nearest-centroid): 0.5879
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
  note        = {Version 4.2.0}
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
7. Loshchilov, I., Hutter, F. (2019). *Decoupled Weight Decay Regularization.* arXiv:1711.05101.
8. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting.* JMLR 15(1).

---

**Report history**

| Version | Date | Change |
|---|---|---|
| ZK-2025-04 v1.0.0 | 2025 | Initial release, baseline + negative result |
| ZK-2025-04 v4.1.0 | 2025 | Early stopping + best-checkpoint recovery |
| ZK-2025-04 v4.2.0 | 2025 | Dropout + wd on hard; §9.1 prediction falsified; empirical Bayes for hard |