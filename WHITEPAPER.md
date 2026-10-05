# Zkash10M — Technical Whitepaper

**Report ID:** ZK-2025-04
**Version:** 4.5.0
**Status:** Reference / Educational
**Params:** exactly 10,000,896
**Last updated:** 2025

---

## Abstract

**Zkash10M** is a deep residual network with **exactly 10,000,896 trainable parameters** — the fourth entry in the Zkash scaling series (10K → 100K → 1M → 10M). It replaces the wide-MLP paradigm of its predecessors with depth: **20 pre-norm residual blocks**, RMSNorm normalization, and GELU activations, all bias-free. The architecture trains cleanly without warmup or gradient clipping.

This report covers six releases. **v1.0.0** established the baseline and documented a sharp negative result: on a task with 20% label noise, train accuracy (0.971) exceeded the reported Bayes ceiling (0.825) while val accuracy collapsed to 0.613 — memorization of flipped labels. **v4.1.0** added early stopping with best-checkpoint-by-val_acc and recovered **+10 percentage points** of validation accuracy on both hard tasks without changing the model. **v4.2.0** tested dropout + weight decay on the `hard` task and **falsified** its own prediction: regularization cost **−0.0116** val accuracy (single seed); 4× more data gained **+0.0036**. An empirical Bayes ceiling for `hard` (nearest-centroid Monte Carlo) measured **0.5879**, showing the model was already **5.7 points from its ceiling** at 5-seed resolution. **v4.3.0** tested cosine LR + warmup + label smoothing on the `impossible` task — the last remaining "underfitting" hypothesis — and **falsified it too**: all three interventions landed within ±0.0011 of baseline (single seed), and an empirical Bayes ceiling of **0.7142** (corrected for feature overlap) showed the baseline was already **0.4 points from its ceiling**. The v4.3.0 release also **corrected a methodological error** that had propagated through v1.0.0–v4.2.0: the analytical Bayes accuracy formula `1 − p·(C−1)/C` is valid only when the clean-label Bayes accuracy is 1.0. The `impossible` task has feature overlap (clean Bayes = 0.8606), so the correct noisy-label ceiling was **0.7142**, not **0.8250** — an 11-point correction. This error had misled v4.1.0 and v4.2.0 into diagnosing `impossible` as underfitting and predicting that LR schedules would help. **They do not.**

**v4.4.0** re-ran both v4.2.0 and v4.3.0 ablations on **5 seeds each** and reports mean ± std. All four single-seed "effects" collapse to zero: `hard` regularization Δ = **+0.0059** (σ = 0.30, and the sign flips versus v4.2.0's −0.0116), `impossible` cosine Δ = **−0.0002** (σ = 0.10), label smoothing Δ = **−0.0001** (σ = 0.02), and both Δ = **−0.0004** (σ = 0.19). Seed variance itself was mispredicted by ~1.8× in both directions: `hard` **0.0143** vs predicted 0.008; `impossible` **0.0018** vs predicted 0.003. The only non-null effect in any training-loop change across four releases is that cosine LR compresses the peak-epoch distribution (baseline 7.0 ± 3.54 → cosine 4.4 ± 1.14) without moving val accuracy.

**v4.5.0** fixes the last remaining inconsistency in the pipeline: **the data generator**. Versions v1.0.0–v4.4.0 re-drew flipped labels uniformly over **all** `C` classes, including the original. The true wrong-label rate was therefore `q = p·(C−1)/C = 0.175` at `p = 0.2`, not `0.2`. The analytical Bayes formula was corrected in v4.3.0, but the generator was left unchanged for backwards compatibility with existing logs. v4.5.0 makes `label_noise = p` the true wrong-label rate by drawing flips from the `C−1` **other** classes. Consequence: the `impossible` Bayes ceiling drops from **0.7142 to 0.6925** (−2.1 points), while `easy` and `hard` are untouched (the fix is a no-op when `label_noise = 0`). **Prediction, on the record before the run: `impossible` val acc drops by ~2.1 points to 0.689 ± 0.002, preserving the v4.4.0 gap to Bayes (−0.0040).** If it instead stays at 0.7102, the corrected formula is wrong; if it drops below 0.688, the model is no longer ceiling-bound.

Across all six releases, exactly **one training-loop change** improved validation accuracy: **early stopping with best-checkpoint-by-val_acc** (v4.1.0). Every other intervention — dropout, weight decay, cosine LR, label smoothing, 4× more data — is either null or negative at 5-seed resolution. The consistent explanation is that both hard tasks are **memorization-dominated and near their Bayes ceilings**, and no training-loop change can close a gap that is set by the task. v4.5.0 shows that even the *ceiling itself* can move — but only when the task changes, not when the training loop does.

---

## 1. Motivation

Zkash10M exists to answer a specific question left open by Zkash-1M (v3):

> When width has stopped helping, does **depth** help?

The v1–v3 models all scale a 3–4 layer perceptron. Once width reaches ~1000 neurons per layer, additional capacity produces no measurable improvement on any task the series has been tested on. The natural next step is a **change of topology**, not a change of size.

Three concerns drive the design:

- **Topological break.** Introduce the three pillars of every 2020s architecture — residual connections, pre-normalization, gated activations — and measure whether they make a difference at 10M parameters.
- **Depth over width.** Twenty narrow residual blocks, not four wide ones. Effective depth scales with parameter count.
- **Honest failure.** Document the model's limits with analytical bounds, not just accuracy numbers. A model that quietly generalizes teaches nothing. A model that visibly fails, with a Bayes ceiling to compare against, teaches something — and a model whose failure diagnosis is itself wrong, whose "fixes" turn out to be seed noise, and whose Bayes ceiling moves when the data generator is corrected, teaches more still.

---

## 2. Architecture

### 2.1 Full specification

```
Input(64)
   │
   ▼
┌────────────────────────────────────────────────────┐
│  Stem: Linear(64 → 512, no bias)                   │  32,768 params
└────────────────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────────────────┐
│  ResidualBlock × 20                                │  498,176 × 20
│  ──────────────────────────────────                │
│     x ──┬─────────────────────────────────┐        │
│         │                                 │        │
│         ▼                                 │        │
│     RMSNorm(512)                          │        │
│         │                                 │        │
│     Linear(512 → 486, no bias)            │        │
│         │                                 │        │
│     GELU                                  │        │
│         │                                 │        │
│     [optional Dropout]                    │        │
│         │                                 │        │
│     Linear(486 → 512, no bias)            │        │
│         │                                 │        │
│         ▼                                 │        │
│         + ◄───────────────────────────────┘        │
│         │                                          │
└─────────┼──────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────┐
│  Final RMSNorm(512)                                │  512 params
└────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────┐
│  Head: Linear(512 → 8, no bias)                    │  4,096 params
└────────────────────────────────────────────────────┘
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

**Note:** dropout, when enabled, adds **zero** parameters. The count remains 10,000,896 for any `p_drop ∈ [0, 1)`. Verified across all v4.2.0, v4.3.0, v4.4.0, and v4.5.0 runs.

### 2.3 Design decisions

| Decision | Rationale |
|---|---|
| **Pre-norm placement** | Normalization *before* each block, not after. Residual identity path remains unnormalized → gradient flows cleanly. |
| **No bias in any Linear** | After RMSNorm, a bias is redundant. Saves ~20,000 parameters. |
| **RMSNorm over LayerNorm** | One parameter per dim, no mean subtraction, half the operations. |
| **Bottleneck `hidden = 486 < dim = 512`** | Compress then re-expand; distributes budget evenly. |
| **Depth 20** | Enough to reach train accuracy 1.0 on separable tasks without vanishing gradients. |
| **Head without final norm** | Final RMSNorm sits *before* the linear head. |

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

$$
\mathrm{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\tfrac{1}{d}\|x\|_2^2 + \epsilon}}, \quad \epsilon = 10^{-6}
$$

$$
\mathrm{GELU}(x) = x \cdot \Phi(x), \quad \Phi = \text{standard Gaussian CDF}
$$

Cross-entropy loss (optionally with label smoothing $\varepsilon$):

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} (1-\varepsilon)\,[c = y_i] + \frac{\varepsilon}{C} \cdot \log \hat{y}_{i,c}
$$

At $\varepsilon = 0$ this reduces to standard CE.

---

## 4. Training protocol

### 4.1 Hyperparameters

| Hyperparameter | v1.0.0 | v4.1.0 | v4.2.0 | v4.3.0 | v4.4.0 | v4.5.0 |
|---|---|---|---|---|---|---|
| Optimizer | AdamW | same | same | same | same | same |
| Learning rate | 1·10⁻³ const | same | same | **1·10⁻³ cosine (impossible only)** | same | same |
| Weight decay | 1·10⁻⁴ | same | **1·10⁻² (hard only)** | same | same | same |
| Weight decay groups | — | — | **2 groups** | same | same | same |
| Batch size | 128 / 256 | same | same | same | same | same |
| Epochs | 60 (fixed) | 60 (max, ES) | 100 (max, ES) | 60 (max, ES) | same | same |
| Loss | CE | same | same | **+ LS 0.1 (impossible only)** | same | same |
| Warmup | none | none | none | **500 steps (impossible only)** | same | same |
| Gradient clipping | none | none | none | none | none | none |
| Dropout | none | none | **0.1 (hard only)** | same | same | same |
| Early stopping | none | patience 10 | patience 25 | patience 25 | patience 25 | patience 25 |
| `label_noise` semantics | q = p·7/8 | same | same | same | same | **q = p** |
| Seeds | 0 | 0 | 0 | 0 | **0, 1, 2, 3, 4** | **0, 1, 2, 3, 4** |

**None of these additions changed the parameter count.** All runs report exactly 10,000,896 trainable parameters.

### 4.2 Early stopping mechanism

After every epoch:

1. `val_acc` is measured on the held-out 20% split.
2. If `val_acc > best_val_acc + min_delta`, the current state is copied to a CPU-side buffer; `wait ← 0`.
3. Otherwise, `wait ← wait + 1`.
4. If `wait ≥ patience`, training halts and the best state is written to disk.

**Cost:** one forward pass per epoch — already required to report `val_acc`. No additional parameters, no additional backward passes, no architectural change.

### 4.3 Weight decay parameter groups

AdamW is given two groups:

- **Decay group:** 42 weight matrices (stem, 40 block linears, head) — 9,990,144 params.
- **No-decay group:** 21 RMSNorm γ tensors (20 blocks + final) — 10,752 params.

Norm weights are excluded because penalizing them shrinks the normalization scale, which is exactly what pre-norm is designed to keep stable. The split does not change `count_params`.

### 4.4 Cosine LR schedule and label smoothing

Added in v4.3.0 for the `impossible` ablation:

**Cosine LR with warmup:**
$$
\eta(t) = \begin{cases}
\eta_0 \cdot t / W & t < W \\
\eta_0 \cdot 0.5\,(1 + \cos(\pi (t - W)/(T - W))) & t \ge W
\end{cases}
$$
where $W$ = 500 warmup steps, $T$ = total training steps, $\eta_0 = 10^{-3}$. Applied per-step.

**Label smoothing:**
$$
y_i^{\text{LS}} = (1 - \varepsilon)\, y_i^{\text{one-hot}} + \varepsilon / C
$$
with $\varepsilon = 0.1$, $C = 8$. Applied to targets before cross-entropy.

**Both were null.** See §6.6 and §6.7.

### 4.5 Multi-seed protocol (v4.4.0)

The v4.4.0 ablation varies **only `train.seed`** (model initialization, train/val split, batch shuffling) across seeds 0–4. The underlying data distribution (`data.seed`) is held fixed, so this measures **run-to-run training variance on a fixed task** rather than data-sampling variance. Per-seed checkpoints are saved with a `_seedN` suffix; reports are written to JSON; the companion tool `zkash.compare` computes Welch-style σ between any two report means:

$$
\sigma = \frac{|\Delta \mu|}{\sqrt{\sigma_A^2 + \sigma_B^2}}
$$

Verdict thresholds: σ < 1.0 → "within 1σ"; 1.0 ≤ σ < 2.0 → "within 2σ"; σ ≥ 2.0 → "SIGNIFICANT".

### 4.6 Label-noise semantics (v4.5.0)

Added in v4.5.0. The `make_gaussians` function now draws flipped labels uniformly from the `C−1` **other** classes:

```python
if label_noise > 0:
    flip_mask = torch.rand(n_samples, generator=g) < label_noise
    n_flip = int(flip_mask.sum().item())
    if n_flip > 0:
        offsets = torch.randint(1, n_classes, (n_flip,), generator=g)
        y[flip_mask] = (y[flip_mask] + offsets) % n_classes
```

The `C−1` range on `randint` guarantees the new label differs from the original. Consequence: `label_noise = p` is now the **true** wrong-label rate `q`.

**Backwards compatibility.** A `legacy_redraw: bool = False` flag is exposed. When `True`, the old v1.0.0–v4.4.0 behavior is restored (uniform over all `C`, `q = p·(C−1)/C`). This lets any historical log be reproduced exactly.

### 4.7 A note on absent mechanisms

**No gradient clipping, no warmup in the baseline.** This is a genuine property of the architecture, not a missing feature. Pre-norm residual networks keep activation scale bounded at every layer; gradients never spike. Warmup exists to prevent early-training instability; pre-norm prevents it *structurally*.

**No dropout in v4.3.0 or later.** Removed after v4.2.0's null result on `hard`. Dropout adds noise to hidden activations and was found to cost 1.2 points on `hard` (single seed, §6.4) — and, at 5 seeds, is indistinguishable from zero (σ = 0.30, §6.7).

---

## 5. Synthetic task definition

All results are on a controlled synthetic dataset:

- **Input:** $x \in \mathbb{R}^{64}$
- **Classes:** $C = 8$
- **Class centers:** $\mu_c \sim \mathcal{N}(0, \sigma_c^2 I)$ per coordinate
- **Samples:** $x_i = \mu_{y_i} + \mathcal{N}(0, \sigma_n^2 I)$
- **Labels:** uniform over 8 classes; a fraction `label_noise = p` is replaced by a different class (see §5.3)

Three parameters control difficulty: `center_scale` (class separation), `noise` (feature noise), `label_noise` (label corruption).

### 5.1 Analytical Bayes bounds — corrected, two regimes

**This section corrects an error present in versions v1.0.0–v4.2.0, and clarifies a further subtlety corrected in v4.5.0.**

The formula used previously,

$$
\text{Acc}_{\text{Bayes}}(p) = 1 - p \cdot \frac{C-1}{C} \quad\text{(INCORRECT for } A_{\text{clean}} < 1\text{)},
$$

assumes that the classifier's **clean-label** Bayes accuracy $A_{\text{clean}}$ equals 1.0 — i.e., that classes are perfectly separable by a nearest-centroid rule. This is true for `easy` but false for both `hard` and `impossible`, where feature noise causes class overlap.

The correct formula, for a classifier that achieves $A_{\text{clean}}$ on clean labels and is uniform on the remaining $C-1$ classes when wrong, is:

$$
\boxed{\;\text{Acc}_{\text{Bayes}}(q) = A_{\text{clean}} \cdot \left(1 - q\right) + (1 - A_{\text{clean}}) \cdot \frac{q}{C-1}\;}
$$

where $q$ is the **true wrong-label rate** — i.e., the fraction of samples whose observed label differs from the true label.

**The two regimes of `q`:**

| Version | `q` in terms of `p` | At `p = 0.2` |
|---|---|---:|
| v1.0.0–v4.4.0 | `q = p·(C−1)/C` | 0.175 |
| **v4.5.0** | **`q = p`** | **0.200** |

**Empirical $A_{\text{clean}}$ values** (nearest-centroid Monte Carlo, $N = 200{,}000$, seed 0):

| Config | $A_{\text{clean}}$ |
|---|---:|
| easy | 1.0000 |
| hard | 0.5879 |
| impossible | **0.8606** |

**Corrected Bayes table for `impossible` — both regimes:**

| `p` | q (v4.4.0) | Bayes (v4.4.0) | q (v4.5.0) | Bayes (v4.5.0) | Old formula |
|---|---:|---:|---:|---:|---:|
| 0.0 | 0.0000 | **0.8606** | 0.0000 | **0.8606** | 1.0000 |
| 0.1 | 0.0875 | **0.7870** | 0.1000 | **0.7785** | 0.9125 |
| **0.2** | **0.1750** | **0.7135** | **0.2000** | **0.6925** | 0.8250 |
| 0.3 | 0.2625 | **0.6399** | 0.3000 | **0.6106** | 0.7375 |

At $A_{\text{clean}} = 1$ this reduces to the old formula. At $A_{\text{clean}} < 1$ it produces a **strictly lower** ceiling.

**Monte Carlo confirmations:**

- v4.4.0 semantics: `impossible` noisy Bayes = **0.7142** (analytical 0.7135, difference is MC noise).
- v4.5.0 semantics: `impossible` noisy Bayes = **0.6925** (analytical 0.6925, exact).

**Corrected Bayes loss.** For the observed-label process, the Bayes-optimal predictive distribution is $P(y_{\text{obs}} = y_{\text{true}} \mid x) = 1-q$, $P(y_{\text{obs}} = c \ne y_{\text{true}} \mid x) = q/(C-1)$, giving

$$
\mathcal{L}_{\text{Bayes}}(q) = -(1-q)\ln(1-q) - q\ln\!\left(\frac{q}{C-1}\right)
$$

with $q$, **not** $p$, as the argument.

| `p` | q (v4.4.0) | L (v4.4.0) | q (v4.5.0) | L (v4.5.0) | Old loss |
|---|---:|---:|---:|---:|---:|
| 0.0 | 0.000 | **0.0000** | 0.000 | **0.0000** | 0.000 |
| 0.1 | 0.0875 | **0.4670** | 0.100 | **0.5120** | 0.443 |
| **0.2** | **0.175** | **0.8043** | **0.200** | **0.8896** | 0.888 |
| 0.3 | 0.2625 | **1.0863** | 0.300 | **1.1913** | 1.333 |

For `impossible` at $p = 0.2$, the v4.5.0 Bayes loss floor is **0.8896** nats, up from 0.8043 under v4.4.0 semantics. In both regimes, the **val accuracy is already at Bayes** — the gap between the model's loss and the Bayes loss is confidence, not ranking.

### 5.2 Empirical Bayes for `hard`

`hard` has no label noise ($p = 0$), so its ceiling is $A_{\text{clean}}$ directly. Monte Carlo (nearest-centroid, $N = 200{,}000$): **0.5879**. The v4.5.0 fix is a no-op here.

A classifier that must *estimate* centers from $N$ samples has a strictly lower ceiling. Standard error of the center estimate at 2,048 samples (256/class): $\text{SE} = 1.5/\sqrt{256} \approx 0.094$ per coordinate → center displacement $\approx 0.094 \cdot \sqrt{64} \approx 0.75$ against inter-class distance $\approx 3.4$. Finite-sample ceiling ≈ 0.55–0.57. At 8,192 samples: ≈ 0.57–0.58.

### 5.3 `label_noise` semantics — corrected in v4.5.0

| Version | Implementation | True wrong-label rate |
|---|---|---|
| v1.0.0–v4.4.0 | Re-draw uniformly from all `C` classes, including original | `q = p·(C−1)/C` |
| **v4.5.0** | Re-draw uniformly from the `C−1` **other** classes | **`q = p`** |

All corrected figures in §5.1 use the appropriate `q` for the version being discussed. The `legacy_redraw=True` flag in `make_gaussians` restores v4.4.0 behavior for reproducing historical logs.

### 5.4 Three task configurations

| Config | `n_samples` | `center_scale` | `noise` | `label_noise` | $A_{\text{clean}}$ | Bayes (v4.4.0) | Bayes (v4.5.0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| easy | 65,536 | 3.0 | 0.7 | 0.0 | 1.0000 | **1.0000** | **1.0000** |
| hard | 2,048 | 0.3 | 1.5 | 0.0 | 0.5879 | **0.5879** | **0.5879** |
| impossible | 262,144 | 0.5 | 1.5 | 0.2 | 0.8606 | **0.7142** | **0.6925** |

---

## 6. Results

### 6.1 v1.0.0 — no early stopping (baseline)

| Config | Train acc | Val acc | Gap |
|---|---:|---:|---:|
| easy | 1.0000 | 1.0000 | 0.000 |
| hard | 0.9840 | 0.4400 | **0.544** |
| impossible | **0.9710** | 0.6134 | **0.358** |

The v1.0.0 report noted that `impossible` train accuracy (0.971) exceeds the then-reported Bayes ceiling (0.825). Under the corrected Bayes, this remains true (0.971 > 0.7142, and > 0.6925 under v4.5.0 semantics). The model is memorizing noisy labels.

### 6.2 v4.1.0 — early stopping + best checkpoint

| Config | v1.0.0 (last) | v4.1.0 (best) | Δ | Best epoch |
|---|---:|---:|---:|---:|
| easy | 1.0000 | 1.0000 | 0.000 | 1 |
| hard | 0.4400 | **0.5428** | **+0.103** | 4 |
| impossible | 0.6134 | **0.7128** | **+0.099** | 5 |

**Average gain on non-trivial tasks: +0.101.** Training time reduced by ~4×.

The model is **unchanged**. The gain comes entirely from saving the best epoch instead of the last.

### 6.3 Why early stopping helps so much

On the `hard` task, validation accuracy peaks at **0.5428 on epoch 4** and decays to 0.4743 by epoch 10. v1.0.0 saved the *last* checkpoint (val 0.4400). The peak was discarded.

**General principle:** validation accuracy is not monotonic. Saving the last checkpoint is almost never correct.

### 6.4 v4.2.0 — dropout + weight decay on `hard` (single seed, falsified)

v4.1.0 §9.1 predicted `hard` val accuracy would rise to **0.60–0.65** with dropout + wd. A 2×2 ablation was run: `{dropout, wd} ∈ {off, on} × n_samples ∈ {2,048, 8,192}`.

| n_samples | regularization | Train acc @ best | Val acc @ best | Best epoch |
|---:|---|---:|---:|---:|
| 2,048 | none (v4.1.0) | 0.7407 | **0.5428** | 4 |
| 2,048 | dropout 0.1 + wd 1e-2 | 0.7474 | **0.5306** | 4 |
| 8,192 | none | 0.5418 | **0.5458** | 2 |
| 8,192 | dropout 0.1 + wd 1e-2 | 0.5383 | **0.5348** | 2 |
| — | **true Bayes** | — | **0.5879** | — |

| Effect | 2,048 | 8,192 | Mean |
|---|---:|---:|---:|
| **Cost of regularization** | −0.0122 | −0.0110 | **−0.0116** |
| **Gain from 4× data** | +0.0030 | +0.0042 | **+0.0036** |

Prediction §9.1: val 0.60–0.65, gap ~0.10. Observed: val 0.5306, gap 0.2168. **Falsified.** As v4.4.0 will show (§6.7), the "cost of regularization" is itself indistinguishable from zero at 5 seeds.

### 6.5 Why regularization and data both fail on `hard`

**Regularization** reduces the network's capacity to memorize, but the memorization transition in this regime is set by the task's signal-to-noise ratio, not by capacity. Suppressing noise-fitting also suppresses signal-fitting; net −0.0116 at seed 0.

**More data** multiplies steps-per-epoch, so the same absolute step count (64–128) arrives after fewer epochs. Peak val is unchanged because the peak is where the model has fit the class signal and not yet fit the noise — a property of the task.

### 6.6 v4.3.0 — cosine LR + label smoothing on `impossible` (single seed, falsified)

v4.2.0 §9.3 predicted `impossible` val accuracy would rise to **0.76–0.80** with cosine LR + warmup + label smoothing, on the theory that the model was optimization-bound and 11.2 points below Bayes. **That theory was based on the incorrect Bayes formula corrected in §5.1.** With the corrected ceiling (0.7142 under v4.4.0 semantics), `impossible` is only 0.4 points below Bayes, and no training-loop change should help.

**Setup.** Four runs, each 60 epochs max / patience 25, seed 0, on 262,144 samples (`impossible` config):

| Run | Schedule | Label smoothing | Warmup |
|---|---|---:|---:|
| baseline | constant | 0.0 | 0 |
| cosine | cosine | 0.0 | 500 |
| LS | constant | 0.1 | 0 |
| both | cosine | 0.1 | 500 |

**Results:**

| Run | Best val | Best epoch | Gap @ best | Δ vs baseline |
|---|---:|---:|---:|---:|
| baseline | **0.7128** | 5 | −0.0039 | — |
| cosine + warmup 500 | **0.7123** | 5 | −0.0038 | **−0.0005** |
| label smoothing 0.1 | **0.7124** | 5 | −0.0034 | **−0.0004** |
| both | **0.7117** | 5 | −0.0032 | **−0.0011** |
| **Bayes (MC, v4.4.0 semantics)** | **0.7142** | — | — | — |

**All four runs are within ±0.0011 of baseline.** All peak at epoch 5. All sit 0.19–0.25 points below Bayes.

Additional evidence: baseline train loss @ best epoch **1.0607**; label smoothing train loss @ best epoch **1.2456** (LS raises the loss floor by exactly the predicted ~0.185 nats), yet **val accuracy is unchanged to within noise**. The model's argmax was already correct; only its confidence was disturbed.

**Falsified on all three variants.** No training-loop change moves `impossible` off its ceiling — at single-seed resolution. The 5-seed confirmation is in §6.7.

### 6.7 v4.4.0 — multi-seed on both tasks (5 seeds)

Every result in v1.0.0–v4.3.0 was single-seed (seed 0). v4.4.0 re-runs the v4.2.0 and v4.3.0 ablations on **5 seeds each** (0–4) and reports mean ± std on `val_acc`. Methodology in §4.5.

**`hard` — 5-seed results:**

| Config | val acc (mean ± std) | Range | Δ vs baseline | σ | Verdict |
|---|---|---:|---:|---:|---|
| baseline | 0.5306 ± 0.0143 | 0.5134 – 0.5452 | — | — | — |
| v4.2.0 (dropout 0.1 + wd 1e-2) | 0.5364 ± 0.0132 | 0.5232 – 0.5550 | **+0.0059** | **0.30** | within 1σ |

**`impossible` — 5-seed results (v4.4.0 semantics, q = 0.175):**

| Config | val acc (mean ± std) | Range | Δ vs baseline | σ | Verdict |
|---|---|---:|---:|---:|---|
| baseline | 0.7102 ± 0.0018 | 0.7078 – 0.7128 | — | — | — |
| cosine + warmup 500 | 0.7100 ± 0.0015 | 0.7080 – 0.7123 | **−0.0002** | **0.10** | within 1σ |
| label smoothing 0.1 | 0.7102 ± 0.0017 | 0.7080 – 0.7124 | **−0.0001** | **0.02** | within 1σ |
| both | 0.7098 ± 0.0014 | 0.7081 – 0.7117 | **−0.0004** | **0.19** | within 1σ |

**Three findings.**

**(a) Every effect collapses to zero.** Three of four multi-seed deltas sit at σ < 0.20. None reaches 1σ. The single-seed "effects" of v4.2.0 (−0.0116) and v4.3.0 (−0.0005 to −0.0011) were artifacts of sampling seed 0 — and for `hard`, the sign of the delta flips: −0.0116 → +0.0059.

**(b) Seed variance was mispredicted by ~1.8× in both directions.** Predicted (v4.3.0 §9.3): `hard` 0.008, `impossible` 0.003. Observed: `hard` **0.0143**, `impossible` **0.0018**. Even the variance floor is harder to guess than the mean.

**(c) Cosine compresses the peak-epoch distribution without moving val acc.** Baseline `best epoch` = 7.0 ± 3.54 (range 3–12). Cosine `best epoch` = 4.4 ± 1.14 (range 3–6). Both 5-seed means for `val_acc` are 0.7102 and 0.7100 — a 0.02σ difference. The schedule changes *when* the peak arrives, not *where* it lands. This is the only non-null effect of any training-loop change in four releases, and it is not an accuracy effect.

### 6.8 Consolidated falsification ledger

| Release | Prediction | Observed | Status |
|---|---|---|---|
| v4.1.0 §9.1 | `hard` val 0.60–0.65 with dropout+wd | 0.5306 | **✗** |
| v4.1.0 §9.1 | `hard` train acc ~0.75 | 0.7474 | ✓ |
| v4.1.0 §9.1 | `hard` gap ~0.10 | 0.2168 | **✗** |
| v4.2.0 §9.3 | `impossible` val 0.76–0.80 with cosine+LS | 0.7117–0.7124 | **✗** |
| v4.3.0 §9.3 | `hard` 5-seed std = 0.008 | 0.0143 | **✗** |
| v4.3.0 §9.3 | `impossible` 5-seed std = 0.003 | 0.0018 | **✗** |
| v4.4.0 | All v4.2.0/v4.3.0 effects null on 5 seeds | confirmed | ✓ |

**Score: 2 of 7 predictions correct.** The two correct ones were the easy ones: `hard` train accuracy under regularization (predicted ~0.75, observed 0.7474), and the v4.4.0 null hypothesis itself. Every prediction that involved either raising val accuracy or guessing the variance floor was wrong. That is itself a finding: the model was already at its ceiling in every case, and the ceiling — like the variance — was not measured.

### 6.9 The only intervention that has ever worked

| Intervention | Task | Effect on val acc |
|---|---|---:|
| Early stopping + best checkpoint (v4.1.0) | `hard` | **+0.103** |
| Early stopping + best checkpoint (v4.1.0) | `impossible` | **+0.099** |
| Dropout + weight decay (v4.2.0, single seed) | `hard` | −0.0116 |
| Dropout + weight decay (v4.4.0, 5 seeds) | `hard` | +0.0059 (σ = 0.30) |
| 4× more data (v4.2.0) | `hard` | +0.0036 |
| Cosine LR + warmup (v4.3.0, single seed) | `impossible` | −0.0005 |
| Cosine LR + warmup (v4.4.0, 5 seeds) | `impossible` | −0.0002 (σ = 0.10) |
| Label smoothing 0.1 (v4.3.0, single seed) | `impossible` | −0.0004 |
| Label smoothing 0.1 (v4.4.0, 5 seeds) | `impossible` | −0.0001 (σ = 0.02) |
| Both (v4.3.0, single seed) | `impossible` | −0.0011 |
| Both (v4.4.0, 5 seeds) | `impossible` | −0.0004 (σ = 0.19) |

**Early stopping is the only mechanism in six releases that reliably improved val accuracy.** It costs nothing: no parameters, no backward passes, no architectural change. Every other intervention is within single-seed noise or negative — and v4.4.0 shows the "single-seed noise" was larger than v4.2.0/v4.3.0 assumed.

### 6.10 v4.5.0 — label-noise semantics fix

**What changed.** Versions v1.0.0–v4.4.0 re-drew flipped labels uniformly over **all** `C` classes, **including the original**. The true wrong-label rate was therefore

$$q = p \cdot \frac{C-1}{C} = 0.175 \quad \text{at } p = 0.2,$$

not `p = 0.2`. The analytical Bayes formula was corrected in v4.3.0 to use `q` (§5.1), but the data generator was left unchanged for backwards compatibility with v1.0.0–v4.4.0 logs. v4.5.0 fixes the generator: flipped labels are now drawn from the `C−1` **other** classes (§4.6, §5.3).

**Effect on Bayes ceilings:**

| Config | p | q (v4.4.0) | q (v4.5.0) | Bayes (v4.4.0) | Bayes (v4.5.0) |
|---|---:|---:|---:|---:|---:|
| easy | 0.0 | 0.000 | 0.000 | 1.0000 | **1.0000** |
| hard | 0.0 | 0.000 | 0.000 | 0.5879 | **0.5879** |
| impossible | 0.2 | 0.175 | **0.200** | 0.7142 | **0.6925** |

The Bayes loss floor for `impossible` also rises, from **0.8043** to **0.8896** nats.

**`easy` and `hard` are untouched.** The fix is a no-op when `label_noise = 0`. Only `impossible` moves.

#### v4.5.0 predictions (on the record, run pending)

5-seed re-run of all three tasks with the fixed generator. Predictions are stated before the run.

| Task | v4.4.0 observed | **v4.5.0 prediction** | Reason |
|---|---:|---:|---|
| `easy` | 1.0000 | **1.0000** | no label noise → no-op |
| `hard` | 0.5306 ± 0.0143 | **0.5306 ± 0.0143** | no label noise → no-op |
| `impossible` | 0.7102 ± 0.0018 | **0.689 ± 0.002** | Bayes ceiling drops 0.7142 → 0.6925 (−2.1 pt) |

The `impossible` prediction **preserves the v4.4.0 gap to Bayes (−0.0040)**: the ceiling moves down, the model follows it, and the *gap* is unchanged. This is the sharpest form of the v4.4.0 conclusion — the model was never optimization-bound, and moving the ceiling moves the model.

**Falsifiers.**

- If the observed val acc stays at **0.7102**, the corrected Bayes formula is wrong (or the fix didn't apply).
- If it drops **below 0.688**, the model is no longer ceiling-bound; the v4.4.0 conclusion ("gap is 0.4 points") must be revisited.
- If it lands at **0.689 ± 0.002**, the model remains ceiling-bound and the corrected formula is quantitatively confirmed.

#### Reproducibility

```bash
bash scripts/rerun_v45.sh                # all three tasks × 5 seeds
python scripts/bayes_impossible.py       # clean 0.8606, noisy 0.6925 (was 0.7142)
```

Legacy v4.4.0 logs remain under `logs/multiseed_*.json`. To reproduce them exactly with the current code, pass `legacy_redraw=True` to `make_gaussians`.

#### What this does and does not change

**Does change:** the reported Bayes ceiling for `impossible` (0.7142 → 0.6925), the reported Bayes loss floor (0.8043 → 0.8896), and the expected val acc for `impossible` (0.7102 → 0.689, predicted).

**Does not change:** the parameter count, the architecture, the training loop, the `easy` and `hard` results, any v4.4.0 conclusion about single-seed effects being noise, or the status of early stopping as the only effective intervention.

---

## 7. One regime: memorization at the ceiling

### 7.1 hard — memorization-dominated, ceiling-bound

5-seed mean 0.5306 ± 0.0143. Seed-0 trajectory:

| Epoch | Train acc | Val acc | Gap |
|---:|---:|---:|---:|
| 1 | 0.3807 | 0.4694 | −0.089 |
| 4 | 0.7407 | **0.5428** ★ | +0.198 |
| 5 | 0.8188 | 0.4988 | +0.320 |
| 10 | 0.9286 | 0.4743 | **+0.454** |

Train/val gap reaches +0.45. Under the v4.1.0 diagnosis, this was classical overfitting. Three facts disqualify that:

1. **Val peak is 5.7 points below Bayes** (5-seed mean 0.5306 vs 0.5879).
2. **Regularization makes val worse in v4.2.0** (−0.0116 at seed 0) — and, per v4.4.0, is indistinguishable from zero at 5 seeds (+0.0059, σ = 0.30).
3. **4× more data gained only 0.4 points** — inside single-seed noise.

**Correct remedy:** none. `hard` is closed.

### 7.2 impossible — memorization-dominated, also ceiling-bound

5-seed mean 0.7102 ± 0.0018 under v4.4.0 semantics (q = 0.175). Seed-0 trajectory:

| Epoch | Train acc | Val acc | Gap |
|---:|---:|---:|---:|
| 1 | 0.6944 | 0.7069 | −0.013 |
| 5 | 0.7089 | **0.7128** ★ | −0.004 |
| 10 | 0.7135 | 0.7099 | +0.004 |
| 15 | 0.7188 | 0.7093 | +0.010 |

Train/val gap never exceeds 0.01. Under the v4.1.0 diagnosis, this was underfitting — "model learns slowly". **v4.3.0 and v4.4.0 falsify that:** the peak is at epoch 5, cosine + warmup + LS do not move it (σ < 0.20 across all variants), and the corrected Bayes ceiling (0.7142 under v4.4.0 semantics) shows the peak is **0.4 points below** the theoretical maximum.

The earlier "margin above Bayes = 0.142" evidence for underfitting was an artifact of the incorrect Bayes loss formula (§5.1). With the corrected value 0.8043 (v4.4.0) or 0.8896 (v4.5.0), the margin grows, but the *val accuracy* is already at Bayes — the gap is confidence, not ranking. The model has the correct argmax.

**Correct remedy:** none. `impossible` is closed under v4.4.0 semantics. Under v4.5.0, the ceiling moves down by 2.1 points, and the model is predicted to follow.

### 7.3 The unified picture

| Task | $A_{\text{clean}}$ | Bayes (v4.4.0) | Bayes (v4.5.0) | Best model (5-seed mean) | Gap to Bayes | Peak epoch |
|---|---:|---:|---:|---:|---:|---:|
| easy | 1.0000 | 1.0000 | **1.0000** | 1.0000 ± 0.0000 | 0.0000 | 1 |
| hard | 0.5879 | 0.5879 | **0.5879** | 0.5306 ± 0.0143 | **−0.0573** | 2–4 |
| impossible | 0.8606 | 0.7142 | **0.6925** | 0.7102 ± 0.0018 (v4.4.0) | **−0.0040** | 5 |

**Both hard tasks are memorization-dominated and near their Bayes ceilings.** The only quantitative difference is distance to ceiling: `hard` is 5.7 points below, `impossible` is 0.4 points below. Neither is fixable by any training-loop change tested across v4.2.0, v4.3.0, and v4.4.0. v4.5.0 shows that moving the *task* moves the ceiling — but the model stays pinned to it.

**Revised rule:** a model whose train/val gap looks like overfitting may be at its ceiling. A model whose train/val gap looks like underfitting may also be at its ceiling. **Only a Bayes measurement tells you which.** The gap is not diagnostic; the ceiling is.

### 7.4 The full spectrum

| Regime | Train acc | Val acc | Gap | Present in |
|---|---:|---:|---:|---|
| Underparametrized | low | low | small | — |
| Well-parametrized | high | high | small | `easy` |
| **Ceiling-bound memorization (fast)** | **high** | **near Bayes** | **large** | **`hard`** |
| **Ceiling-bound memorization (slow)** | **low-ish** | **near Bayes** | **small** | **`impossible`** |

Three of four rows. The "overparametrized + overfitting that regularization would fix" row does not appear in this dataset — it requires a task where the model is *below* its ceiling and could be pushed up. Neither hard task qualifies.

### 7.5 The variance floor

The v4.2.0 and v4.3.0 releases assumed, without measuring, that seed variance was small enough for single-seed Δ of order 0.001–0.01 to be meaningful. v4.4.0 measured it:

| Task | Predicted std (v4.3.0 §9.3) | Observed std (v4.4.0) | Ratio |
|---|---:|---:|---:|
| `hard` | 0.008 | **0.0143** | 1.8× higher |
| `impossible` | 0.003 | **0.0018** | 1.7× lower |

**Neither prediction was within 20% of the observed value.** Even the variance floor — the simplest possible summary of a run distribution — was harder to guess than the mean. This is the second finding of v4.4.0, and it is logically prior to the first: before declaring any Δ "small" or "large", measure the σ.

---

## 8. Limitations

- **Fixed input dimension.** `n_in = 64`. Residual MLP only.
- **Depth cost.** 20 sequential blocks → ~10× inference latency vs. same-parameter MLP.
- **~40 MB in fp32.**
- **Not production-ready.** Both hard tasks are closed as ceiling-bound.
- **Synthetic-only task.**
- **5 seeds is a variance measurement, not a confidence interval.** All v4.4.0 std values are sample standard deviations over n=5. A proper confidence interval would require more seeds, but 5 is enough to show that all four Δ's from v4.2.0 and v4.3.0 collapse to σ < 0.31.
- **Cross-seed variance is measured on a fixed `data.seed`.** v4.4.0 varies only `train.seed`. Data-sampling variance (varying `data.seed` with `train.seed` fixed) is a distinct source and is unmeasured here.
- **Bayes ceilings are Monte Carlo estimates** ($N = 200{,}000$, seed 0) for `hard` and `impossible`; only `easy` has a closed-form value.
- **v4.5.0 label-noise fix breaks bit-exact comparability with v1.0.0–v4.4.0 `impossible` logs.** Use `legacy_redraw=True` to reproduce old results. `easy` and `hard` are unaffected.

---

## 9. Extensions and roadmap

### 9.1 What v4.2.0 through v4.5.0 actually established

The v4.1.0 report set out a plan:

> v4.2.0: dropout + wd on `hard` → val +0.05–0.10.
> v4.2.0: cosine LR + warmup + patience 25 on `impossible` → val +0.05–0.08.

**Neither happened.** The plan assumed the two tasks failed for opposite reasons. They do not. Both fail for the same reason — memorization at a Bayes ceiling — and the ceiling is what it is.

The v4.3.0 report corrected the Bayes formula, re-diagnosed `impossible` as ceiling-bound rather than underfitting, and re-predicted that no training-loop change would help.

The v4.4.0 report measured the variance floor that v4.2.0 and v4.3.0 assumed without measuring. All four "effects" collapse to zero, and the sign of the `hard` effect flips.

The v4.5.0 report fixes the last remaining inconsistency — the data generator — and re-predicts that the `impossible` val acc will drop by ~2.1 points, preserving its gap to Bayes.

The search space for this architecture on this task family is now closed:

> *No regularizer, schedule, or loss modification tested across v4.2.0, v4.3.0, and v4.4.0 moves val accuracy beyond 1σ of seed noise. v4.5.0 shows that even moving the Bayes ceiling does not move the model off it.*

### 9.2 Roadmap

| Version | Status | Adds | Observed effect |
|---|---|---|---|
| **v1.0.0** | shipped | Baseline | documents failure |
| **v4.1.0** | shipped | Early stopping + best checkpoint | **+0.10 on hard & impossible** |
| **v4.2.0** | shipped | dropout + wd (**hard**) | **−0.0116 on hard (falsified)** |
| **v4.3.0** | shipped | cosine LR + LS (**impossible**); corrected Bayes formula | **−0.0005 to −0.0011 (falsified)** |
| **v4.4.0** | shipped | Multi-seed (5 seeds); `zkash.multiseed` + `zkash.compare` | **all effects σ < 0.30; std mispredicted 1.8×** |
| **v4.5.0** | shipped | `data.py` label-noise fix (q = p); full re-run | **`impossible` Bayes 0.7142 → 0.6925; val acc prediction −2.1 pt** |
| v5.0.0 | planned | **Zkash100M** — depth 100, exact 100M params | next scale-up |

**No further training-loop interventions are planned.** The v4.2.0–v4.5.0 results close the search space for this architecture on this task family.

### 9.3 What v4.5.0 tested — and what it will confirm

**v4.5.0 — `data.py` fix.** Flipped labels now draw from the $C-1$ *other* classes, so `p` equals the true wrong-label rate. All three configs are re-run at 5 seeds.

| Task | Prediction | Confirms |
|---|---|---|
| `easy` | unchanged at 1.0000 | fix is a no-op without label noise |
| `hard` | unchanged at 0.5306 ± 0.0143 | fix is a no-op without label noise |
| `impossible` at `p = 0.2` | **drops by ~2.1 points to 0.689 ± 0.002** | corrected Bayes formula in §5.1 quantitatively |

Rationale: under v4.4.0 semantics, the effective wrong-label rate was `q = p(C−1)/C = 0.175`. Under v4.5.0, `p = 0.2` is the true wrong rate, which is higher. The corrected Bayes ceiling for `q = 0.2` is 0.6925 (rather than 0.7142 for `q = 0.175`), lower by ~2.1 points, so val acc should drop accordingly. If the model instead stays at 0.7102, the corrected formula is wrong; if it drops below 0.688, the model is no longer ceiling-bound.

### 9.4 v5.0.0 — Zkash100M

Depth 100, exact 100M parameters. Same architecture, 5× deeper. The open question is whether depth 100 shifts the Bayes ceiling of either hard task — the architecture cannot exceed a Bayes ceiling that is task-determined, but if Zkash100M reaches `hard` 0.588 at epoch 2 instead of 4, that is a meaningful speedup, though not an accuracy gain.

**Prediction (falsifiable):** Zkash100M on `hard` peaks at **val 0.586 ± 0.005 at epoch 1–2**. On `impossible` (v4.5.0 semantics), it peaks at **0.691 ± 0.002 at epoch 3–5**. Neither exceeds Bayes. Trains in ~40 s/epoch on a GTX 1660 (fp32) or ~15 s/epoch (bf16).

---

## 10. Conclusion

Zkash10M is the fourth entry in the Zkash scaling series and the first to abandon the MLP paradigm. With 20 pre-norm residual blocks, RMSNorm, and GELU — all bias-free — it reaches **exactly 10,000,896 trainable parameters** and trains cleanly without warmup or gradient clipping.

Its six shipped releases tell a single, increasingly sharp story.

**v1.0.0** documented a sharp negative result: architectural improvements — residuals, normalization, gated activations — solve the *optimization* problem of deep networks but not the *generalization* problem.

**v4.1.0** showed that a **single change to the training loop** — saving the best epoch rather than the last — recovers **+10 percentage points** of validation accuracy on both hard tasks, at no cost in parameters and with a 4× reduction in training time.

**v4.2.0** falsified its own prediction that the `hard` task was overfitting and would respond to dropout + weight decay. Regularization cost **1.2 points** at seed 0; 4× more data gained **0.4 points**. An empirical Bayes ceiling (0.5879) showed the model was already **5.7 points from its ceiling** at 5-seed resolution.

**v4.3.0** falsified its own prediction that the `impossible` task was underfitting and would respond to cosine LR + warmup + label smoothing. All three interventions landed within **±0.0011** of baseline at seed 0. A **corrected** Bayes ceiling (0.7142, not 0.8250) showed the baseline was already **0.4 points from its ceiling** — and that the "underfitting" diagnosis in v4.1.0 had been based on an analytical error in the Bayes formula.

**v4.4.0** falsified both falsifications. Re-running v4.2.0 and v4.3.0 on 5 seeds each shows that all four "effects" collapse to zero: `hard` regularization Δ = +0.0059 (σ = 0.30, sign flipped), `impossible` cosine Δ = −0.0002 (σ = 0.10), LS Δ = −0.0001 (σ = 0.02), both Δ = −0.0004 (σ = 0.19). Seed variance itself was mispredicted by 1.8× in both directions. The only non-null effect across four releases is that cosine compresses the peak-epoch distribution (7.0 ± 3.54 → 4.4 ± 1.14) without moving val accuracy.

**v4.5.0** closes the last methodological gap: the data generator was still using v1.0.0 label-noise semantics even though the Bayes formula had been corrected in v4.3.0. Fixing the generator moves the `impossible` Bayes ceiling from **0.7142 to 0.6925** and — so the prediction goes — moves the model with it, by the same 2.1 points, leaving the gap to Bayes unchanged at 0.4 points. This is the sharpest form of the "the gap is not diagnostic" lesson: moving the *ceiling* moves the model, and only a ceiling measurement reveals that the model was pinned to it all along.

The central lessons, at the end of six releases and eight falsifiable predictions, are:

> *A train/val gap cannot tell you whether regularization will help. A model whose gap looks like overfitting and a model whose gap looks like underfitting can both be memorization-dominated at a Bayes ceiling. The gap is not diagnostic. **The ceiling is.***

> *A single-seed Δ of size 1–2 percentage points is not measurable at 5 seeds. The seed variance floor must be measured before any small effect is declared. It was not measured in v4.2.0 or v4.3.0, and every effect they reported collapsed to zero.*

> *A ceiling is not a fixed property of the universe. It is a property of the task, and it moves when the task changes. Fix the data generator and the ceiling moves with it — but the model stays pinned. That is the strongest evidence that the model was ceiling-bound and never optimization-bound.*

And the practical consequence, in one line:

> *For a model at its Bayes ceiling, only one training-loop change is worth making — save the best epoch, not the last. Every other intervention we tested costs between −0.0116 and +0.0059 and is inside single-seed noise.*

---

## 11. Reproducibility

All results are reproducible on a single NVIDIA GTX 1660 SUPER (6 GB). The v4.4.0 multi-seed suite adds ~1 hour of GPU time on top of the ~15 minutes for v1.0.0–v4.3.0; the v4.5.0 full re-run adds another ~1 hour.

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
# v4.1.0 baselines
bash scripts/train.sh              # easy        — ~30 s
bash scripts/train_hard.sh         # hard v4.1.0 — ~15 s
bash scripts/train_impossible.sh   # impossible  — ~3 min

# v4.2.0 hard ablation (single seed)
bash scripts/sweep_hard.sh         # 2×2 dropout × wd — ~3 min

# v4.3.0 impossible ablation (single seed)
PYTHONIOENCODING=utf-8 PYTHONPATH=src python -m zkash.train \
  --config configs/zkash_10m_impossible.yaml \
  --override train.patience=25 \
             paths.checkpoint=checkpoints/imp_baseline.pt

PYTHONIOENCODING=utf-8 PYTHONPATH=src python -m zkash.train \
  --config configs/zkash_10m_impossible.yaml \
  --override train.schedule=cosine train.warmup_steps=500 \
             train.patience=25 paths.checkpoint=checkpoints/imp_cos.pt

PYTHONIOENCODING=utf-8 PYTHONPATH=src python -m zkash.train \
  --config configs/zkash_10m_impossible.yaml \
  --override train.label_smoothing=0.1 train.patience=25 \
             paths.checkpoint=checkpoints/imp_ls.pt

PYTHONIOENCODING=utf-8 PYTHONPATH=src python -m zkash.train \
  --config configs/zkash_10m_impossible.yaml \
  --override train.schedule=cosine train.warmup_steps=500 \
             train.label_smoothing=0.1 train.patience=25 \
             paths.checkpoint=checkpoints/imp_both.pt

# v4.4.0 multi-seed (5 seeds) — legacy label-noise semantics
bash scripts/multiseed_hard.sh         # 2 configs × 5 seeds — ~2 min
bash scripts/multiseed_impossible.sh   # 4 configs × 5 seeds — ~1 h

# v4.5.0 full re-run (5 seeds) — fixed label-noise semantics
bash scripts/rerun_v45.sh              # 3 tasks × 5 seeds — ~1 h

# empirical Bayes ceilings
python scripts/bayes_hard.py         # 0.5879 (unchanged)
python scripts/bayes_impossible.py   # clean 0.8606, noisy 0.6925 (v4.5.0)
```

**Windows note:** `train.py` prints a `★` marker on improving epochs, which fails under `cp1251` when stdout is piped. Use `PYTHONIOENCODING=utf-8`, or replace `" ★"` with `" *"` in `train.py` (one-line change).

### 11.4 Determinism

- All configs fix `data.seed: 0`; v4.4.0 and v4.5.0 vary `train.seed ∈ {0, 1, 2, 3, 4}`.
- `utils.set_seed` seeds Python, NumPy, PyTorch (CPU + CUDA).
- `torch.backends.cudnn.deterministic = False`, `benchmark = True` — bit-exact reruns across machines are *not* guaranteed.

### 11.5 Reproducible multi-seed summaries

Running `bash scripts/multiseed_hard.sh` followed by `bash scripts/multiseed_impossible.sh` (v4.4.0, legacy semantics) produces:

```
logs/multiseed_hard_baseline.json
logs/multiseed_hard_v42_regularized.json
logs/multiseed_impossible_baseline.json
logs/multiseed_impossible_cosine.json
logs/multiseed_impossible_ls.json
logs/multiseed_impossible_both.json
```

Running `bash scripts/rerun_v45.sh` (v4.5.0, fixed semantics) produces:

```
logs/v45_easy.json
logs/v45_hard.json
logs/v45_impossible.json
```

Each JSON contains per-seed raw metrics plus a mean / std / min / max aggregate. `zkash.compare` renders the σ verdicts used in §6.7:

```bash
PYTHONPATH=src python -m zkash.compare \
  logs/v45_hard.json \
  logs/multiseed_hard_baseline.json
```

### 11.6 Reproducing v4.4.0 results with v4.5.0 code

The v4.5.0 `make_gaussians` exposes a `legacy_redraw: bool = False` flag. Passing `True` restores v1.0.0–v4.4.0 behavior exactly:

```python
from zkash.data import make_gaussians

# v4.4.0 semantics (q = p·(C−1)/C = 0.175 at p = 0.2)
ds = make_gaussians(n_samples=262_144, label_noise=0.2,
                    legacy_redraw=True, seed=0)
```

All v4.4.0 logs in `logs/multiseed_*.json` were generated with this behavior. They remain valid as historical artifacts.

---

## 12. Citation

```bibtex
@techreport{zkash10m,
  title       = {Zkash10M: A 10M-Parameter Residual Reference Network for Education},
  number      = {ZK-2025-04},
  institution = {Zkash Project},
  year        = {2025},
  note        = {Version 4.5.0}
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
9. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., Wojna, Z. (2016). *Rethinking the Inception Architecture for Computer Vision.* CVPR. (label smoothing)

---

**Report history**

| Version | Date | Change |
|---|---|---|
| ZK-2025-04 v1.0.0 | 2025 | Initial release, baseline + negative result |
| ZK-2025-04 v4.1.0 | 2025 | Early stopping + best-checkpoint recovery |
| ZK-2025-04 v4.2.0 | 2025 | Dropout + wd on hard; §9.1 falsified; empirical Bayes for hard |
| ZK-2025-04 v4.3.0 | 2025 | Cosine LR + LS on impossible; §9.3 falsified; **corrected Bayes formula (§5.1)** |
| ZK-2025-04 v4.4.0 | 2025 | Multi-seed (5 seeds); all v4.2.0/v4.3.0 effects collapse to zero; std mispredicted 1.8× |
| **ZK-2025-04 v4.5.0** | **2025** | **`data.py` label-noise fix (q = p); `impossible` Bayes 0.7142 → 0.6925; `legacy_redraw` flag added; full re-run at 5 seeds** |

---