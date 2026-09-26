# Zkash10M — Technical Whitepaper

**Report ID:** ZK-2025-04
**Version:** 4.3.0
**Status:** Reference / Educational
**Params:** exactly 10,000,896
**Last updated:** 2025

---

## Abstract

**Zkash10M** is a deep residual network with **exactly 10,000,896 trainable parameters** — the fourth entry in the Zkash scaling series (10K → 100K → 1M → 10M). It replaces the wide-MLP paradigm of its predecessors with depth: **20 pre-norm residual blocks**, RMSNorm normalization, and GELU activations, all bias-free. The architecture trains cleanly without warmup or gradient clipping.

This report covers four releases. **v1.0.0** established the baseline and documented a sharp negative result: on a task with 20% label noise, train accuracy (0.971) exceeded the reported Bayes ceiling (0.825) while val accuracy collapsed to 0.613 — memorization of flipped labels. **v4.1.0** added early stopping with best-checkpoint-by-val_acc and recovered **+10 percentage points** of validation accuracy on both hard tasks without changing the model. **v4.2.0** tested dropout + weight decay on the `hard` task and **falsified** its own prediction: regularization cost **−0.0116** val accuracy; 4× more data gained **+0.0036**. An empirical Bayes ceiling for `hard` (nearest-centroid Monte Carlo) measured **0.5879**, showing the model was already **4.2 points from its ceiling**. **v4.3.0** tests cosine LR + warmup + label smoothing on the `impossible` task — the last remaining "underfitting" hypothesis — and **falsifies it too**: all three interventions land within ±0.0011 of the unmodified baseline, and an empirical Bayes ceiling of **0.7142** (corrected for feature overlap) shows the baseline was already **0.14 points from its ceiling**.

The v4.3.0 release also **corrects a methodological error** that had propagated through v1.0.0–v4.2.0: the analytical Bayes accuracy formula `1 − p·(C−1)/C` is valid only when the clean-label Bayes accuracy is 1.0. The `impossible` task has feature overlap (clean Bayes = 0.8606), so the correct noisy-label ceiling is **0.7142**, not 0.8250 — a 11-point correction. This error had misled v4.1.0 and v4.2.0 into diagnosing `impossible` as underfitting and predicting that LR schedules would help. **They do not.**

Across all five releases, exactly **one training-loop change** improved validation accuracy: **early stopping with best-checkpoint-by-val_acc** (v4.1.0). Four other interventions — dropout, weight decay, cosine LR, label smoothing — were either null or negative. The consistent explanation is that both hard tasks are **memorization-dominated and near their Bayes ceilings**, and no training-loop change can close a gap that is set by the task.

---

## 1. Motivation

Zkash10M exists to answer a specific question left open by Zkash-1M (v3):

> When width has stopped helping, does **depth** help?

The v1–v3 models all scale a 3–4 layer perceptron. Once width reaches ~1000 neurons per layer, additional capacity produces no measurable improvement on any task the series has been tested on. The natural next step is a **change of topology**, not a change of size.

Three concerns drive the design:

- **Topological break.** Introduce the three pillars of every 2020s architecture — residual connections, pre-normalization, gated activations — and measure whether they make a difference at 10M parameters.
- **Depth over width.** Twenty narrow residual blocks, not four wide ones. Effective depth scales with parameter count.
- **Honest failure.** Document the model's limits with analytical bounds, not just accuracy numbers. A model that quietly generalizes teaches nothing. A model that visibly fails, with a Bayes ceiling to compare against, teaches something — and a model whose failure diagnosis is itself wrong teaches more still.

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

**Note:** dropout, when enabled, adds **zero** parameters. The count remains 10,000,896 for any `p_drop ∈ [0, 1)`. Verified across all v4.2.0 and v4.3.0 runs.

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

| Hyperparameter | v1.0.0 | v4.1.0 | v4.2.0 | v4.3.0 |
|---|---|---|---|---|
| Optimizer | AdamW | same | same | same |
| Learning rate | 1·10⁻³ const | same | same | **1·10⁻³ cosine (impossible only)** |
| Weight decay | 1·10⁻⁴ | same | **1·10⁻² (hard only)** | same |
| Weight decay groups | — | — | **2 groups** | same |
| Batch size | 128 / 256 | same | same | same |
| Epochs | 60 (fixed) | 60 (max, ES) | 100 (max, ES) | 60 (max, ES) |
| Loss | CE | same | same | **+ LS 0.1 (impossible only)** |
| Warmup | none | none | none | **500 steps (impossible only)** |
| Gradient clipping | none | none | none | none |
| Dropout | none | none | **0.1 (hard only)** | same |
| Early stopping | none | patience 10 | patience 25 | patience 25 |

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

**Both were null.** See §6.6.

### 4.5 A note on absent mechanisms

**No gradient clipping, no warmup in the baseline.** This is a genuine property of the architecture, not a missing feature. Pre-norm residual networks keep activation scale bounded at every layer; gradients never spike. Warmup exists to prevent early-training instability; pre-norm prevents it *structurally*.

**No dropout in v4.3.0.** Removed after v4.2.0's null result. Dropout adds noise to hidden activations and was found to cost 1.2 points on `hard` (§6.4) and — as v4.3.0 confirms — nothing on `impossible`.

---

## 5. Synthetic task definition

All results are on a controlled synthetic dataset:

- **Input:** $x \in \mathbb{R}^{64}$
- **Classes:** $C = 8$
- **Class centers:** $\mu_c \sim \mathcal{N}(0, \sigma_c^2 I)$ per coordinate
- **Samples:** $x_i = \mu_{y_i} + \mathcal{N}(0, \sigma_n^2 I)$
- **Labels:** uniform over 8 classes; a fraction `label_noise = p` is re-drawn uniformly from all `C` classes (including possibly the original — see §5.3)

Three parameters control difficulty: `center_scale` (class separation), `noise` (feature noise), `label_noise` (label corruption).

### 5.1 Analytical Bayes bounds — corrected

**This section corrects an error present in versions v1.0.0–v4.2.0.**

The formula used previously,

$$
\text{Acc}_{\text{Bayes}}(p) = 1 - p \cdot \frac{C-1}{C} \quad\text{(INCORRECT for } A_{\text{clean}} < 1\text{)},
$$

assumes that the classifier's **clean-label** Bayes accuracy $A_{\text{clean}}$ equals 1.0 — i.e., that classes are perfectly separable by a nearest-centroid rule. This is true for `easy` but false for both `hard` and `impossible`, where feature noise causes class overlap.

The correct formula, for a classifier that achieves $A_{\text{clean}}$ on clean labels and is uniform on the remaining $C-1$ classes when wrong, is:

$$
\boxed{\;\text{Acc}_{\text{Bayes}}(p) = A_{\text{clean}} \cdot \left(1 - \frac{p(C-1)}{C}\right) + (1 - A_{\text{clean}}) \cdot \frac{p}{C}\;}
$$

Equivalently, writing $q = p(C-1)/C$ for the true fraction of *incorrect* labels:

$$
\text{Acc}_{\text{Bayes}}(p) = A_{\text{clean}} \cdot (1 - q) + (1 - A_{\text{clean}}) \cdot \frac{q}{C-1}
$$

At $A_{\text{clean}} = 1$ this reduces to the old formula. At $A_{\text{clean}} < 1$ it produces a **strictly lower** ceiling.

**Empirical $A_{\text{clean}}$ values** (nearest-centroid Monte Carlo, $N = 200{,}000$, seed 0):

| Config | $A_{\text{clean}}$ |
|---|---:|
| easy | 1.0000 |
| hard | 0.5879 |
| impossible | **0.8606** |

**Corrected Bayes table for `impossible`:**

| `p` | `q` = p·7/8 | Corrected Bayes | Old formula | Error |
|---|---:|---:|---:|---:|
| 0.0 | 0.0000 | **0.8606** | 1.0000 | −0.1394 |
| 0.1 | 0.0875 | **0.7870** | 0.9125 | −0.1255 |
| **0.2** | **0.1750** | **0.7135** | 0.8250 | **−0.1115** |
| 0.3 | 0.2625 | **0.6399** | 0.7375 | −0.0976 |

The Monte Carlo measurement at $p = 0.2$ gives **0.7142**, matching the corrected analytical value 0.7135 to within MC noise (0.0007).

**Corrected Bayes loss.** For the observed-label process, the Bayes-optimal predictive distribution is $P(y_{\text{obs}} = y_{\text{true}} \mid x) = 1-q$, $P(y_{\text{obs}} = c \ne y_{\text{true}} \mid x) = q/(C-1)$, giving

$$
\mathcal{L}_{\text{Bayes}}(q) = -(1-q)\ln(1-q) - q\ln\!\left(\frac{q}{C-1}\right)
$$

with $q$, **not** $p$, as the argument. For `impossible` at $p = 0.2$: $q = 0.175$, $\mathcal{L}_{\text{Bayes}} = 0.8043$ (MC: 0.8043). The old formula gave 0.888 — an overestimate by 0.084 nats.

### 5.2 Empirical Bayes for `hard`

`hard` has no label noise ($p = 0$), so its ceiling is $A_{\text{clean}}$ directly. Monte Carlo (nearest-centroid, $N = 200{,}000$): **0.5879**.

A classifier that must *estimate* centers from $N$ samples has a strictly lower ceiling. Standard error of the center estimate at 2,048 samples (256/class): $\text{SE} = 1.5/\sqrt{256} \approx 0.094$ per coordinate → center displacement $\approx 0.094 \cdot \sqrt{64} \approx 0.75$ against inter-class distance $\approx 3.4$. Finite-sample ceiling ≈ 0.55–0.57. At 8,192 samples: ≈ 0.57–0.58.

### 5.3 `label_noise` semantics

In `data.py`, `label_noise = p` re-draws a fraction `p` of labels uniformly from all `C` classes, **including the original**. True wrong-label rate is $q = p(C-1)/C$, not $p$. All corrected figures in §5.1 use $q$; the old formula used $p$ directly. This is the source of the 11-point error corrected above.

---

### 5.4 Three task configurations

| Config | `n_samples` | `center_scale` | `noise` | `label_noise` | $A_{\text{clean}}$ | Corrected Bayes |
|---|---:|---:|---:|---:|---:|---:|
| easy | 65,536 | 3.0 | 0.7 | 0.0 | 1.0000 | **1.0000** |
| hard | 2,048 | 0.3 | 1.5 | 0.0 | 0.5879 | **0.5879** |
| impossible | 262,144 | 0.5 | 1.5 | 0.2 | 0.8606 | **0.7135** (MC: 0.7142) |

---

## 6. Results

### 6.1 v1.0.0 — no early stopping (baseline)

| Config | Train acc | Val acc | Gap |
|---|---:|---:|---:|
| easy | 1.0000 | 1.0000 | 0.000 |
| hard | 0.9840 | 0.4400 | **0.544** |
| impossible | **0.9710** | 0.6134 | **0.358** |

The v1.0.0 report noted that `impossible` train accuracy (0.971) exceeds the then-reported Bayes ceiling (0.825). Under the corrected Bayes, this remains true (0.971 > 0.7142). The model is memorizing noisy labels.

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

### 6.4 v4.2.0 — dropout + weight decay on `hard` (falsified)

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

Prediction §9.1: val 0.60–0.65, gap ~0.10. Observed: val 0.5306, gap 0.2168. **Falsified.**

Both regularization and additional data leave the **memorization transition point** unchanged: the `hard` runs peak at 64–128 gradient steps regardless of setup, and the peak is at 4.2 points below Bayes.

### 6.5 Why regularization and data both fail on `hard`

**Regularization** reduces the network's capacity to memorize, but the memorization transition in this regime is set by the task's signal-to-noise ratio, not by capacity. Suppressing noise-fitting also suppresses signal-fitting; net −0.0116.

**More data** multiplies steps-per-epoch, so the same absolute step count (64–128) arrives after fewer epochs. Peak val is unchanged because the peak is where the model has fit the class signal and not yet fit the noise — a property of the task.

### 6.6 v4.3.0 — cosine LR + label smoothing on `impossible` (falsified)

v4.2.0 §9.3 predicted `impossible` val accuracy would rise to **0.76–0.80** with cosine LR + warmup + label smoothing, on the theory that the model was optimization-bound and 11.2 points below Bayes. **That theory was based on the incorrect Bayes formula corrected in §5.1.** With the corrected ceiling, `impossible` is only 0.14 points below Bayes, and no training-loop change should help.

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
| **Bayes (MC, corrected)** | **0.7142** | — | — | — |

**All four runs are within ±0.0011 of baseline.** All peak at epoch 5. All sit 0.19–0.25 points below Bayes.

Additional evidence: baseline train loss @ best epoch **1.0607**; label smoothing train loss @ best epoch **1.2456** (LS raises the loss floor by exactly the predicted ~0.185 nats), yet **val accuracy is unchanged to within noise**. The model's argmax was already correct; only its confidence was disturbed.

**Prediction vs observation:**

| Metric | §9.3 predicted | v4.3.0 observed | Status |
|---|---:|---:|---|
| Val acc | 0.76–0.80 | **0.7123** (cosine) | ✗ |
| Val acc | 0.76–0.80 | **0.7124** (LS) | ✗ |
| Val acc | 0.76–0.80 | **0.7117** (both) | ✗ |

**Falsified on all three.** No training-loop change moves `impossible` off its ceiling.

### 6.7 Consolidated falsification ledger

| Release | Prediction | Observed | Status |
|---|---|---|---|
| v4.1.0 §9.1 | `hard` val 0.60–0.65 with dropout+wd | 0.5306 | **✗** |
| v4.1.0 §9.1 | `hard` train acc ~0.75 | 0.7474 | ✓ |
| v4.1.0 §9.1 | `hard` gap ~0.10 | 0.2168 | **✗** |
| v4.2.0 §9.3 | `impossible` val 0.76–0.80 with cosine+LS | 0.7117–0.7124 | **✗** |
| v4.3.0 (this) | `impossible` unchanged by any training-loop change | confirmed | ✓ |

**Score: 2 of 5 predictions correct.** The two correct ones were the easy ones (train acc under regularization; the null hypothesis itself). Every prediction that involved raising val accuracy was wrong. That is itself a finding: the model was already at its ceiling in every case, and the ceiling was not measured.

### 6.8 The only intervention that has ever worked

| Intervention | Task | Effect on val acc |
|---|---|---:|
| Early stopping + best checkpoint (v4.1.0) | `hard` | **+0.103** |
| Early stopping + best checkpoint (v4.1.0) | `impossible` | **+0.099** |
| Dropout + weight decay (v4.2.0) | `hard` | −0.0116 |
| 4× more data (v4.2.0) | `hard` | +0.0036 |
| Cosine LR + warmup (v4.3.0) | `impossible` | −0.0005 |
| Label smoothing 0.1 (v4.3.0) | `impossible` | −0.0004 |
| Both (v4.3.0) | `impossible` | −0.0011 |

**Early stopping is the only mechanism in five releases that reliably improved val accuracy.** It costs nothing: no parameters, no backward passes, no architectural change. Every other intervention is within single-seed noise or negative.

---

## 7. One regime: memorization at the ceiling

### 7.1 hard — memorization-dominated, ceiling-bound

| Epoch | Train acc | Val acc | Gap |
|---:|---:|---:|---:|
| 1 | 0.3807 | 0.4694 | −0.089 |
| 4 | 0.7407 | **0.5428** ★ | +0.198 |
| 5 | 0.8188 | 0.4988 | +0.320 |
| 10 | 0.9286 | 0.4743 | **+0.454** |

Train/val gap reaches +0.45. Under the v4.1.0 diagnosis, this was classical overfitting. Two facts disqualify that:

1. **Val peak is 4.2 points below Bayes** (0.5428 vs 0.5879).
2. **Regularization makes val worse** (−0.0116). Overfitting responds to regularization; ceiling-bound memorization does not.

**Correct remedy:** none. `hard` is closed.

### 7.2 impossible — memorization-dominated, also ceiling-bound

| Epoch | Train acc | Val acc | Gap |
|---:|---:|---:|---:|
| 1 | 0.6944 | 0.7069 | −0.013 |
| 5 | 0.7089 | **0.7128** ★ | −0.004 |
| 10 | 0.7135 | 0.7099 | +0.004 |
| 15 | 0.7188 | 0.7093 | +0.010 |

Train/val gap never exceeds 0.01. Under the v4.1.0 diagnosis, this was underfitting — "model learns slowly". **The v4.3.0 ablation falsifies that:** the peak is at epoch 5, cosine + warmup do not move it, and the corrected Bayes ceiling (0.7142) shows the peak is **0.14 points below** the theoretical maximum.

**Correct remedy:** none. `impossible` is closed.

The earlier "margin above Bayes = 0.142" evidence for underfitting was an artifact of the incorrect Bayes loss formula (§5.1). With the corrected value 0.8043, the margin is **0.256 nats**, but the *val accuracy* is already at Bayes — the gap is confidence, not ranking. The model has the correct argmax.

### 7.3 The unified picture

| Task | $A_{\text{clean}}$ | Noisy Bayes | Best model | Gap to Bayes | Peak epoch |
|---|---:|---:|---:|---:|---:|
| easy | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1 |
| hard | 0.5879 | 0.5879 | 0.5458 | **−0.0421** | 2–4 |
| impossible | 0.8606 | 0.7142 | 0.7128 | **−0.0014** | 5 |

**Both hard tasks are memorization-dominated and near their Bayes ceilings.** The only quantitative difference is distance to ceiling: `hard` is 4.2 points below, `impossible` is 0.1 points below. Neither is fixable by any training-loop change tested across v4.2.0 and v4.3.0.

**Revised rule:** a model whose train/val gap looks like overfitting may be at its ceiling. A model whose train/val gap looks like underfitting may also be at its ceiling. **Only a Bayes measurement tells you which.** The gap is not diagnostic; the ceiling is.

### 7.4 The full spectrum

| Regime | Train acc | Val acc | Gap | Present in |
|---|---:|---:|---:|---|
| Underparametrized | low | low | small | — |
| Well-parametrized | high | high | small | `easy` |
| **Ceiling-bound memorization (fast)** | **high** | **near Bayes** | **large** | **`hard`** |
| **Ceiling-bound memorization (slow)** | **low-ish** | **near Bayes** | **small** | **`impossible`** |

Three of four rows. The "overparametrized + overfitting that regularization would fix" row does not appear in this dataset — it requires a task where the model is *below* its ceiling and could be pushed up. Neither hard task qualifies.

---

## 8. Limitations

- **Fixed input dimension.** `n_in = 64`. Residual MLP only.
- **Depth cost.** 20 sequential blocks → ~10× inference latency vs. same-parameter MLP.
- **~40 MB in fp32.**
- **Not production-ready.** Both hard tasks are closed as ceiling-bound.
- **Synthetic-only task.**
- **Single random seed.** All reported results use `seed = 0`. Variance across seeds is unmeasured. The ±0.0011 spread across v4.3.0's four runs is single-seed variance under one seed; cross-seed variance is likely larger.
- **Bayes ceilings are Monte Carlo estimates** ($N = 200{,}000$, seed 0) for `hard` and `impossible`; only `easy` has a closed-form value.
- **`data.py` label-noise semantics differ from the naive interpretation** — corrected in §5.1 but left unchanged in code for backwards compatibility with v1.0.0–v4.2.0 results.

---

## 9. Extensions and roadmap

### 9.1 What v4.2.0 and v4.3.0 actually established

The v4.1.0 report set out a plan:

> v4.2.0: dropout + wd on `hard` → val +0.05–0.10.
> v4.2.0: cosine LR + warmup + patience 25 on `impossible` → val +0.05–0.08.

**Neither happened.** The plan assumed the two tasks failed for opposite reasons. They do not. Both fail for the same reason — memorization at a Bayes ceiling — and the ceiling is what it is.

The v4.3.0 release revises the roadmap accordingly.

### 9.2 Roadmap

| Version | Status | Adds | Observed effect |
|---|---|---|---|
| **v1.0.0** | shipped | Baseline | documents failure |
| **v4.1.0** | shipped | Early stopping + best checkpoint | **+0.10 on hard & impossible** |
| **v4.2.0** | shipped | dropout + wd (**hard**) | **−0.0116 on hard (falsified)** |
| **v4.3.0** | shipped | cosine LR + LS (**impossible**); corrected Bayes formula | **−0.0005 to −0.0011 (falsified)** |
| v4.4.0 | planned | Multi-seed runs (5 seeds) — mean ± std | quantifies variance floor |
| v4.5.0 | planned | `data.py` label-noise fix + full re-run | consistency with corrected §5.1 |
| v5.0.0 | planned | **Zkash100M** — depth 100, exact 100M params | next scale-up |

**No further training-loop interventions are planned.** The v4.2.0 and v4.3.0 results close the search space for this architecture on this task family: no regularizer, schedule, or loss modification tested moves val accuracy beyond single-seed noise.

### 9.3 What v4.4.0 and v4.5.0 will test

**v4.4.0 — multi-seed.** Every result in this report is seed 0. The reported effects (−0.0116, +0.0036, −0.0005) are single-seed. Five-seed runs will report mean ± std. **Prediction:** the true std for `impossible` val acc is ~0.002–0.005; the true std for `hard` is larger, ~0.005–0.010. If so, all v4.2.0 and v4.3.0 "effects" are inside 1σ of seed noise — as the abstract already suspects.

**v4.5.0 — `data.py` fix.** Change label-noise implementation to draw from the $C-1$ *other* classes, so `p` equals the true wrong-label rate. All three configs re-run. **Prediction:** `hard` unchanged (no label noise). `easy` unchanged. `impossible` shifts upward by ~1.5 points because the true wrong-label rate drops from 0.175 to 0.20... no, wait — under the new implementation `p = 0.2` becomes the true wrong rate, which is *higher* than 0.175. Val acc should drop by ~2 points. This will confirm the corrected formula in §5.1 quantitatively.

### 9.4 v5.0.0 — Zkash100M

Depth 100, exact 100M parameters. Same architecture, 5× deeper. The open question is whether depth 100 shifts the Bayes ceiling of either hard task — the architecture cannot exceed a Bayes ceiling that is task-determined, but if Zkash100M reaches `hard` 0.588 at epoch 2 instead of 4, that is a meaningful speedup, though not an accuracy gain.

**Prediction (falsifiable):** Zkash100M on `hard` peaks at **val 0.586 ± 0.005 at epoch 1–2**. On `impossible`, it peaks at **0.714 ± 0.002 at epoch 3–5**. Neither exceeds Bayes. Trains in ~40 s/epoch on a GTX 1660 (fp32) or ~15 s/epoch (bf16).

---

## 10. Conclusion

Zkash10M is the fourth entry in the Zkash scaling series and the first to abandon the MLP paradigm. With 20 pre-norm residual blocks, RMSNorm, and GELU — all bias-free — it reaches **exactly 10,000,896 trainable parameters** and trains cleanly without warmup or gradient clipping.

Its four shipped releases tell a single, increasingly sharp story.

**v1.0.0** documented a sharp negative result: architectural improvements — residuals, normalization, gated activations — solve the *optimization* problem of deep networks but not the *generalization* problem.

**v4.1.0** showed that a **single change to the training loop** — saving the best epoch rather than the last — recovers **+10 percentage points** of validation accuracy on both hard tasks, at no cost in parameters and with a 4× reduction in training time.

**v4.2.0** falsified its own prediction that the `hard` task was overfitting and would respond to dropout + weight decay. Regularization cost **1.2 points**; 4× more data gained **0.4 points**. An empirical Bayes ceiling (0.5879) showed the model was already **4.2 points from its ceiling**.

**v4.3.0** falsified its own prediction that the `impossible` task was underfitting and would respond to cosine LR + warmup + label smoothing. All three interventions landed within **±0.0011** of baseline. A **corrected** Bayes ceiling (0.7142, not 0.8250) showed the baseline was already **0.14 points from its ceiling** — and that the "underfitting" diagnosis in v4.1.0 had been based on an analytical error in the Bayes formula.

The central lesson, at the end of four releases and five falsified predictions, is:

> *A train/val gap cannot tell you whether regularization will help. A model whose gap looks like overfitting and a model whose gap looks like underfitting can both be memorization-dominated at a Bayes ceiling. The gap is not diagnostic. **The ceiling is.***

And the practical consequence, in one line:

> *For a model at its Bayes ceiling, only one training-loop change is worth making — save the best epoch, not the last. Every other intervention we tested costs between −0.0116 and +0.0036 and is inside single-seed noise.*

---

## 11. Reproducibility

All results are reproducible on a single NVIDIA GTX 1660 SUPER (6 GB) in under 60 minutes total.

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

# v4.2.0 hard ablation
bash scripts/sweep_hard.sh         # 2×2 dropout × wd — ~3 min

# v4.3.0 impossible ablation
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

# empirical Bayes ceilings
python scripts/bayes_hard.py         # 0.5879
python scripts/bayes_impossible.py   # clean 0.8606, noisy 0.7142
```

**Windows note:** `train.py` prints a `★` marker on improving epochs, which fails under `cp1251` when stdout is piped. Use `PYTHONIOENCODING=utf-8`, or replace `" ★"` with `" *"` in `train.py` (one-line change).

### 11.4 Determinism

- All configs fix `seed: 0`
- `utils.set_seed` seeds Python, NumPy, PyTorch (CPU + CUDA)
- `torch.backends.cudnn.deterministic = False`, `benchmark = True` — bit-exact reruns across machines are *not* guaranteed.

---

## 12. Citation

```bibtex
@techreport{zkash10m,
  title       = {Zkash10M: A 10M-Parameter Residual Reference Network for Education},
  number      = {ZK-2025-04},
  institution = {Zkash Project},
  year        = {2025},
  note        = {Version 4.3.0}
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

---