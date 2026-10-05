"""Empirical Bayes for the `impossible` config (v4.5.0 semantics).

Uses the TRUE class centers (nearest-centroid), which no trained
model can beat because it never sees them. Reports:

  - clean accuracy: argmax over P(y | x) before label noise
  - noisy accuracy: argmax over P(y_obs | x) after label noise
                    (this is the true ceiling for a classifier
                     trained on noisy labels)
  - analytical corrected Bayes for the v4.5.0 regime (q = p)

v4.5.0: label noise draws from the C-1 *other* classes, so
`label_noise = p` equals the true wrong-label rate q. The corrected
formula is

    Acc_Bayes(q) = A_clean * (1 - q) + (1 - A_clean) * q / (C - 1)

with q = p (v4.5.0) rather than q = p*(C-1)/C (v1.0.0-v4.4.0).

Expected output:
    Bayes (clean labels):      0.8606
    Bayes (label_noise=0.2):   0.6925
    Analytical (q = p):        0.6925
    True wrong-label rate q:   0.2000
    Bayes loss (corrected):    0.8896
"""
import torch

torch.manual_seed(0)
C, dim, N = 8, 64, 200_000
center_scale, noise, label_noise = 0.5, 1.5, 0.2

g = torch.Generator().manual_seed(0)
centers = torch.randn(C, dim, generator=g) * center_scale

y_true = torch.randint(0, C, (N,), generator=g)
X = centers[y_true] + noise * torch.randn(N, dim, generator=g)

# --- clean Bayes: nearest centroid on true labels (no label noise yet) ---
d2 = (X[:, None, :] - centers[None, :, :]).pow(2).sum(-1)
y_pred = d2.argmin(1)
A_clean = (y_pred == y_true).float().mean().item()
print(f"Bayes (clean labels):      {A_clean:.4f}")

# --- noisy labels: draw from C-1 *other* classes with prob p ---
y_obs = y_true.clone()
mask = torch.rand(N, generator=g) < label_noise
n_flip = int(mask.sum().item())
if n_flip > 0:
    offsets = torch.randint(1, C, (n_flip,), generator=g)
    y_obs[mask] = (y_true[mask] + offsets) % C

acc_noisy = (y_pred == y_obs).float().mean().item()
print(f"Bayes (label_noise=0.2):   {acc_noisy:.4f}")

# --- analytical corrected Bayes: q = p (v4.5.0 semantics) ---
q = label_noise
acc_analytical = A_clean * (1 - q) + (1 - A_clean) * q / (C - 1)
print(f"Analytical (q = p):        {acc_analytical:.4f}")

# --- Bayes loss for the observed-label process ---
loss_bayes = -(1 - q) * torch.log(torch.tensor(1 - q)) \
             - q * torch.log(torch.tensor(q / (C - 1)))
print(f"True wrong-label rate q:   {q:.4f}")
print(f"Bayes loss (corrected):    {loss_bayes.item():.4f}")