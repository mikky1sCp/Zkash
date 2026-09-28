"""Empirical Bayes for the `impossible` config.

Uses the TRUE class centers (nearest-centroid), which no trained
model can beat because it never sees them. Reports BOTH:
  - clean-accuracy: argmax over P(y | x) before label noise
  - noisy-accuracy: argmax over P(y_obs | x) after label noise
The second is the true ceiling for a classifier trained on noisy labels.
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
acc_clean = (y_pred == y_true).float().mean().item()
print(f"Bayes (clean labels):      {acc_clean:.4f}")

# --- noisy Bayes: labels re-drawn from all C classes with prob p ---
y_obs = y_true.clone()
mask = torch.rand(N, generator=g) < label_noise
y_obs[mask] = torch.randint(0, C, (int(mask.sum()),), generator=g)
acc_noisy = (y_pred == y_obs).float().mean().item()
print(f"Bayes (label_noise=0.2):   {acc_noisy:.4f}")

# --- Bayes loss for the observed-label process ---
q = label_noise * (C - 1) / C          # true wrong rate
loss_bayes = -(1 - q) * torch.log(torch.tensor(1 - q)) \
             - q * torch.log(torch.tensor(q / (C - 1)))
print(f"True wrong-label rate q:   {q:.4f}")
print(f"Bayes loss (corrected):    {loss_bayes.item():.4f}")
print(f"Bayes loss (WHITEPAPER §5.1 formula): "
      f"{-(1-label_noise)*torch.log(torch.tensor(1-label_noise)) - label_noise*torch.log(torch.tensor(label_noise/(C-1))):.4f}")