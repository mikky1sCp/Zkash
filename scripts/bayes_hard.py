# scripts/bayes_hard.py
import torch

torch.manual_seed(0)
C, dim, N = 8, 64, 200_000
center_scale, noise = 0.3, 1.5

g = torch.Generator().manual_seed(0)
centers = torch.randn(C, dim, generator=g) * center_scale

# Monte Carlo: sample, classify by nearest centroid
y = torch.randint(0, C, (N,), generator=g)
X = centers[y] + noise * torch.randn(N, dim, generator=g)

# nearest centroid in one shot
d2 = (X[:, None, :] - centers[None, :, :]).pow(2).sum(-1)
pred = d2.argmin(1)

print(f"Bayes (nearest-centroid): {(pred == y).float().mean():.4f}")