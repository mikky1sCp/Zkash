import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square normalization (Zhang & Sennrich, 2019)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * rms


class ResidualBlock(nn.Module):
    """Pre-norm residual MLP: x + W2 · GELU(W1 · RMSNorm(x))."""
    def __init__(self, dim: int = 512, hidden: int = 486):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.fc2(h)
        return x + h


class Zkash10M(nn.Module):
    """
    Zkash10M: ~10M-parameter deep residual network.

    Param budget (exact):
        stem:       Linear(64, 512, bias=False)      =    32,768
        block ×20:  RMSNorm(512) + 2×Linear(…, bias=False)
                                                     =   498,176 each
        final_norm: RMSNorm(512)                     =       512
        head:       Linear(512, 8, bias=False)       =     4,096
        ------------------------------------------------
        total                                        = 10,000,896
    """
    def __init__(self,
                 n_in: int = 64,
                 n_out: int = 8,
                 dim: int = 512,
                 hidden: int = 486,
                 depth: int = 20):
        super().__init__()
        self.stem = nn.Linear(n_in, dim, bias=False)
        self.blocks = nn.ModuleList(
            [ResidualBlock(dim, hidden) for _ in range(depth)]
        )
        self.final_norm = RMSNorm(dim)
        self.head = nn.Linear(dim, n_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.final_norm(x)
        return self.head(x)