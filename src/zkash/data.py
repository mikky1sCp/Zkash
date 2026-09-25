import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


def make_gaussians(n_samples: int = 4096,
                   n_classes: int = 8,
                   dim: int = 64,
                   noise: float = 0.7,
                   seed: int = 0) -> TensorDataset:
    """Synthetic dataset: `n_classes` Gaussian clouds in R^dim."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_classes, dim, generator=g) * 3.0
    y = torch.randint(0, n_classes, (n_samples,), generator=g)
    X = centers[y] + noise * torch.randn(n_samples, dim, generator=g)
    return TensorDataset(X, y)


def split_dataset(ds: TensorDataset,
                  val_frac: float = 0.2,
                  seed: int = 0) -> tuple[TensorDataset, TensorDataset]:
    n_val = int(len(ds) * val_frac)
    n_train = len(ds) - n_val
    g = torch.Generator().manual_seed(seed)
    return random_split(ds, [n_train, n_val], generator=g)


def make_loader(dataset, batch_size: int = 64, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
