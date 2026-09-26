import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


def make_gaussians(n_samples: int = 65536,
                   n_classes: int = 8,
                   dim: int = 64,
                   noise: float = 0.7,
                   center_scale: float = 3.0,
                   label_noise: float = 0.0,
                   seed: int = 0) -> TensorDataset:
    """Synthetic dataset: `n_classes` Gaussian clouds in R^dim.

    Args:
        n_samples:    number of samples
        n_classes:    number of classes
        dim:          feature dimension
        noise:        std of Gaussian noise added to each sample
        center_scale: std of class centers (smaller → harder task)
        label_noise:  fraction of labels randomly flipped (intrinsic Bayes error)
        seed:         RNG seed
    """
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_classes, dim, generator=g) * center_scale
    y = torch.randint(0, n_classes, (n_samples,), generator=g)
    X = centers[y] + noise * torch.randn(n_samples, dim, generator=g)

    if label_noise > 0:
        flip_mask = torch.rand(n_samples, generator=g) < label_noise
        n_flip = int(flip_mask.sum().item())
        y[flip_mask] = torch.randint(0, n_classes, (n_flip,), generator=g)

    return TensorDataset(X, y)


def split_dataset(ds: TensorDataset,
                  val_frac: float = 0.2,
                  seed: int = 0):
    n_val = int(len(ds) * val_frac)
    n_train = len(ds) - n_val
    g = torch.Generator().manual_seed(seed)
    return random_split(ds, [n_train, n_val], generator=g)


def make_loader(dataset, batch_size: int = 128, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)