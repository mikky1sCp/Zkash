import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


def make_gaussians(n_samples: int = 65536,
                   n_classes: int = 8,
                   dim: int = 64,
                   noise: float = 0.7,
                   center_scale: float = 3.0,
                   label_noise: float = 0.0,
                   legacy_redraw: bool = False,
                   seed: int = 0) -> TensorDataset:
    """Synthetic dataset: `n_classes` Gaussian clouds in R^dim.

    Args:
        n_samples:     number of samples
        n_classes:     number of classes
        dim:           feature dimension
        noise:         std of Gaussian noise added to each sample
        center_scale:  std of class centers (smaller -> harder task)
        label_noise:   fraction of labels replaced by a *different* class
                       (intrinsic Bayes error). In v4.5.0 semantics
                       (default), this equals the true wrong-label rate q.
                       See `legacy_redraw`.
        legacy_redraw: if True, reproduce the v1.0.0-v4.4.0 implementation,
                       which re-drew flipped labels uniformly over ALL C
                       classes (including the original), so the effective
                       wrong-label rate was q = p*(C-1)/C. Default False.
        seed:          RNG seed

    Note (v4.5.0):
        Default behavior draws flips from the C-1 *other* classes, so
        `label_noise = p` is the true wrong rate q. Pass
        `legacy_redraw=True` to reproduce v1.0.0-v4.4.0 results.
        See WHITEPAPER sections 5.1, 5.3 and 4.6.
    """
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_classes, dim, generator=g) * center_scale
    y = torch.randint(0, n_classes, (n_samples,), generator=g)
    X = centers[y] + noise * torch.randn(n_samples, dim, generator=g)

    if label_noise > 0:
        flip_mask = torch.rand(n_samples, generator=g) < label_noise
        n_flip = int(flip_mask.sum().item())
        if n_flip > 0:
            if legacy_redraw:
                # v1.0.0-v4.4.0: uniform over all C classes.
                # Effective wrong-label rate q = p*(C-1)/C.
                y[flip_mask] = torch.randint(
                    0, n_classes, (n_flip,), generator=g
                )
            else:
                # v4.5.0+: uniform over the C-1 *other* classes.
                # True wrong-label rate q = p.
                offsets = torch.randint(1, n_classes, (n_flip,), generator=g)
                y[flip_mask] = (y[flip_mask] + offsets) % n_classes

    return TensorDataset(X, y)


def split_dataset(ds: TensorDataset,
                  val_frac: float = 0.2,
                  seed: int = 0):
    n_val = int(len(ds) * val_frac)
    n_train = len(ds) - n_val
    g = torch.Generator().manual_seed(seed)
    return random_split(ds, [n_train, n_val], generator=g)


def make_loader(dataset, batch_size: int = 128,
                shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)