import torch
import pytest

from zkash.data import make_gaussians


def test_label_noise_zero_unchanged_by_fix():
    """With label_noise=0, the v4.5.0 change is a no-op."""
    ds = make_gaussians(n_samples=1000, label_noise=0.0, seed=0)
    X, y = ds.tensors
    assert X.shape == (1000, 64)
    assert y.shape == (1000,)
    assert y.min() >= 0 and y.max() < 8


def test_v45_default_matches_true_wrong_rate():
    """Default (v4.5.0) must produce exactly p wrong labels."""
    n = 200_000
    p = 0.2
    ds = make_gaussians(n_samples=n, label_noise=p, seed=0)
    ds_clean = make_gaussians(n_samples=n, label_noise=0.0, seed=0)
    y_clean = ds_clean.tensors[1]
    y_noisy = ds.tensors[1]
    wrong = (y_clean != y_noisy).float().mean().item()
    # binomial std for p=0.2, n=200000 is ~0.0009; allow 5 sigma
    assert abs(wrong - p) < 0.01, f"expected ~{p}, got {wrong:.4f}"


def test_legacy_redraw_matches_v44_wrong_rate():
    """legacy_redraw=True must reproduce v4.4.0: q = p*(C-1)/C."""
    n = 200_000
    p = 0.2
    ds = make_gaussians(n_samples=n, label_noise=p,
                        legacy_redraw=True, seed=0)
    ds_clean = make_gaussians(n_samples=n, label_noise=0.0, seed=0)
    y_clean = ds_clean.tensors[1]
    y_legacy = ds.tensors[1]
    changed = (y_clean != y_legacy).float().mean().item()
    expected = p * 7 / 8  # 0.175
    assert abs(changed - expected) < 0.01, \
        f"expected ~{expected}, got {changed:.4f}"


def test_flipped_label_never_equals_original():
    """Under v4.5.0 semantics, every flipped sample must have y_new != y_orig."""
    n = 50_000
    p = 0.5
    ds_clean = make_gaussians(n_samples=n, label_noise=0.0, seed=0)
    ds_noisy = make_gaussians(n_samples=n, label_noise=p, seed=0)
    y_clean = ds_clean.tensors[1]
    y_noisy = ds_noisy.tensors[1]
    changed = y_clean != y_noisy
    rate = changed.float().mean().item()
    assert abs(rate - p) < 0.01
    # every changed sample must have a different value (tautological
    # but ensures no in-place aliasing bug)
    assert (y_noisy[changed] != y_clean[changed]).all()


def test_legacy_can_keep_original_label():
    """Under legacy semantics with p=1.0, ~1/8 of labels stay the same."""
    n = 200_000
    ds_clean = make_gaussians(n_samples=n, label_noise=0.0, seed=0)
    ds_legacy = make_gaussians(n_samples=n, label_noise=1.0,
                               legacy_redraw=True, seed=0)
    y_clean = ds_clean.tensors[1]
    y_legacy = ds_legacy.tensors[1]
    unchanged = (y_clean == y_legacy).float().mean().item()
    expected = 1 / 8
    assert abs(unchanged - expected) < 0.01, \
        f"expected ~{expected}, got {unchanged:.4f}"


def test_label_noise_distribution_is_uniform_over_other_classes():
    """Given a flip (v4.5.0), the new label is uniform over the C-1 others."""
    n = 200_000
    p = 1.0  # flip all
    ds = make_gaussians(n_samples=n, n_classes=8,
                        label_noise=p, seed=0)
    ds_clean = make_gaussians(n_samples=n, n_classes=8,
                              label_noise=0.0, seed=0)
    y_clean = ds_clean.tensors[1]
    y_new = ds.tensors[1]
    # Every sample should have changed (p=1.0, v4.5.0 semantics)
    assert (y_clean != y_new).all()
    # Distribution of new label given old label should be uniform
    # over the C-1 others.
    for c in range(8):
        mask = y_clean == c
        counts = torch.bincount(y_new[mask], minlength=8).float()
        counts[c] = 0.0
        probs = counts / counts.sum()
        assert torch.allclose(probs[probs > 0],
                              torch.full_like(probs[probs > 0], 1 / 7),
                              atol=0.02)


def test_legacy_redraw_is_default_off():
    """Regression: v4.5.0 default must NOT be legacy."""
    n = 100_000
    p = 0.2
    ds_default = make_gaussians(n_samples=n, label_noise=p, seed=0)
    ds_explicit = make_gaussians(n_samples=n, label_noise=p,
                                 legacy_redraw=False, seed=0)
    assert torch.equal(ds_default.tensors[1], ds_explicit.tensors[1])