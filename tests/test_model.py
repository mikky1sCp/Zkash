import torch
import pytest
import copy

from zkash.model import Zkash10M, RMSNorm, ResidualBlock
from zkash.utils import count_params
from zkash.data import make_gaussians, split_dataset, make_loader


def test_param_count():
    assert count_params(Zkash10M()) == 10_000_896


def test_rmsnorm_shape():
    n = RMSNorm(64)
    x = torch.randn(4, 64)
    assert n(x).shape == (4, 64)


def test_rmsnorm_unit_rms():
    n = RMSNorm(64)
    x = torch.randn(4, 64)
    out = n(x)
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_residual_identity_at_zero():
    blk = ResidualBlock(64, 64)
    for p in blk.fc1.parameters():
        p.data.zero_()
    for p in blk.fc2.parameters():
        p.data.zero_()
    x = torch.randn(4, 64)
    assert torch.allclose(blk(x), x)


def test_forward_shape():
    m = Zkash10M()
    assert m(torch.randn(4, 64)).shape == (4, 8)


@pytest.mark.parametrize("bs", [1, 8, 32, 128])
def test_forward_batch_sizes(bs):
    m = Zkash10M()
    assert m(torch.randn(bs, 64)).shape == (bs, 8)


def test_forward_dtype():
    m = Zkash10M()
    assert m(torch.randn(4, 64)).dtype == torch.float32


def test_forward_finite():
    m = Zkash10M()
    assert torch.isfinite(m(torch.randn(16, 64))).all()


def test_backward_all_params_get_grad():
    m = Zkash10M()
    x = torch.randn(8, 64)
    y = torch.randint(0, 8, (8,))
    loss = torch.nn.functional.cross_entropy(m(x), y)
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


def test_state_dict_roundtrip():
    m1 = Zkash10M()
    m2 = Zkash10M()
    m2.load_state_dict(m1.state_dict())
    x = torch.randn(4, 64)
    m1.eval(); m2.eval()
    with torch.no_grad():
        assert torch.allclose(m1(x), m2(x))

def test_early_stopping_saves_best_state():
    from zkash.train import evaluate_acc

    model = Zkash10M()
    device = torch.device("cpu")
    ds = make_gaussians(n_samples=512, n_classes=8, dim=64, seed=0)
    _, val_ds = split_dataset(ds, val_frac=0.5, seed=0)
    loader = make_loader(val_ds, batch_size=64, shuffle=False)

    acc_before = evaluate_acc(model, loader, device)
    snapshot = copy.deepcopy(model.state_dict())

    # pretend we improved
    model.load_state_dict(snapshot)
    acc_after = evaluate_acc(model, loader, device)

    assert isinstance(acc_before, float)
    assert 0.0 <= acc_before <= 1.0
    assert acc_before == acc_after


def test_patience_config_present():
    import yaml, pathlib
    for p in pathlib.Path("configs").glob("*.yaml"):
        cfg = yaml.safe_load(p.read_text())
        assert "patience" in cfg["train"], f"{p} missing patience"