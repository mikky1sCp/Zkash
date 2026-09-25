import torch
import pytest

from zkash.model import Zkash10K, Zkash01M, Zkash1M, build_model
from zkash.utils import count_params

def test_param_count_10k():
    assert count_params(Zkash10K()) == 10_000


def test_param_count_01m():
    assert count_params(Zkash01M()) == 100_000


def test_param_count_1m():
    assert count_params(Zkash1M()) == 1_000_000

@pytest.mark.parametrize("ModelCls", [Zkash10K, Zkash01M, Zkash1M])
def test_forward_shape(ModelCls):
    m = ModelCls()
    x = torch.randn(4, 64)
    y = m(x)
    assert y.shape == (4, 8)

@pytest.mark.parametrize("ModelCls", [Zkash10K, Zkash01M, Zkash1M])
def test_forward_batch_sizes(ModelCls):
    m = ModelCls()
    for bs in (1, 8, 32):
        y = m(torch.randn(bs, 64))
        assert y.shape == (bs, 8)


@pytest.mark.parametrize("ModelCls", [Zkash10K, Zkash01M, Zkash1M])
def test_forward_dtype(ModelCls):
    m = ModelCls()
    y = m(torch.randn(4, 64))
    assert y.dtype == torch.float32

@pytest.mark.parametrize("ModelCls", [Zkash10K, Zkash01M, Zkash1M])
def test_forward_finite(ModelCls):
    m = ModelCls()
    y = m(torch.randn(16, 64))
    assert torch.isfinite(y).all()

def test_dropout_train_eval_differs():
    m = Zkash1M(p_drop=0.5)
    x = torch.randn(8, 64)

    m.train()
    y1 = m(x)
    y2 = m(x)
    assert not torch.allclose(y1, y2), "dropout should randomize in train mode"

    m.eval()
    y3 = m(x)
    y4 = m(x)
    assert torch.allclose(y3, y4), "eval mode must be deterministic"


def test_no_dropout_deterministic():
    m = Zkash1M(p_drop=0.0)
    x = torch.randn(8, 64)

    m.train()
    y1 = m(x)
    y2 = m(x)
    assert torch.allclose(y1, y2)

def test_build_model_by_name():
    assert count_params(build_model("zkash_10k")) == 10_000
    assert count_params(build_model("zkash_01m")) == 100_000
    assert count_params(build_model("zkash_1m"))  == 1_000_000


def test_build_model_unknown_raises():
    with pytest.raises(ValueError, match="unknown model"):
        build_model("zkash_999m")


def test_build_model_kwargs():
    m = build_model("zkash_1m", p_drop=0.3)
    assert isinstance(m, Zkash1M)

@pytest.mark.parametrize("ModelCls", [Zkash10K, Zkash01M, Zkash1M])
def test_backward_all_params_get_grad(ModelCls):
    m = ModelCls()
    x = torch.randn(8, 64)
    y = torch.randint(0, 8, (8,))

    logits = m(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()

    for name, p in m.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"

@pytest.mark.parametrize("ModelCls", [Zkash10K, Zkash01M, Zkash1M])
def test_state_dict_roundtrip(ModelCls):
    m1 = ModelCls()
    m2 = ModelCls()
    m2.load_state_dict(m1.state_dict())

    x = torch.randn(4, 64)
    m1.eval(); m2.eval()
    with torch.no_grad():
        assert torch.allclose(m1(x), m2(x))
