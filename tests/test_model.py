import torch
from zkash.model import Zkash10K
from zkash.utils import count_params


def test_param_count():
    m = Zkash10K()
    assert count_params(m) == 10_000


def test_forward_shape():
    m = Zkash10K(n_in=64, n_out=8)
    x = torch.randn(4, 64)
    y = m(x)
    assert y.shape == (4, 8)


def test_no_bias_in_output():
    m = Zkash10K()
    assert m.fc3.bias is None