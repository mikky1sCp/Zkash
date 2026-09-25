import torch
from zkash.model import Zkash10K, Zkash01M
from zkash.utils import count_params


def test_param_count_10k():
    assert count_params(Zkash10K()) == 10_000


def test_param_count_01m():
    assert count_params(Zkash01M()) == 100_000


def test_forward_shape_10k():
    m = Zkash10K()
    y = m(torch.randn(4, 64))
    assert y.shape == (4, 8)


def test_forward_shape_01m():
    m = Zkash01M()
    y = m(torch.randn(4, 64))
    assert y.shape == (4, 8)
