import torch
import pytest

from zkash.model import Zkash10M, RMSNorm, ResidualBlock
from zkash.train import build_param_groups
from zkash.utils import count_params

EXPECTED_PARAMS = 10_000_896

def test_two_groups_returned():
    groups = build_param_groups(Zkash10M(), weight_decay=1e-2)
    assert len(groups) == 2
    assert groups[0]["weight_decay"] == 1e-2
    assert groups[1]["weight_decay"] == 0.0


def test_no_parameter_appears_twice():
    groups = build_param_groups(Zkash10M(), weight_decay=1e-2)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)), "a parameter is in both groups"


def test_group_union_covers_all_trainable_params():
    m = Zkash10M()
    groups = build_param_groups(m, weight_decay=1e-2)
    grouped = sum(p.numel() for g in groups for p in g["params"])
    assert grouped == count_params(m) == EXPECTED_PARAMS


def test_rmsnorm_params_are_not_decayed():
    """The whole point of the split: RMSNorm.γ must NOT be weight-decayed."""
    m = Zkash10M()
    groups = build_param_groups(m, weight_decay=1e-2)
    decay_ids = {id(p) for p in groups[0]["params"]}
    no_decay_ids = {id(p) for p in groups[1]["params"]}

    for name, p in m.named_parameters():
        if name.endswith("norm.weight") or name.endswith("final_norm.weight"):
            assert id(p) in no_decay_ids, f"{name} should not be decayed"
            assert id(p) not in decay_ids


def test_linear_weights_are_decayed():
    m = Zkash10M()
    groups = build_param_groups(m, weight_decay=1e-2)
    decay_ids = {id(p) for p in groups[0]["params"]}

    for name, p in m.named_parameters():
        if name.endswith(".weight") and p.ndim == 2:
            assert id(p) in decay_ids, f"{name} should be decayed"


def test_decay_group_size_matches_budget():
    """20 blocks × 2 linears + stem + head = 42 matrices."""
    groups = build_param_groups(Zkash10M(), weight_decay=1e-2)
    assert len(groups[0]["params"]) == 42


def test_no_decay_group_size_matches_norm_count():
    """20 block norms + 1 final norm = 21 vectors."""
    groups = build_param_groups(Zkash10M(), weight_decay=1e-2)
    assert len(groups[1]["params"]) == 21

def test_weight_decay_value_propagates():
    for wd in (0.0, 1e-4, 1e-2, 1e-1):
        groups = build_param_groups(Zkash10M(), weight_decay=wd)
        assert groups[0]["weight_decay"] == wd
        assert groups[1]["weight_decay"] == 0.0


def test_frozen_params_are_excluded():
    m = Zkash10M()
    m.stem.weight.requires_grad = False
    groups = build_param_groups(m, weight_decay=1e-2)
    all_ids = {id(p) for g in groups for p in g["params"]}
    assert id(m.stem.weight) not in all_ids


def test_zeroed_block_stays_identity_with_groups():
    blk = ResidualBlock(64, 64)
    for p in list(blk.fc1.parameters()) + list(blk.fc2.parameters()):
        p.data.zero_()
    x = torch.randn(4, 64)
    assert torch.allclose(blk(x), x)

def test_adamw_accepts_groups_and_steps():
    m = Zkash10M()
    opt = torch.optim.AdamW(
        build_param_groups(m, weight_decay=1e-2),
        lr=1e-3,
    )
    x = torch.randn(8, 64)
    y = torch.randint(0, 8, (8,))
    loss = torch.nn.functional.cross_entropy(m(x), y)
    loss.backward()
    opt.step()

    for name, p in m.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


def test_norm_weight_actually_untouched_by_decay():
    m = Zkash10M()
    opt = torch.optim.AdamW(
        build_param_groups(m, weight_decay=1e-1),   # aggressive on purpose
        lr=1e-3,
    )
    before = m.blocks[0].norm.weight.detach().clone()

    # zero every grad so only weight decay could move params
    for p in m.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()

    after = m.blocks[0].norm.weight.detach()
    assert torch.allclose(before, after), "norm weight moved under zero grad"