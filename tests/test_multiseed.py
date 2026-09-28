import math
import pytest

from zkash.multiseed import aggregate, format_summary, _stats, run_seeds
from zkash.compare import format_comparison, welch_sigma


def _fake_result(seed, val, train, epoch=3):
    return {
        "seed": seed,
        "best_val_acc": val,
        "best_train_acc": train,
        "best_train_loss": 0.5,
        "best_epoch": epoch,
        "params": 10_000_896,
        "checkpoint": f"ckpt_seed{seed}.pt",
    }


def test_stats_single_value():
    s = _stats([0.5])
    assert s["mean"] == 0.5
    assert s["std"] == 0.0
    assert s["n"] == 1


def test_stats_multiple_values():
    s = _stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["mean"] == 3.0
    assert s["min"] == 1.0
    assert s["max"] == 5.0
    assert s["n"] == 5


def test_stats_std_is_sample_stdev():
    s = _stats([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    assert s["std"] == pytest.approx(2.138, abs=0.001)


def test_aggregate_produces_all_keys():
    results = [_fake_result(i, 0.5 + 0.01 * i, 0.6) for i in range(5)]
    agg = aggregate(results)
    for key in ("val_acc", "train_acc", "train_loss", "gap", "epoch"):
        assert key in agg
    assert agg["val_acc"]["n"] == 5
    assert agg["seeds"] == [0, 1, 2, 3, 4]


def test_aggregate_rejects_empty():
    with pytest.raises(ValueError):
        aggregate([])


def test_aggregate_gap_sign():
    agg = aggregate([_fake_result(0, 0.5, 0.7)])
    assert agg["gap"]["mean"] == pytest.approx(0.2)


def test_aggregate_gap_negative():
    agg = aggregate([_fake_result(0, 0.7, 0.5)])
    assert agg["gap"]["mean"] == pytest.approx(-0.2)


def test_welch_sigma_identical():
    a = {"aggregate": {"val_acc": {"mean": 0.5, "std": 0.01}}}
    b = {"aggregate": {"val_acc": {"mean": 0.5, "std": 0.01}}}
    assert welch_sigma(a, b, "val_acc") == 0.0


def test_welch_sigma_zero_std():
    a = {"aggregate": {"val_acc": {"mean": 0.5, "std": 0.0}}}
    b = {"aggregate": {"val_acc": {"mean": 0.6, "std": 0.0}}}
    assert welch_sigma(a, b, "val_acc") == float("inf")


def test_welch_sigma_correct_value():
    a = {"aggregate": {"val_acc": {"mean": 0.50, "std": 0.01}}}
    b = {"aggregate": {"val_acc": {"mean": 0.52, "std": 0.01}}}
    assert welch_sigma(a, b, "val_acc") == pytest.approx(1.4142, abs=0.001)


def test_format_summary_contains_key_fields():
    results = [_fake_result(i, 0.5 + 0.01 * i, 0.6) for i in range(5)]
    text = format_summary("mytest", aggregate(results))
    assert "mytest" in text and "val acc" in text and "gap" in text


def _report(name, mean, std):
    return {"config": name,
            "aggregate": {"val_acc": {"mean": mean, "std": std,
                                      "min": mean - std, "max": mean + std,
                                      "values": [], "n": 5}}}


def test_format_comparison_marks_baseline_and_1sigma():
    text = format_comparison([_report("base.yaml", 0.50, 0.01),
                              _report("other.yaml", 0.501, 0.01)])
    assert "baseline" in text and "within 1σ" in text


def test_format_comparison_marks_2sigma():
    text = format_comparison([_report("base.yaml", 0.50, 0.01),
                              _report("mid.yaml", 0.52, 0.01)])
    assert "within 2σ" in text


def test_format_comparison_marks_significant():
    text = format_comparison([_report("base.yaml", 0.50, 0.005),
                              _report("far.yaml", 0.60, 0.005)])
    assert "SIGNIFICANT" in text


def test_run_seeds_calls_train_per_seed(monkeypatch):
    from zkash import multiseed
    calls = []

    def fake_train(cfg):
        seed = cfg["train"]["seed"]
        calls.append(seed)
        return {"best_val_acc": 0.5 + 0.01 * seed, "best_train_acc": 0.6,
                "best_train_loss": 0.5, "best_epoch": 3,
                "params": 10_000_896,
                "checkpoint": cfg["paths"]["checkpoint"]}

    monkeypatch.setattr(multiseed, "train", fake_train)
    cfg = {"train": {"seed": 0}, "paths": {"checkpoint": "ckpt.pt"}}
    results = multiseed.run_seeds(cfg, [0, 1, 2], verbose=False)

    assert calls == [0, 1, 2]
    assert [r["seed"] for r in results] == [0, 1, 2]


def test_run_seeds_suffixes_checkpoint(monkeypatch):
    from zkash import multiseed
    seen = []

    def fake_train(cfg):
        seen.append(cfg["paths"]["checkpoint"])
        return {"best_val_acc": 0.5, "best_train_acc": 0.5,
                "best_train_loss": 0.5, "best_epoch": 1,
                "params": 10_000_896, "checkpoint": cfg["paths"]["checkpoint"]}

    monkeypatch.setattr(multiseed, "train", fake_train)
    cfg = {"train": {"seed": 0}, "paths": {"checkpoint": "ckpts/model.pt"}}
    multiseed.run_seeds(cfg, [7, 8], verbose=False)
    assert seen == ["ckpts/model_seed7.pt", "ckpts/model_seed8.pt"]


def test_run_seeds_does_not_mutate_base_cfg(monkeypatch):
    from zkash import multiseed
    monkeypatch.setattr(multiseed, "train", lambda cfg: {
        "best_val_acc": 0.5, "best_train_acc": 0.5, "best_train_loss": 0.5,
        "best_epoch": 1, "params": 10_000_896,
        "checkpoint": cfg["paths"]["checkpoint"]})
    cfg = {"train": {"seed": 0}, "paths": {"checkpoint": "m.pt"}}
    multiseed.run_seeds(cfg, [1, 2], verbose=False)
    assert cfg["train"]["seed"] == 0
    assert cfg["paths"]["checkpoint"] == "m.pt"