import argparse
import math
import os
import warnings
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import yaml

from .model import Zkash10M
from .data import make_gaussians, make_loader, split_dataset
from .utils import count_params, set_seed, load_config, get_device

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")

EXPECTED_PARAMS = 10_000_896


@torch.no_grad()
def evaluate_acc(model, loader, device) -> float:
    """Return classification accuracy on a loader."""
    model.eval()
    correct, seen = 0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item()
        seen += xb.size(0)
    return correct / seen


def build_param_groups(model, weight_decay: float):
    """Split params into decay / no-decay groups.

    Norm weights (RMSNorm.γ) and any 1-D param (biases, if ever added)
    are excluded from weight decay — standard practice in GPT/LLaMA.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    return [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _apply_override(cfg: dict, kv: str) -> None:
    key, _, val = kv.partition("=")
    node = cfg
    parts = key.split(".")
    for k in parts[:-1]:
        node = node[k]
    try:
        node[parts[-1]] = yaml.safe_load(val)
    except Exception:
        node[parts[-1]] = val


def train(cfg: dict) -> None:
    set_seed(cfg["train"]["seed"])
    device = get_device(cfg["train"].get("device", "auto"))
    print(f"device: {device}")

    model = Zkash10M(
        n_in=cfg["model"]["n_in"],
        n_out=cfg["model"]["n_out"],
        dim=cfg["model"]["dim"],
        hidden=cfg["model"]["hidden"],
        depth=cfg["model"]["depth"],
        p_drop=cfg["model"].get("p_drop", 0.0),
    ).to(device)

    n = count_params(model)
    print(f"[Zkash10M] trainable params: {n}")
    assert n == EXPECTED_PARAMS, f"expected {EXPECTED_PARAMS}, got {n}"

    # ---- data ----
    ds = make_gaussians(
        n_samples=cfg["data"]["n_samples"],
        n_classes=cfg["data"]["n_classes"],
        dim=cfg["model"]["n_in"],
        noise=cfg["data"]["noise"],
        center_scale=cfg["data"].get("center_scale", 3.0),
        label_noise=cfg["data"].get("label_noise", 0.0),
        seed=cfg["data"]["seed"],
    )
    train_ds, val_ds = split_dataset(ds, val_frac=0.2, seed=cfg["train"]["seed"])
    train_loader = make_loader(train_ds, cfg["train"]["batch_size"], shuffle=True)
    val_loader = make_loader(val_ds, cfg["train"]["batch_size"], shuffle=False)

    # ---- optim ----
    opt = torch.optim.AdamW(
        build_param_groups(model, cfg["train"]["weight_decay"]),
        lr=cfg["train"]["lr"],
    )

    # ---- label smoothing ----
    ls = cfg["train"].get("label_smoothing", 0.0)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=ls)

    # ---- scheduler ----
    schedule = cfg["train"].get("schedule", "constant")
    if schedule == "cosine":
        steps_per_epoch = len(train_loader)
        total_steps = cfg["train"]["epochs"] * steps_per_epoch
        warmup = cfg["train"].get("warmup_steps", 0)
        min_ratio = cfg["train"].get("min_lr_ratio", 0.0)

        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            p = (step - warmup) / max(1, total_steps - warmup)
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * p))

        scheduler = LambdaLR(opt, lr_lambda)
    else:
        scheduler = None

    # ---- run header ----
    print(f"[Zkash10M] p_drop={cfg['model'].get('p_drop', 0.0)}  "
          f"weight_decay={cfg['train']['weight_decay']}")
    decay_n = sum(p.numel() for p in build_param_groups(
        model, cfg["train"]["weight_decay"])[0]["params"])
    nodecay_n = sum(p.numel() for p in build_param_groups(
        model, cfg["train"]["weight_decay"])[1]["params"])
    print(f"[optim] decay-group:     {decay_n:>10,} params "
          f"(wd={cfg['train']['weight_decay']})")
    print(f"[optim] no-decay-group:  {nodecay_n:>10,} params (wd=0)")
    if schedule != "constant":
        print(f"[optim] schedule={schedule}  warmup_steps="
              f"{cfg['train'].get('warmup_steps', 0)}")
    if ls > 0:
        print(f"[loss]  label_smoothing={ls}")

    # ---- early stopping state ----
    patience = cfg["train"].get("patience", 10)
    min_delta = cfg["train"].get("min_delta", 0.0)
    best_val_acc = -1.0
    best_train_acc = 0.0
    best_train_loss = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0

    # ---- loop ----
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        total, correct, seen = 0.0, 0, 0
        bar = tqdm(train_loader, desc=f"epoch {epoch:3d}", leave=False)

        for xb, yb in bar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            if scheduler is not None:
                scheduler.step()

            total += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            seen += xb.size(0)
            bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total / seen
        train_acc = correct / seen
        val_acc = evaluate_acc(model, val_loader, device)

        # ---- best-checkpoint logic ----
        improved = val_acc > best_val_acc + min_delta
        if improved:
            best_val_acc = val_acc
            best_train_acc = train_acc
            best_train_loss = train_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        marker = " *" if improved else ""
        if improved or epoch % 5 == 0 or epoch == cfg["train"]["epochs"]:
            print(f"epoch {epoch:3d} | train loss {train_loss:.4f} | "
                  f"train acc {train_acc:.4f} | val acc {val_acc:.4f}{marker}")

        if wait >= patience:
            print(f"early stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs)")
            break

    # ---- restore best & save ----
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\nbest val acc: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"best train acc @ best epoch: {best_train_acc:.4f}")
    print(f"best train loss @ best epoch: {best_train_loss:.4f}")
    print(f"gap (train-val) @ best epoch: "
          f"{best_train_acc - best_val_acc:+.4f}")

    ckpt = cfg["paths"]["checkpoint"]
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    torch.save({
        "model": best_state,
        "config": cfg,
        "best_val_acc": best_val_acc,
        "best_train_acc": best_train_acc,
        "best_train_loss": best_train_loss,
        "best_epoch": best_epoch,
    }, ckpt)
    print(f"saved best checkpoint -> {ckpt}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/zkash_10m.yaml")
    p.add_argument("--override", nargs="*", default=[])
    args = p.parse_args()

    cfg = load_config(args.config)
    for kv in args.override:
        _apply_override(cfg, kv)
    train(cfg)


if __name__ == "__main__":
    main()