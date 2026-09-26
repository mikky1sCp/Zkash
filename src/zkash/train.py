import argparse
import os
import warnings
import torch
import torch.nn as nn
from tqdm import tqdm

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
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    loss_fn = nn.CrossEntropyLoss()

    # ---- early stopping state ----
    patience = cfg["train"].get("patience", 10)
    min_delta = cfg["train"].get("min_delta", 0.0)
    best_val_acc = -1.0
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
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        marker = " ★" if improved else ""
        # print on every improvement, every 5 epochs, or last epoch
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

    ckpt = cfg["paths"]["checkpoint"]
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    torch.save({
        "model": best_state,
        "config": cfg,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
    }, ckpt)
    print(f"saved best checkpoint → {ckpt}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/zkash_10m.yaml")
    args = p.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()