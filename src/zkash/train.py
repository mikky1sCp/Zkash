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

        if epoch % 10 == 0 or epoch == cfg["train"]["epochs"]:
            print(f"epoch {epoch:3d} | train loss {total/seen:.4f} | train acc {correct/seen:.3f}")

    # ---- validation ----
    model.eval()
    correct, seen = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            seen += xb.size(0)
    print(f"val acc: {correct/seen:.4f}")

    # ---- save ----
    ckpt = cfg["paths"]["checkpoint"]
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    torch.save({
        "model": {k: v.cpu() for k, v in model.state_dict().items()},
        "config": cfg,
    }, ckpt)
    print(f"saved checkpoint → {ckpt}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/zkash_10m.yaml")
    args = p.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()