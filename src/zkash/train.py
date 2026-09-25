import argparse
import os
import warnings
import torch
import torch.nn as nn
from tqdm import tqdm

from .model import Zkash01M as Model
from .data import make_gaussians, make_loader, split_dataset
from .utils import count_params, set_seed, load_config

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")


def train(cfg: dict) -> None:
    set_seed(cfg["train"]["seed"])

    model = Model(
        n_in=cfg["model"]["n_in"],
        n_out=cfg["model"]["n_out"],
        p_drop=cfg["model"].get("p_drop", 0.0),
    )
    n = count_params(model)
    print(f"[Zkash-0.1M] trainable params: {n}")
    assert n == 100_000, f"expected 100000 params, got {n}"

    # ---- data ----
    ds = make_gaussians(
        n_samples=cfg["data"]["n_samples"],
        n_classes=cfg["data"]["n_classes"],
        dim=cfg["model"]["n_in"],
        noise=cfg["data"]["noise"],
        seed=cfg["data"]["seed"],
    )
    train_ds, val_ds = split_dataset(ds, val_frac=0.2, seed=cfg["train"]["seed"])
    train_loader = make_loader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = make_loader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

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
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            seen += xb.size(0)
    print(f"val acc: {correct/seen:.4f}")

    # ---- save ----
    ckpt = cfg["paths"]["checkpoint"]
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg}, ckpt)
    print(f"saved checkpoint → {ckpt}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/zkash_01m.yaml")
    args = p.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
