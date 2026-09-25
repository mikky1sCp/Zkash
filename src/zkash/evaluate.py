import argparse
import warnings
import torch

from .model import build_model
from .data import make_gaussians, split_dataset, make_loader
from .utils import count_params, load_config, get_device

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")


@torch.no_grad()
def evaluate(cfg: dict, ckpt_path: str) -> None:
    device = get_device(cfg["train"].get("device", "auto"))
    print(f"device: {device}")

    model = build_model(
        cfg["model"]["name"],
        n_in=cfg["model"]["n_in"],
        n_out=cfg["model"]["n_out"],
        p_drop=cfg["model"].get("p_drop", 0.0),
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.to(device).eval()

    ds = make_gaussians(
        n_samples=cfg["data"]["n_samples"],
        n_classes=cfg["data"]["n_classes"],
        dim=cfg["model"]["n_in"],
        noise=cfg["data"]["noise"],
        seed=cfg["data"]["seed"],
    )
    _, val_ds = split_dataset(ds, val_frac=0.2, seed=cfg["train"]["seed"])
    val_loader = make_loader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    correct, seen = 0, 0
    for xb, yb in val_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item()
        seen += xb.size(0)

    acc = correct / seen
    print(f"params : {count_params(model)}")
    print(f"samples: {seen}")
    print(f"acc    : {acc:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/zkash_1m.yaml")
    p.add_argument("--ckpt", default="checkpoints/zkash1m.pt")
    args = p.parse_args()
    evaluate(load_config(args.config), args.ckpt)


if __name__ == "__main__":
    main()
