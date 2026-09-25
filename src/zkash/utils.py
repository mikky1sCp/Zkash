import random
import numpy as np
import torch
import torch.nn as nn


def count_params(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int) -> None:
    """Seed everything for reproducibility (CPU + GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Deterministic cuDNN — reproducible but slightly slower.
        # Flip benchmark to True if raw speed matters more than bit-exactness.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer: str = "auto") -> torch.device:
    """Pick a torch device.

    prefer:
        'auto'  → cuda if available, else cpu
        'cpu'   → always cpu
        'cuda'  → cuda (falls back to cpu if unavailable)
        'cuda:0', 'cuda:1', ... → specific GPU
    """
    if prefer == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if prefer.startswith("cuda") and not torch.cuda.is_available():
        print("[utils] CUDA requested but not available — falling back to CPU")
        return torch.device("cpu")

    return torch.device(prefer)


def load_config(path: str) -> dict:
    """Load a YAML config file into a dict."""
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)
