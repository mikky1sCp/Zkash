import torch
import torch.nn as nn
import torch.nn.functional as F


class Zkash10K(nn.Module):
    """Zkash-10K (v1.0.0): exactly 10,000 trainable parameters."""
    def __init__(self, n_in: int = 64, n_out: int = 8,
                 hidden1: int = 64, hidden2: int = 80):
        super().__init__()
        self.fc1 = nn.Linear(n_in, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, n_out, bias=False)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Zkash01M(nn.Module):
    """Zkash-0.1M (v2.0.0): exactly 100,000 trainable parameters."""
    def __init__(self, n_in: int = 64, n_out: int = 8, p_drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(n_in, 191)
        self.fc2 = nn.Linear(191, 256)
        self.fc3 = nn.Linear(256, 145)
        self.fc4 = nn.Linear(145, n_out)
        self.drop = nn.Dropout(p_drop) if p_drop > 0 else nn.Identity()

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.fc4.weight)
        for m in (self.fc1, self.fc2, self.fc3, self.fc4):
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class Zkash1M(nn.Module):
    """Zkash-1M (v3.0.0): exactly 1,000,000 trainable parameters."""
    def __init__(self, n_in: int = 64, n_out: int = 8, p_drop: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(n_in, 508)     #  33,020
        self.fc2 = nn.Linear(508, 640)      # 325,760
        self.fc3 = nn.Linear(640, 988)      # 633,308
        self.fc4 = nn.Linear(988, n_out)    #   7,912
        self.drop = nn.Dropout(p_drop) if p_drop > 0 else nn.Identity()

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.fc4.weight)
        for m in (self.fc1, self.fc2, self.fc3, self.fc4):
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        x = self.drop(x)
        return self.fc4(x)


def build_model(name: str = "zkash_1m", **kwargs) -> nn.Module:
    """Factory: pick model by name."""
    registry = {
        "zkash_10k": Zkash10K,
        "zkash_01m": Zkash01M,
        "zkash_1m":  Zkash1M,
    }
    if name not in registry:
        raise ValueError(f"unknown model: {name}. options: {list(registry)}")
    return registry[name](**kwargs)
