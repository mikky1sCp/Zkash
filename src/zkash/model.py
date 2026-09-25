import torch
import torch.nn as nn
import torch.nn.functional as F


class Zkash10K(nn.Module):
    """Zkash-10K: exactly 10,000 trainable parameters."""

    def __init__(self, n_in: int = 64, n_out: int = 8,
                 hidden1: int = 64, hidden2: int = 80):
        super().__init__()
        self.fc1 = nn.Linear(n_in, hidden1)             # 64*64 + 64 = 4160
        self.fc2 = nn.Linear(hidden1, hidden2)          # 64*80 + 80 = 5200
        self.fc3 = nn.Linear(hidden2, n_out, bias=False)  # 80*8    = 640
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # logits