from __future__ import annotations
import torch
import torch.nn as nn

class TaskILHead(nn.Module):
    def __init__(self, in_dim: int, num_tasks: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_tasks)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)

class SharedBinaryHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h).squeeze(-1)
