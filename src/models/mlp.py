from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn as nn
from .heads import SharedBinaryHead, TaskILHead

class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], scenario: str, num_tasks: int):
        super().__init__()
        dims = [input_dim] + hidden_dims
        self.backbone_layers = nn.ModuleList([nn.Linear(din, dout) for din, dout in zip(dims[:-1], dims[1:])])
        self.activation = nn.ReLU()
        feat_dim = hidden_dims[-1]
        if scenario == "task_il":
            self.head = TaskILHead(feat_dim, num_tasks)
        elif scenario == "shared_head":
            self.head = SharedBinaryHead(feat_dim)
        else:
            raise ValueError(f"Unknown scenario={scenario}")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.backbone_layers:
            h = self.activation(layer(h))
        return h

    def forward(self, x: torch.Tensor, task_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.head(self.encode(x))

    def diagnostic_specs(self):
        specs = []
        for idx, layer in enumerate(self.backbone_layers):
            specs.append({"layer_idx": idx, "component": "basal", "weight": layer.weight, "bias": layer.bias})
        return specs
