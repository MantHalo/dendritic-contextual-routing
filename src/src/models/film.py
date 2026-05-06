from __future__ import annotations
from typing import List
import torch
import torch.nn as nn
from .heads import SharedBinaryHead, TaskILHead

class FiLMLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, context_dim: int, gate_temperature: float = 1.0, mode: str = "full"):
        super().__init__()
        self.basal = nn.Linear(in_dim, out_dim)
        self.gamma = nn.Linear(context_dim, out_dim)
        self.beta = nn.Linear(context_dim, out_dim)
        self.activation = nn.ReLU()
        self.gate_temperature = gate_temperature
        self.mode = mode

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        basal = self.activation(self.basal(x))
        gamma_raw = self.gamma(context)
        beta_raw = self.beta(context)
        gamma = 1.0 + torch.tanh(gamma_raw / self.gate_temperature)
        beta = beta_raw
        if self.mode == "full":
            out = gamma * basal + beta
        elif self.mode == "additive_only":
            out = basal + beta
            gamma = torch.ones_like(basal)
        elif self.mode == "multiplicative_only":
            out = gamma * basal
            beta = torch.zeros_like(basal)
        else:
            raise ValueError(f"Unknown FiLM mode={self.mode}")
        return out, gamma, beta

class FiLMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], scenario: str, num_tasks: int, gate_temperature: float = 1.0, mode: str = "full"):
        super().__init__()
        dims = [input_dim] + hidden_dims
        self.context_dim = num_tasks
        self.mode = mode
        self.layers = nn.ModuleList([FiLMLayer(din, dout, self.context_dim, gate_temperature, mode=mode) for din, dout in zip(dims[:-1], dims[1:])])
        feat_dim = hidden_dims[-1]
        if scenario == "task_il":
            self.head = TaskILHead(feat_dim, num_tasks)
        elif scenario == "shared_head":
            self.head = SharedBinaryHead(feat_dim)
        else:
            raise ValueError(f"Unknown scenario={scenario}")

    def encode(self, x: torch.Tensor, task_context: torch.Tensor, return_gates: bool = False):
        h = x
        gates = []
        for layer in self.layers:
            h, gamma, _beta = layer(h, task_context)
            gates.append(gamma)
        return (h, gates) if return_gates else h

    def forward(self, x: torch.Tensor, task_context: torch.Tensor):
        return self.head(self.encode(x, task_context, return_gates=False))

    def get_gate_vectors(self, task_context: torch.Tensor):
        x = torch.zeros(task_context.shape[0], self.layers[0].basal.in_features, device=task_context.device)
        _, gates = self.encode(x, task_context, return_gates=True)
        return gates

    def diagnostic_specs(self):
        specs = []
        for idx, layer in enumerate(self.layers):
            specs.append({"layer_idx": idx, "component": "basal", "weight": layer.basal.weight, "bias": layer.basal.bias})
            specs.append({"layer_idx": idx, "component": "gamma", "weight": layer.gamma.weight, "bias": layer.gamma.bias})
            specs.append({"layer_idx": idx, "component": "beta", "weight": layer.beta.weight, "bias": layer.beta.bias})
        return specs
