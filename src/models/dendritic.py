from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn as nn
from .heads import SharedBinaryHead, TaskILHead

class DendriticLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, context_dim: int, gate_temperature: float = 1.0, centered_gate: bool = True, no_context: bool = False):
        super().__init__()
        self.basal = nn.Linear(in_dim, out_dim)
        self.apical = None if no_context else nn.Linear(context_dim, out_dim)
        self.apical_bias = nn.Parameter(torch.zeros(out_dim))
        self.activation = nn.ReLU()
        self.no_context = no_context
        self.gate_temperature = gate_temperature
        self.centered_gate = centered_gate

    def gate_from_context(self, context: Optional[torch.Tensor]) -> torch.Tensor:
        if self.no_context:
            batch = 1 if context is None else context.shape[0]
            device = self.basal.weight.device if context is None else context.device
            pre = self.apical_bias.unsqueeze(0).expand(batch, -1).to(device)
        else:
            if context is None:
                raise ValueError("Context required for dendritic_unlock_centered")
            pre = self.apical(context) + self.apical_bias
        if self.centered_gate:
            return 1.0 + torch.tanh(pre / self.gate_temperature)
        return torch.sigmoid(pre / self.gate_temperature)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor]):
        h_basal = self.activation(self.basal(x))
        gate = self.gate_from_context(context)
        return h_basal * gate, gate

class DendriticAffineLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, context_dim: int, gate_temperature: float = 1.0):
        super().__init__()
        self.basal = nn.Linear(in_dim, out_dim)
        self.gate_encoder = nn.Linear(context_dim, out_dim)
        self.gate_bias = nn.Parameter(torch.zeros(out_dim))
        self.add_encoder = nn.Linear(context_dim, out_dim)
        self.add_bias = nn.Parameter(torch.zeros(out_dim))
        self.activation = nn.ReLU()
        self.gate_temperature = gate_temperature

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        if context is None:
            raise ValueError("Context required for dendritic_affine_separate")
        h_basal = self.activation(self.basal(x))
        gate = 1.0 + torch.tanh((self.gate_encoder(context) + self.gate_bias) / self.gate_temperature)
        add = self.add_encoder(context) + self.add_bias
        return h_basal * gate + add, gate

class DendriticClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], scenario: str, num_tasks: int, gate_temperature: float = 1.0, centered_gate: bool = True, no_context: bool = False, affine: bool = False):
        super().__init__()
        dims = [input_dim] + hidden_dims
        self.context_dim = num_tasks
        if affine:
            self.layers = nn.ModuleList([DendriticAffineLayer(din, dout, self.context_dim, gate_temperature) for din, dout in zip(dims[:-1], dims[1:])])
        else:
            self.layers = nn.ModuleList([DendriticLayer(din, dout, self.context_dim, gate_temperature, centered_gate, no_context) for din, dout in zip(dims[:-1], dims[1:])])
        feat_dim = hidden_dims[-1]
        if scenario == "task_il":
            self.head = TaskILHead(feat_dim, num_tasks)
        elif scenario == "shared_head":
            self.head = SharedBinaryHead(feat_dim)
        else:
            raise ValueError(f"Unknown scenario={scenario}")

    def encode(self, x: torch.Tensor, task_context: Optional[torch.Tensor], return_gates: bool = False):
        h = x
        gates = []
        for layer in self.layers:
            h, gate = layer(h, task_context)
            gates.append(gate)
        return (h, gates) if return_gates else h

    def forward(self, x: torch.Tensor, task_context: Optional[torch.Tensor]):
        return self.head(self.encode(x, task_context, return_gates=False))

    def get_gate_vectors(self, task_context: Optional[torch.Tensor]):
        if task_context is None:
            task_context = torch.zeros(1, self.context_dim, device=self.layers[0].basal.weight.device)
        x = torch.zeros(task_context.shape[0], self.layers[0].basal.in_features, device=task_context.device)
        _, gates = self.encode(x, task_context, return_gates=True)
        return gates

    def diagnostic_specs(self):
        specs = []
        for idx, layer in enumerate(self.layers):
            specs.append({"layer_idx": idx, "component": "basal", "weight": layer.basal.weight, "bias": layer.basal.bias})
            if isinstance(layer, DendriticAffineLayer):
                specs.append({"layer_idx": idx, "component": "gate", "weight": layer.gate_encoder.weight, "bias": layer.gate_bias})
                specs.append({"layer_idx": idx, "component": "add", "weight": layer.add_encoder.weight, "bias": layer.add_bias})
            else:
                specs.append({"layer_idx": idx, "component": "apical", "weight": None if layer.apical is None else layer.apical.weight, "bias": layer.apical_bias})
        return specs
