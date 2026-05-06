from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def batch_context(task_ids: torch.Tensor, num_tasks: int) -> torch.Tensor:
    return F.one_hot(task_ids, num_classes=num_tasks).float()


def one_hot_context(task_id: int, num_tasks: int, batch_size: int, device: torch.device) -> torch.Tensor:
    idx = torch.full((batch_size,), task_id, dtype=torch.long, device=device)
    return batch_context(idx, num_tasks)


def collect_epoch_diagnostics(model: torch.nn.Module, run_name: str, benchmark: str, scenario: str, task_id: int, epoch: int, seed: int) -> List[Dict]:
    rows = []
    if not hasattr(model, "diagnostic_specs"):
        return rows
    for spec in model.diagnostic_specs():
        weight = spec.get("weight", None)
        bias = spec.get("bias", None)
        rows.append({
            "run_name": run_name,
            "benchmark": benchmark,
            "scenario": scenario,
            "task_id": task_id,
            "epoch": epoch,
            "seed": seed,
            "layer_idx": spec["layer_idx"],
            "component": spec["component"],
            "weight_norm": float(weight.data.norm().item()) if weight is not None else 0.0,
            "grad_norm": float(weight.grad.norm().item()) if (weight is not None and weight.grad is not None) else 0.0,
            "bias_norm": float(bias.data.norm().item()) if bias is not None else 0.0,
            "bias_grad_norm": float(bias.grad.norm().item()) if (bias is not None and bias.grad is not None) else 0.0,
        })
    return rows


def summarize_epoch_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.groupby(["run_name","benchmark","scenario","layer_idx","component"], as_index=False).agg(
        mean_weight_norm=("weight_norm","mean"),
        mean_grad_norm=("grad_norm","mean"),
        mean_bias_norm=("bias_norm","mean"),
        mean_bias_grad_norm=("bias_grad_norm","mean"),
        max_grad_norm=("grad_norm","max"),
        n_rows=("grad_norm","size"),
    )


def gate_reports_for_model(model: torch.nn.Module, run_name: str, benchmark: str, scenario: str, num_tasks: int, device: torch.device):
    if not hasattr(model, "get_gate_vectors"):
        return [], []
    task_contexts = [one_hot_context(t, num_tasks, 1, device) for t in range(num_tasks)]
    per_task_gate_vectors = []
    for ctx in task_contexts:
        gates = model.get_gate_vectors(ctx)
        per_task_gate_vectors.append([g.squeeze(0).detach().cpu().numpy().astype(np.float64) for g in gates])
    gate_rows, sim_rows = [], []
    num_layers = len(per_task_gate_vectors[0]) if per_task_gate_vectors else 0
    for layer_idx in range(num_layers):
        stacked = np.stack([per_task_gate_vectors[t][layer_idx] for t in range(num_tasks)], axis=0)
        neuron_means = stacked.mean(axis=0)
        neuron_stds = stacked.std(axis=0)
        for task_id in range(num_tasks):
            vec = stacked[task_id]
            gate_rows.append({
                "run_name": run_name,
                "benchmark": benchmark,
                "scenario": scenario,
                "layer_idx": layer_idx,
                "task_id": task_id,
                "gate_mean": float(vec.mean()),
                "gate_std": float(vec.std()),
                "gate_min": float(vec.min()),
                "gate_max": float(vec.max()),
                "fraction_lt_0_1": float((vec < 0.1).mean()),
                "fraction_gt_0_9": float((vec > 0.9).mean()),
                "mean_neuron_std": float(neuron_stds.mean()),
                "std_of_neuron_means": float(neuron_means.std()),
            })
        for i in range(num_tasks):
            for j in range(num_tasks):
                a, b = stacked[i], stacked[j]
                cos = float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))
                sim_rows.append({
                    "run_name": run_name,
                    "benchmark": benchmark,
                    "scenario": scenario,
                    "layer_idx": layer_idx,
                    "task_i": i,
                    "task_j": j,
                    "cosine_similarity": cos,
                    "is_mirror_pair": int((i, j) in {(0, 3), (3, 0), (1, 2), (2, 1)}),
                })
    return gate_rows, sim_rows
