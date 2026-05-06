from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from .diagnostics import batch_context, collect_epoch_diagnostics, gate_reports_for_model
from .metrics import average_accuracy, average_forgetting, backward_transfer


@dataclass
class TrainResult:
    acc_matrix: np.ndarray
    summary: Dict
    epoch_rows: List[Dict]
    gate_rows: List[Dict]
    gate_sim_rows: List[Dict]
    matrix_rows: List[Dict]
    buffer_rows: List[Dict]


class ReplayBuffer:
    def __init__(self, replay_size_per_task: int):
        self.replay_size_per_task = int(max(replay_size_per_task, 0))
        self.storage: Dict[int, Dict[str, torch.Tensor]] = {}

    def __len__(self) -> int:
        return sum(v["x"].shape[0] for v in self.storage.values())

    def add_task_dataset(self, dataset, rng: np.random.Generator) -> None:
        if self.replay_size_per_task <= 0:
            return
        n = len(dataset)
        k = min(self.replay_size_per_task, n)
        idx = rng.choice(n, size=k, replace=False)
        xs = torch.stack([dataset[int(i)]["x"] for i in idx], dim=0)
        ys = torch.stack([dataset[int(i)]["y"] for i in idx], dim=0)
        tids = torch.stack([dataset[int(i)]["task_id"] for i in idx], dim=0)
        self.storage[int(tids[0].item())] = {"x": xs, "y": ys, "task_id": tids}

    def sample(self, n: int, device: torch.device):
        if len(self) == 0 or n <= 0:
            return None
        xs = torch.cat([v["x"] for v in self.storage.values()], dim=0)
        ys = torch.cat([v["y"] for v in self.storage.values()], dim=0)
        tids = torch.cat([v["task_id"] for v in self.storage.values()], dim=0)
        k = min(n, xs.shape[0])
        idx = torch.randint(0, xs.shape[0], (k,))
        return {
            "x": xs[idx].to(device),
            "y": ys[idx].to(device),
            "task_id": tids[idx].to(device),
        }

    def stats_rows(self, run_name: str, benchmark: str, scenario: str, seed: int, stage_label: str, stage_index: int) -> List[Dict]:
        rows: List[Dict] = []
        total = len(self)
        per_task = {task: v["x"].shape[0] for task, v in self.storage.items()}
        if not per_task:
            rows.append({
                "run_name": run_name,
                "benchmark": benchmark,
                "scenario": scenario,
                "seed": seed,
                "stage_label": stage_label,
                "stage_index": stage_index,
                "buffer_total": 0,
                "buffer_task_id": -1,
                "buffer_task_count": 0,
            })
            return rows
        for task_id, count in sorted(per_task.items()):
            rows.append({
                "run_name": run_name,
                "benchmark": benchmark,
                "scenario": scenario,
                "seed": seed,
                "stage_label": stage_label,
                "stage_index": stage_index,
                "buffer_total": total,
                "buffer_task_id": task_id,
                "buffer_task_count": count,
            })
        return rows


def build_optimizer(model: torch.nn.Module, lr: float, apical_lr_mult: float, weight_decay: float):
    apical_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if any(k in lname for k in ("apical", "gamma", "beta", "gate", "add")):
            apical_params.append(param)
        else:
            other_params.append(param)
    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": lr, "weight_decay": weight_decay})
    if apical_params:
        groups.append({"params": apical_params, "lr": lr * apical_lr_mult, "weight_decay": weight_decay})
    return torch.optim.Adam(groups)


def loss_for_batch(logits: torch.Tensor, y: torch.Tensor, scenario: str, task_ids: torch.Tensor) -> torch.Tensor:
    if scenario == "task_il":
        selected = logits.gather(1, task_ids.unsqueeze(1)).squeeze(1)
        return F.binary_cross_entropy_with_logits(selected, y)
    if scenario == "shared_head":
        return F.binary_cross_entropy_with_logits(logits, y)
    raise ValueError(f"Unknown scenario={scenario}")


@torch.no_grad()
def evaluate_task_with_stats(model: torch.nn.Module, dataset, scenario: str, num_tasks: int, batch_size: int, device: torch.device) -> Dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    total_pos = 0
    all_logits = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        task_ids = batch["task_id"].to(device)
        context = batch_context(task_ids, num_tasks).to(device)
        logits = model(x, context)
        pred_logit = logits.gather(1, task_ids.unsqueeze(1)).squeeze(1) if scenario == "task_il" else logits
        preds = (torch.sigmoid(pred_logit) > 0.5).float()
        correct += int((preds == y).sum().item())
        total += int(y.numel())
        total_pos += int(preds.sum().item())
        all_logits.append(pred_logit.detach().cpu())
    logits_cat = torch.cat(all_logits) if all_logits else torch.zeros(1)
    return {
        "accuracy": correct / max(total, 1),
        "positive_rate": total_pos / max(total, 1),
        "mean_logit": float(logits_cat.mean().item()),
        "std_logit": float(logits_cat.std(unbiased=False).item()),
    }


def _record_stage_metrics(model, data_bundle, scenario, num_tasks, batch_size, device, run_name, benchmark, seed, stage_label, stage_index) -> Tuple[np.ndarray, List[Dict]]:
    accs = np.zeros((num_tasks,), dtype=np.float64)
    rows: List[Dict] = []
    for task_id in range(num_tasks):
        stats = evaluate_task_with_stats(model, data_bundle.test[task_id], scenario, num_tasks, batch_size, device)
        accs[task_id] = stats["accuracy"]
        rows.append({
            "run_name": run_name,
            "benchmark": benchmark,
            "scenario": scenario,
            "seed": seed,
            "stage_label": stage_label,
            "stage_index": stage_index,
            "eval_task": task_id,
            **stats,
        })
    return accs, rows


def _train_epoch(model, loader, optimizer, scenario, num_tasks, device, run_name, benchmark, seed, task_id, epoch, replay_buffer: ReplayBuffer | None = None, replay_batch_size: int = 0):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    epoch_rows: List[Dict] = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        task_ids = batch["task_id"].to(device)
        if replay_buffer is not None and len(replay_buffer) > 0 and replay_batch_size > 0:
            replay = replay_buffer.sample(replay_batch_size, device)
            if replay is not None:
                x = torch.cat([x, replay["x"]], dim=0)
                y = torch.cat([y, replay["y"]], dim=0)
                task_ids = torch.cat([task_ids, replay["task_id"]], dim=0)
        context = batch_context(task_ids, num_tasks).to(device)
        optimizer.zero_grad()
        logits = model(x, context)
        loss = loss_for_batch(logits, y, scenario, task_ids)
        loss.backward()
        epoch_rows.extend(collect_epoch_diagnostics(model, run_name, benchmark, scenario, task_id, epoch, seed))
        optimizer.step()
        total_loss += float(loss.item()) * x.shape[0]
        pred_logit = logits.gather(1, task_ids.unsqueeze(1)).squeeze(1) if scenario == "task_il" else logits
        preds = (torch.sigmoid(pred_logit) > 0.5).float()
        total_correct += int((preds == y).sum().item())
        total_examples += int(y.numel())
    return total_loss / max(total_examples, 1), total_correct / max(total_examples, 1), epoch_rows


def train_sequential(model: torch.nn.Module, data_bundle, scenario: str, num_tasks: int, epochs: int, batch_size: int, lr: float, apical_lr_mult: float, weight_decay: float, device: torch.device, run_name: str, benchmark: str, seed: int, replay_size_per_task: int = 0) -> TrainResult:
    optimizer = build_optimizer(model, lr, apical_lr_mult, weight_decay)
    model.to(device)
    acc_matrix = np.zeros((num_tasks, num_tasks), dtype=np.float64)
    epoch_rows: List[Dict] = []
    matrix_rows: List[Dict] = []
    buffer_rows: List[Dict] = []
    replay_buffer = ReplayBuffer(replay_size_per_task)
    rng = np.random.default_rng(seed + 777)
    for task_id in range(num_tasks):
        train_loader = DataLoader(data_bundle.train[task_id], batch_size=batch_size, shuffle=True)
        epoch_rows.extend(collect_epoch_diagnostics(model, run_name, benchmark, scenario, task_id, 0, seed))
        for epoch in range(1, epochs + 1):
            avg_loss, avg_acc, epoch_diag = _train_epoch(
                model, train_loader, optimizer, scenario, num_tasks, device, run_name, benchmark, seed, task_id, epoch,
                replay_buffer=replay_buffer, replay_batch_size=batch_size,
            )
            epoch_rows.extend(epoch_diag)
            print(f"task={task_id} epoch={epoch}/{epochs} loss={avg_loss:.4f} acc={avg_acc:.4f}")
        stage_accs, stage_rows = _record_stage_metrics(model, data_bundle, scenario, num_tasks, batch_size, device, run_name, benchmark, seed, f"after_task_{task_id}", task_id)
        acc_matrix[task_id] = stage_accs
        matrix_rows.extend(stage_rows)
        seen = [round(stage_accs[t], 4) for t in range(task_id + 1)]
        print(f"Seen-task accuracies after task {task_id}: {seen}")
        replay_buffer.add_task_dataset(data_bundle.train[task_id], rng)
        buffer_rows.extend(replay_buffer.stats_rows(run_name, benchmark, scenario, seed, f"after_task_{task_id}", task_id))
    summary = {
        "average_accuracy": average_accuracy(acc_matrix),
        "average_forgetting": average_forgetting(acc_matrix),
        "backward_transfer": backward_transfer(acc_matrix),
    }
    gate_rows, gate_sim_rows = gate_reports_for_model(model, run_name, benchmark, scenario, num_tasks, device)
    return TrainResult(acc_matrix=acc_matrix, summary=summary, epoch_rows=epoch_rows, gate_rows=gate_rows, gate_sim_rows=gate_sim_rows, matrix_rows=matrix_rows, buffer_rows=buffer_rows)


def train_joint(model: torch.nn.Module, data_bundle, scenario: str, num_tasks: int, epochs: int, batch_size: int, lr: float, apical_lr_mult: float, weight_decay: float, device: torch.device, run_name: str, benchmark: str, seed: int) -> TrainResult:
    optimizer = build_optimizer(model, lr, apical_lr_mult, weight_decay)
    model.to(device)
    joint_dataset = ConcatDataset([data_bundle.train[t] for t in range(num_tasks)])
    train_loader = DataLoader(joint_dataset, batch_size=batch_size, shuffle=True)
    epoch_rows: List[Dict] = []
    epoch_rows.extend(collect_epoch_diagnostics(model, run_name, benchmark, scenario, -1, 0, seed))
    for epoch in range(1, epochs + 1):
        avg_loss, avg_acc, epoch_diag = _train_epoch(model, train_loader, optimizer, scenario, num_tasks, device, run_name, benchmark, seed, -1, epoch)
        epoch_rows.extend(epoch_diag)
        print(f"joint epoch={epoch}/{epochs} loss={avg_loss:.4f} acc={avg_acc:.4f}")
    stage_accs, stage_rows = _record_stage_metrics(model, data_bundle, scenario, num_tasks, batch_size, device, run_name, benchmark, seed, "joint_final", 0)
    acc_matrix = np.expand_dims(stage_accs, axis=0)
    summary = {
        "average_accuracy": float(np.mean(stage_accs)),
        "average_forgetting": 0.0,
        "backward_transfer": 0.0,
    }
    gate_rows, gate_sim_rows = gate_reports_for_model(model, run_name, benchmark, scenario, num_tasks, device)
    return TrainResult(acc_matrix=acc_matrix, summary=summary, epoch_rows=epoch_rows, gate_rows=gate_rows, gate_sim_rows=gate_sim_rows, matrix_rows=stage_rows, buffer_rows=[])
