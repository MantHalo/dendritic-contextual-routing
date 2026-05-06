from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
import torch
from torch.utils.data import Dataset

class SDFCTaskDataset(Dataset):
    def __init__(self, x: np.ndarray, y_task: np.ndarray, task_id: int):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y_task).float()
        self.task_id = task_id

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return {
            "x": self.x[idx],
            "y": self.y[idx],
            "task_id": torch.tensor(self.task_id, dtype=torch.long),
        }

@dataclass
class SDFCDataBundle:
    train: Dict[int, SDFCTaskDataset]
    val: Dict[int, SDFCTaskDataset]
    test: Dict[int, SDFCTaskDataset]
    num_tasks: int
    input_dim: int

def build_data_bundle(splits, num_tasks: int) -> SDFCDataBundle:
    train, val, test = {}, {}, {}
    x_train, x_val, x_test = splits["train"]["x"], splits["val"]["x"], splits["test"]["x"]
    y_train, y_val, y_test = splits["train"]["y"], splits["val"]["y"], splits["test"]["y"]
    for task_id in range(num_tasks):
        train[task_id] = SDFCTaskDataset(x_train, y_train[:, task_id], task_id)
        val[task_id] = SDFCTaskDataset(x_val, y_val[:, task_id], task_id)
        test[task_id] = SDFCTaskDataset(x_test, y_test[:, task_id], task_id)
    return SDFCDataBundle(train=train, val=val, test=test, num_tasks=num_tasks, input_dim=x_train.shape[1])
