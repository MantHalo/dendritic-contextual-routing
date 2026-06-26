from __future__ import annotations

import numpy as np
import torch

from src.models.dendritic import DendriticClassifier
from src.models.film import FiLMClassifier
from src.training.metrics import average_accuracy, average_forgetting, backward_transfer


def _context(task_ids: torch.Tensor, num_tasks: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(task_ids, num_classes=num_tasks).float()


def test_models_produce_shared_head_logits():
    x = torch.randn(6, 32)
    task_ids = torch.tensor([0, 1, 2, 3, 0, 1])
    context = _context(task_ids, 4)

    models = [
        FiLMClassifier(32, [16], scenario="shared_head", num_tasks=4, mode="full"),
        DendriticClassifier(32, [16], scenario="shared_head", num_tasks=4, affine=True),
    ]

    for model in models:
        logits = model(x, context)
        gates = model.get_gate_vectors(context)

        assert logits.shape == (6,)
        assert len(gates) == 1
        assert gates[0].shape == (6, 16)


def test_metrics_match_expected_sequential_behavior():
    acc = np.array(
        [
            [0.95, 0.50, 0.50, 0.50],
            [0.80, 0.96, 0.50, 0.50],
            [0.70, 0.90, 0.97, 0.50],
            [0.60, 0.88, 0.94, 0.98],
        ],
        dtype=np.float64,
    )

    assert average_accuracy(acc) == np.mean(acc[-1])
    assert np.isclose(average_forgetting(acc), np.mean([0.95 - 0.60, 0.96 - 0.88, 0.97 - 0.94]))
    assert np.isclose(backward_transfer(acc), np.mean([0.60 - 0.95, 0.88 - 0.96, 0.94 - 0.97]))
