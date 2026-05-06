from __future__ import annotations
import numpy as np

def average_accuracy(acc_matrix: np.ndarray) -> float:
    return float(np.mean(acc_matrix[-1]))

def average_forgetting(acc_matrix: np.ndarray) -> float:
    forget = []
    for t in range(acc_matrix.shape[1] - 1):
        best = np.max(acc_matrix[:, t])
        final = acc_matrix[-1, t]
        forget.append(best - final)
    return float(np.mean(forget)) if forget else 0.0

def backward_transfer(acc_matrix: np.ndarray) -> float:
    vals = [acc_matrix[-1, t] - acc_matrix[t, t] for t in range(acc_matrix.shape[1] - 1)]
    return float(np.mean(vals)) if vals else 0.0
