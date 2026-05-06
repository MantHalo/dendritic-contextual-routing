from __future__ import annotations
from pathlib import Path
from typing import Dict
import json
import numpy as np

DEFAULT_BENCHMARK_SEED = 12345
INPUT_DIM = 32
LATENT_DIM = 8

def artifacts_dir(project_root: Path) -> Path:
    d = project_root / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d

def projection_path(project_root: Path) -> Path:
    return artifacts_dir(project_root) / "projection_P.npy"

def meta_path(project_root: Path) -> Path:
    return artifacts_dir(project_root) / "benchmark_meta.json"

def build_projection(benchmark_seed: int = DEFAULT_BENCHMARK_SEED) -> np.ndarray:
    rng = np.random.default_rng(benchmark_seed)
    p = rng.normal(0.0, 1.0, size=(INPUT_DIM, LATENT_DIM)).astype(np.float32)
    col_norms = np.linalg.norm(p, axis=0, keepdims=True) + 1e-8
    return (p / col_norms).astype(np.float32)

def ensure_projection(project_root: Path, benchmark_seed: int = DEFAULT_BENCHMARK_SEED) -> np.ndarray:
    p_path = projection_path(project_root)
    m_path = meta_path(project_root)
    if p_path.exists():
        return np.load(p_path)
    p = build_projection(benchmark_seed)
    np.save(p_path, p)
    meta = {
        "benchmark_name": "sdfc",
        "benchmark_seed": benchmark_seed,
        "input_dim": INPUT_DIM,
        "latent_dim": LATENT_DIM,
        "noise_sigma": 0.1,
    }
    m_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return p

def rule_for_task(z: np.ndarray, task_id: int) -> np.ndarray:
    if task_id == 0:
        s = z[:, 0] + z[:, 1]
    elif task_id == 1:
        s = -z[:, 0] + z[:, 1]
    elif task_id == 2:
        s = z[:, 0] - z[:, 1]
    elif task_id == 3:
        s = -z[:, 0] - z[:, 1]
    elif task_id == 4:
        s = z[:, 2] + z[:, 3]
    elif task_id == 5:
        s = -z[:, 2] + z[:, 3]
    elif task_id == 6:
        s = z[:, 2] - z[:, 3]
    elif task_id == 7:
        s = -z[:, 2] - z[:, 3]
    else:
        raise ValueError(f"Unsupported task_id={task_id}. Use tasks in [4, 8].")
    return (s > 0).astype(np.float32)

def generate_split(p: np.ndarray, n_samples: int, num_tasks: int, split_seed: int, noise_sigma: float = 0.1) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(split_seed)
    z = rng.normal(0.0, 1.0, size=(n_samples, LATENT_DIM)).astype(np.float32)
    eps = rng.normal(0.0, noise_sigma, size=(n_samples, INPUT_DIM)).astype(np.float32)
    x = z @ p.T + eps
    y = np.stack([rule_for_task(z, t) for t in range(num_tasks)], axis=1).astype(np.float32)
    return {"x": x.astype(np.float32), "y": y, "z": z}

def generate_all_splits(project_root: Path, num_tasks: int, n_train: int, n_val: int, n_test: int, benchmark_seed: int = DEFAULT_BENCHMARK_SEED, noise_sigma: float = 0.1):
    if num_tasks not in (4, 8):
        raise ValueError("For SDFC, use num_tasks=4 or 8 in this version.")
    p = ensure_projection(project_root, benchmark_seed)
    return {
        "train": generate_split(p, n_train, num_tasks, benchmark_seed + 1, noise_sigma),
        "val": generate_split(p, n_val, num_tasks, benchmark_seed + 2, noise_sigma),
        "test": generate_split(p, n_test, num_tasks, benchmark_seed + 3, noise_sigma),
    }
