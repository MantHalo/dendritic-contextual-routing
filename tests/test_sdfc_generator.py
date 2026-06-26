from __future__ import annotations

import numpy as np

from src.data.sdfc_generator import INPUT_DIM, LATENT_DIM, build_projection, generate_all_splits, rule_for_task


def test_projection_is_deterministic_and_column_normalized():
    p1 = build_projection(12345)
    p2 = build_projection(12345)

    assert p1.shape == (INPUT_DIM, LATENT_DIM)
    assert np.allclose(p1, p2)
    assert np.allclose(np.linalg.norm(p1, axis=0), np.ones(LATENT_DIM), atol=1e-5)


def test_sdfc_mirror_tasks_flip_same_latent_features():
    z = np.zeros((2, LATENT_DIM), dtype=np.float32)
    z[0, 0] = 1.0
    z[0, 1] = 1.0
    z[1, 0] = -1.0
    z[1, 1] = -1.0

    task0 = rule_for_task(z, 0)
    task3 = rule_for_task(z, 3)

    assert task0.tolist() == [1.0, 0.0]
    assert task3.tolist() == [0.0, 1.0]


def test_generate_all_splits_shapes(tmp_path):
    splits = generate_all_splits(tmp_path, num_tasks=4, n_train=11, n_val=7, n_test=5, benchmark_seed=12345)

    assert splits["train"]["x"].shape == (11, INPUT_DIM)
    assert splits["train"]["y"].shape == (11, 4)
    assert splits["val"]["x"].shape == (7, INPUT_DIM)
    assert splits["test"]["x"].shape == (5, INPUT_DIM)
    assert (tmp_path / "artifacts" / "projection_P.npy").exists()
    assert (tmp_path / "artifacts" / "benchmark_meta.json").exists()
