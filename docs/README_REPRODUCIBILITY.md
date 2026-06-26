# Reproducibility Guide

This guide explains how to reproduce the final Dendritic v2 / SDFC shared-head results.

## Fast Verification Path

Use this path when you want to check that the repository is wired correctly without rerunning the full multiseed sweep.

```powershell
python -m src.main --help
python -m src.main --make-benchmark --benchmark-seed 12345
powershell -ExecutionPolicy Bypass -File .\scripts\run_sdfc_replay_joint_smoke.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_sdfc_replay_microbuffer_smoke.ps1
python .\scripts\make_readme_figures.py
```

The smoke scripts are intentionally short. They validate the CLI, data path, model construction, training loop, replay path, aggregation script, and figure-generation path. They are not expected to reproduce the final paper numbers.

## Environment

Recommended:

```text
Python >= 3.11
PyTorch
NumPy
Pandas
Matplotlib
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

For local test development:

```powershell
python -m pip install -r requirements-dev.txt
```

## Step 1 - Generate The Fixed SDFC Benchmark Projection

```powershell
python -m src.main --make-benchmark --benchmark-seed 12345
```

Expected output:

```text
artifacts/projection_P.npy
artifacts/benchmark_meta.json
```

## Step 2 - Joint-Training Reference

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_sdfc_replay_joint_multiseed.ps1
```

Expected output:

```text
results_joint_reference/aggregated_summary.csv
```

## Step 3 - Sequential Micro-Buffer Sweep

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_sdfc_replay_microbuffer_multiseed.ps1
```

Expected output directory:

```text
results_seq_microbuffer/
```

Important files:

```text
aggregated_summary.csv
final_task_accuracy.csv
accuracy_matrix_summary.csv
gate_similarity_by_task_pair.csv
buffer_stats_summary.csv
epoch_diagnostics_summary.csv
runs_summary.csv
```

## Regenerate README And Analysis Figures

```powershell
python .\scripts\make_readme_figures.py
```

Expected figure outputs:

```text
paper/figures/readme_overview.png
paper/figures/sdfc_feature_conflict.png
paper/figures/fig1_replay_summary.png
paper/figures/fig3_gate_similarity_readable.png
```

## Verify Replay Budgets

All final CSVs should contain the replay fractions:

```text
0.00
0.02
0.05
0.10
```

and the replay sizes:

```text
0
200
500
1000
```

If a file contains `0.20` or `2000`, it belongs to the older 0/5/20% sweep and should not be used for final paper tables.

## Expected Main Results

| Model | Replay | Accuracy | Forgetting |
|---|---:|---:|---:|
| film_full | 0% | ~63.9% | ~43.2% |
| film_full | 2% | ~95.4% | ~1.1% |
| film_full | 5% | ~95.9% | ~0.5% |
| film_full | 10% | ~96.0% | ~0.3% |
| dendritic_affine_separate | 0% | ~63.8% | ~43.2% |
| dendritic_affine_separate | 2% | ~95.4% | ~1.1% |
| dendritic_affine_separate | 5% | ~95.9% | ~0.4% |
| dendritic_affine_separate | 10% | ~96.0% | ~0.4% |

## Optional Smoke Tests

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_sdfc_replay_joint_smoke.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_sdfc_replay_microbuffer_smoke.ps1
```

## Unit Tests

The tracked tests are lightweight checks for deterministic benchmark generation, model forward passes, metrics, and figure generation assumptions.

```powershell
python -m pytest -q
```

## Demo Notebook

The notebook is a quick CPU demonstration of the benchmark and replay path. It is not expected to reproduce the final multi-seed numbers.

```powershell
jupyter notebook .\notebooks\quick_sdfc_demo.ipynb
```

If Jupyter is not installed, the notebook can still be inspected as a documented workflow. The validation path for final results remains the CLI and tests above.

## Common Pitfalls

Do not mix CSV files from `0 / 5 / 20%` with files from `0 / 2 / 5 / 10%`.

The clean repository should not contain:

```text
src/src/
scripts/scripts/
configs/configs/
artifacts/artifacts/
```
