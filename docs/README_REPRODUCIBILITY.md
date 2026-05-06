# Reproducibility Guide

This guide explains how to reproduce the final Dendritic v2 / SDFC shared-head results.

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

## Step 1 — Generate the fixed SDFC benchmark projection

```powershell
python -m src.main --make-benchmark --benchmark-seed 12345
```

Expected output:

```text
artifacts/projection_P.npy
artifacts/benchmark_meta.json
```

## Step 2 — Joint-training reference

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_sdfc_replay_joint_multiseed.ps1
```

Expected output:

```text
results_joint_reference/aggregated_summary.csv
```

## Step 3 — Sequential micro-buffer sweep

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

## Verify replay budgets

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

## Expected main results

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

## Optional smoke tests

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_sdfc_replay_joint_smoke.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_sdfc_replay_microbuffer_smoke.ps1
```

## Common pitfalls

Do not mix CSV files from `0 / 5 / 20%` with files from `0 / 2 / 5 / 10%`.

The clean repository should not contain:

```text
src/src/
scripts/scripts/
configs/configs/
artifacts/artifacts/
```