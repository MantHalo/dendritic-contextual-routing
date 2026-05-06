# Reproducibility Guide

This document describes how to reproduce the final Dendritic v2 / SDFC shared-head results.

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

If no `requirements.txt` is present yet, create one from the working environment:

```powershell
python -m pip freeze > requirements.txt
```

## Step 1 — Generate the fixed benchmark projection

The SDFC benchmark uses a fixed projection matrix `P`.

```powershell
python -m src.main --make-benchmark --benchmark-seed 12345
```

Expected output:

```text
artifacts/
```

containing the fixed benchmark projection.

## Step 2 — Joint-training reference

This run establishes the upper bound under non-sequential training.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_sdfc_replay_joint_multiseed.ps1
```

Expected output:

```text
results_joint_reference/aggregated_summary.csv
```

Expected result:

- `film_full` ≈ 96%
- `dendritic_affine_separate` ≈ 96%

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

## Step 4 — Verify replay budgets

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

If a file contains `0.20` or `2000`, it belongs to the older 0/5/20% sweep and should not be used for the final paper tables.

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

## Figure generation

Use the final CSV files to generate:

1. Accuracy and forgetting vs replay budget.
2. Accuracy matrices `A[i,j]`.
3. Gate similarity for mirror vs non-mirror task pairs.

The final figure pack should include:

```text
fig1a_accuracy_vs_replay.png
fig1b_forgetting_vs_replay.png
fig2_matrix_*.png
fig3_gate_similarity_mirror_vs_nonmirror.png
```

## Common pitfalls

### Mixed CSV files

Do not mix files from:

```text
0 / 5 / 20%
```

with files from:

```text
0 / 2 / 5 / 10%
```

### Smoke tests

Smoke tests are optional. They are useful for checking that the pipeline runs, but they are not required for the final multi-seed results.

### PowerShell policy

If scripts are blocked:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

or run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\script_name.ps1
```