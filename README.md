# Dendritic Contextual Routing

This repository contains the reproducibility package for a set of experiments on **contextual affine modulation**, **dendritic-inspired routing**, and **micro-replay** in sequential learning.

The core question is:

> When does contextual routing become functionally necessary, and what is required to preserve it under sequential training?

## Main result

On a controlled benchmark called **SDFC shared-head** (*Same-Dimension Feature Conflict*), models without useful contextual conditioning remain at chance level, while context-conditioned models solve the benchmark under joint training.

In sequential training, however, even strong context-conditioned models suffer severe interference. A very small replay buffer fixes this:

> A replay buffer containing only **2% of each task’s training set** raises final accuracy from about **64%** to **95.4%** and reduces forgetting from about **43%** to **1%**, nearly matching joint training.

## Key findings

1. **Contextual conditioning is necessary** on SDFC shared-head.
2. A simple multiplicative dendritic gate is insufficient.
3. The useful primitive is **additive + multiplicative affine modulation**:
   ```text
   h = gamma(context) * h_basal + beta(context)
   ```
4. `film_full` and `dendritic_affine_separate` are statistically indistinguishable across replay budgets.
5. Micro-replay preserves the contextual solution under sequential learning.
6. Replay restores the oldest mirror-conflicted task from about **28%** to **94%** with only a **2%** buffer.

## Repository structure

```text
.
├── src/                         # source code
├── scripts/                     # PowerShell scripts for reproducing runs
├── configs/                     # experiment configs, if applicable
├── results/
│   ├── raw_csv/                 # raw CSV outputs
│   ├── processed/               # processed tables
│   └── main_tables/             # final tables used in the paper
├── paper/
│   ├── figures/                 # final figures
│   └── results_section_dendritic_v2.md
├── docs/
│   ├── README_REPRODUCIBILITY.md
│   ├── EXPERIMENT_LOG.md
│   └── RELEASE_CHECKLIST.md
├── CITATION.cff
├── LICENSE
└── README.md
```

## Minimal reproduction

From the repository root:

```powershell
python -m src.main --make-benchmark --benchmark-seed 12345
powershell -ExecutionPolicy Bypass -File .\scripts\run_sdfc_replay_joint_multiseed.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_sdfc_replay_microbuffer_multiseed.ps1
```

The expected outputs include:

```text
results_seq_microbuffer/aggregated_summary.csv
results_seq_microbuffer/final_task_accuracy.csv
results_seq_microbuffer/accuracy_matrix_summary.csv
results_seq_microbuffer/gate_similarity_by_task_pair.csv
results_seq_microbuffer/buffer_stats_summary.csv
```

## Final replay budgets

The final micro-buffer sweep uses:

| Replay fraction | Examples per task |
|---:|---:|
| 0% | 0 |
| 2% | 200 |
| 5% | 500 |
| 10% | 1000 |

## Final models

The final comparison focuses on:

- `film_full`
- `dendritic_affine_separate`

Earlier experiments also include:

- MLP baselines
- no-context dendritic controls
- multiplicative-only and additive-only ablations
- apical unlock diagnostics
- PermutedMNIST and SplitMNIST controls

## Citation

If you use this repository, please cite it using the metadata in `CITATION.cff`.