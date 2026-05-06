# Dendritic Contextual Routing

Dendritic-inspired contextual routing with affine modulation and micro-replay for sequential feature-conflict learning.

This repository is a reproducibility package for experiments on **contextual affine modulation**, **dendritic-inspired routing**, and **micro-replay** in sequential learning.

Developed under **OPAL-dev / OPAL.inc** as an independent research exploration on dendritic-inspired contextual routing, affine modulation, and micro-replay for continual learning.

Repository: <https://github.com/OPAL-dev/dendritic-contextual-routing>

---

## Core question

> When does contextual routing become functionally necessary, and what is required to preserve it under sequential training?

The project studies this question through a controlled benchmark called **SDFC shared-head** (*Same-Dimension Feature Conflict*), where the same input dimensions must be interpreted differently across tasks.

This creates explicit feature-sign conflicts and mirror-task pairs, making it possible to test whether a model truly uses contextual control rather than merely learning a shared representation.

---

## Main result

On SDFC shared-head, models without useful contextual conditioning remain near chance level, while context-conditioned models solve the benchmark under joint training.

In sequential training, however, even strong context-conditioned models suffer severe interference. A small replay buffer fixes this:

> A replay buffer containing only **2% of each task’s training set** raises final accuracy from about **64%** to **95.4%** and reduces forgetting from about **43%** to **1%**, nearly matching joint training.

The oldest mirror-conflicted task, **task 0**, recovers from about **28%** to **94%** with only a **2%** replay buffer.

---

## Key findings

1. **Contextual conditioning is necessary** on SDFC shared-head.
2. A simple multiplicative dendritic gate is insufficient.
3. The useful primitive is **additive + multiplicative affine modulation**:

   ```text
   h = gamma(context) * h_basal + beta(context)
   ```

4. A separated affine dendritic variant implements the same functional primitive:

   ```text
   h = g(context) ⊙ h_basal + a(context)
   ```

5. `film_full` and `dendritic_affine_separate` are statistically indistinguishable across replay budgets.
6. Micro-replay preserves the contextual solution under sequential learning.

In short:

> Contextual affine modulation solves the feature-conflict structure; micro-replay preserves it across sequential learning.

---

## Final replay budgets

| Replay fraction | Examples per task |
|---:|---:|
| 0% | 0 |
| 2% | 200 |
| 5% | 500 |
| 10% | 1000 |

---

## Final models

The final comparison focuses on:

- `film_full`
- `dendritic_affine_separate`

Earlier experimental branches also included MLP baselines, no-context dendritic controls, additive/multiplicative ablations, apical unlock diagnostics, PermutedMNIST, and SplitMNIST controls.

---

## Repository structure

```text
.
├── src/
├── scripts/
├── configs/
├── artifacts/
├── results/
│   ├── raw_csv/
│   ├── processed/
│   └── main_tables/
├── paper/
│   ├── figures/
│   └── results_section_dendritic_v2.md
├── docs/
│   ├── README_REPRODUCIBILITY.md
│   ├── EXPERIMENT_LOG.md
│   └── RELEASE_CHECKLIST.md
├── CITATION.cff
├── LICENSE
└── README.md
```

---

## Quick reproduction

From the repository root:

```powershell
python -m src.main --make-benchmark --benchmark-seed 12345

powershell -ExecutionPolicy Bypass -File .\scripts\run_sdfc_replay_joint_multiseed.ps1

powershell -ExecutionPolicy Bypass -File .\scripts\run_sdfc_replay_microbuffer_multiseed.ps1
```

Curated final CSVs are stored in:

```text
results/raw_csv/
```

Final paper tables are stored in:

```text
results/main_tables/
```

Final figures are stored in:

```text
paper/figures/
```

---

## Main final results

| Model | Replay | Accuracy | Forgetting |
|---|---:|---:|---:|
| `film_full` | 0% | ~63.9% | ~43.2% |
| `film_full` | 2% | ~95.4% | ~1.1% |
| `film_full` | 5% | ~95.9% | ~0.5% |
| `film_full` | 10% | ~96.0% | ~0.3% |
| `dendritic_affine_separate` | 0% | ~63.8% | ~43.2% |
| `dendritic_affine_separate` | 2% | ~95.4% | ~1.1% |
| `dendritic_affine_separate` | 5% | ~95.9% | ~0.4% |
| `dendritic_affine_separate` | 10% | ~96.0% | ~0.4% |

---

## Attribution

This project was developed under **OPAL-dev / OPAL.inc** as an independent research exploration on contextual routing, continual learning, and dendritic-inspired architectures.

Main research and implementation: **Yifit / OPAL-dev**.

Experimental design and analysis were assisted by multiple AI systems and cross-checked through iterative review.

---

## Citation

If you use this repository, please cite it using the metadata in [`CITATION.cff`](./CITATION.cff).

A release DOI can be generated through Zenodo by archiving a GitHub release.

---

## License

This project is released under the MIT License. See [`LICENSE`](./LICENSE).