# Release Checklist

## Local repository

- [ ] `python -m src.main --help` works.
- [ ] `python -m src.main --make-benchmark --benchmark-seed 12345` works.
- [ ] `git status` returns a clean working tree.
- [ ] Branch is `main`.

## Repository structure

- [ ] `src/`
- [ ] `scripts/`
- [ ] `configs/`
- [ ] `artifacts/`
- [ ] `docs/`
- [ ] `paper/`
- [ ] `results/raw_csv/`
- [ ] `results/main_tables/`
- [ ] `README.md`
- [ ] `CITATION.cff`
- [ ] `LICENSE`
- [ ] `.gitignore`
- [ ] `requirements.txt`

## Final scripts only

Keep:

- [ ] `scripts/aggregate_results.py`
- [ ] `scripts/run_sdfc_replay_joint_smoke.ps1`
- [ ] `scripts/run_sdfc_replay_joint_multiseed.ps1`
- [ ] `scripts/run_sdfc_replay_microbuffer_smoke.ps1`
- [ ] `scripts/run_sdfc_replay_microbuffer_multiseed.ps1`

## GitHub

- [ ] Repository pushed to `https://github.com/OPAL-dev/dendritic-contextual-routing`
- [ ] Description set.
- [ ] Topics added.
- [ ] README renders correctly.
- [ ] License detected.
- [ ] Citation button works.

Suggested topics:

```text
continual-learning
contextual-routing
dendritic-networks
film
replay-buffer
pytorch
machine-learning
neural-networks
```

## GitHub release

- [ ] Create release `v0.1.0`.
- [ ] Title: `Dendritic Contextual Routing v0.1.0`.
- [ ] Mention final replay sweep and main result in release notes.

## Zenodo

- [ ] Connect GitHub repository to Zenodo.
- [ ] Archive release `v0.1.0`.
- [ ] Verify DOI.
- [ ] Add DOI badge to README after Zenodo creates it.

## Optional

- [ ] Deposit on HAL.
- [ ] Prepare arXiv later if endorsement is available.