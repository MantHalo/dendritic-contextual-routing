# Release Checklist

## Local Repository

- [ ] `python -m src.main --help` works.
- [ ] `python -m src.main --make-benchmark --benchmark-seed 12345` works.
- [ ] `python .\scripts\make_readme_figures.py` works.
- [ ] `python -m pytest -q` works.
- [ ] `notebooks/quick_sdfc_demo.ipynb` runs top to bottom.
- [ ] `git status` returns a clean working tree.
- [ ] Branch is `main`.

## Repository Structure

- [ ] `src/`
- [ ] `scripts/`
- [ ] `configs/`
- [ ] `artifacts/`
- [ ] `docs/`
- [ ] `docs/articles/`
- [ ] `notebooks/`
- [ ] `paper/`
- [ ] `results/raw_csv/`
- [ ] `results/main_tables/`
- [ ] `README.md`
- [ ] `docs/ANALYSIS.md`
- [ ] `paper/preprint_dendritic_contextual_routing.md`
- [ ] `notebooks/quick_sdfc_demo.ipynb`
- [ ] `CITATION.cff`
- [ ] `LICENSE`
- [ ] `.gitignore`
- [ ] `requirements.txt`
- [ ] `requirements-dev.txt`

## Final Scripts

Keep:

- [ ] `scripts/aggregate_results.py`
- [ ] `scripts/make_readme_figures.py`
- [ ] `scripts/run_sdfc_replay_joint_smoke.ps1`
- [ ] `scripts/run_sdfc_replay_joint_multiseed.ps1`
- [ ] `scripts/run_sdfc_replay_microbuffer_smoke.ps1`
- [ ] `scripts/run_sdfc_replay_microbuffer_multiseed.ps1`

## GitHub

- [ ] Repository pushed to `https://github.com/MantHalo/dendritic-contextual-routing`.
- [ ] Description set.
- [ ] Topics added.
- [ ] README renders correctly.
- [ ] README figures are large enough to read but do not dominate the reading flow.
- [ ] `docs/ANALYSIS.md` shows the accuracy matrices and gate-similarity diagnostics.
- [ ] License detected.
- [ ] Citation button works.
- [ ] CI workflow passes.

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
feature-conflict
```

## GitHub Release

- [ ] Create release `v0.2.0`.
- [ ] Title: `Dendritic Contextual Routing v0.2.0`.
- [ ] Mention the README/figure refresh, SDFC benchmark framing, final replay sweep, and main 2% replay result in release notes.

## Zenodo

- [ ] Connect GitHub repository to Zenodo.
- [ ] Archive release `v0.2.0`.
- [ ] Verify DOI.
- [ ] Add DOI badge to README after Zenodo creates it.

## Optional

- [ ] Deposit on HAL.
- [ ] Prepare arXiv later if endorsement is available.
