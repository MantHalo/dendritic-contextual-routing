# Release Checklist

## Before publishing the repository

### Code

- [ ] Remove old broken packs and temporary zip files.
- [ ] Keep only the final clean source tree.
- [ ] Ensure `src.main` runs from the repository root.
- [ ] Ensure all scripts use valid Windows paths.
- [ ] Ensure aggregation scripts deduplicate by `run_name`.

### Results

- [ ] Keep final CSVs for the 0/2/5/10% micro-buffer sweep.
- [ ] Keep joint-training reference CSVs.
- [ ] Move old 0/5/20% sweep to an archive or remove it from main results.
- [ ] Ensure no final table mixes old and new replay budgets.

### Figures

- [ ] Add Figure 1a: accuracy vs replay.
- [ ] Add Figure 1b: forgetting vs replay.
- [ ] Add Figure 2: accuracy matrices.
- [ ] Add Figure 3: gate similarity mirror vs non-mirror.
- [ ] Check all figure captions.

### Documentation

- [ ] README.md
- [ ] README_REPRODUCIBILITY.md
- [ ] EXPERIMENT_LOG.md
- [ ] results_section_dendritic_v2.md
- [ ] requirements.txt
- [ ] LICENSE
- [ ] CITATION.cff

### GitHub release

- [ ] Create public GitHub repository.
- [ ] Push clean code and results.
- [ ] Add final figures.
- [ ] Tag release as `v0.1.0`.
- [ ] Write release notes.
- [ ] Connect repository to Zenodo.
- [ ] Archive release and obtain DOI.

### Optional

- [ ] Deposit manuscript on HAL.
- [ ] Seek arXiv endorsement later if needed.