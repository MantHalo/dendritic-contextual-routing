# Experiment Log — Dendritic v2

## Phase 1 — Initial dendritic gating

Initial experiments tested a dendritic-style multiplicative gate:

```text
h = gate(context) * h_basal
```

On PermutedMNIST, this model showed a modest gain over a plain MLP. However, gate diagnostics showed that the learned gates were nearly task-invariant.

## Phase 2 — Mechanistic controls

Controls included parameter-matched MLP, fixed-gate dendritic variant, no-context dendritic variant, FiLM-style conditioning, gate reports, and gradient diagnostics.

The strongest conclusion was that the useful effect on PermutedMNIST came mostly from heterogeneous neuronal scaling, not from learned contextual routing.

## Phase 3 — Apical unlock

Apical unlock produced more differentiated gates, but did not improve performance on PermutedMNIST.

Decision: PermutedMNIST is not the right primary benchmark for contextual routing.

## Phase 4 — SDFC benchmark

SDFC shared-head was introduced to force feature-sign conflicts on the same input dimensions.

Results:

- context-free models remain at chance level;
- context-conditioned models perform above chance;
- full affine modulation is necessary.

Decision: SDFC shared-head is the correct benchmark for testing useful contextual routing.

## Phase 5 — FiLM dissection

Ablations:

- `film_full`
- `film_additive_only`
- `film_multiplicative_only`
- `dendritic_unlock_centered`
- `dendritic_affine_separate`

Result:

- additive-only is insufficient;
- multiplicative-only is insufficient;
- combined affine modulation is necessary;
- dendritic affine separate ≈ film full.

## Phase 6 — Replay

Sequential training remained far below joint training. Replay was introduced to test whether the remaining bottleneck was memory/consolidation.

Results:

- 2% replay nearly closes the gap to joint training;
- 5% and 10% add only marginal gains;
- film_full and dendritic_affine_separate remain equivalent;
- task 0 recovers from about 28% to about 94% with 2% replay.

Decision: the final Dendritic v2 result is contextual affine modulation + micro-replay.