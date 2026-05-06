# Results Section — Dendritic v2 / SDFC Shared-Head

## Controlled feature-conflict benchmark

We evaluate contextual routing on SDFC shared-head, a controlled benchmark in which the same input dimensions must be interpreted differently across tasks. This setup creates explicit feature-sign conflicts, including two mirror-task pairs: task 0 vs task 3 and task 1 vs task 2.

The purpose of this benchmark is to distinguish models that merely share a representation from models that can condition their computation on the task context.

## Contextual affine modulation is necessary

In joint training, models with useful contextual conditioning reach approximately 96% final accuracy, while models without effective conditioning remain at chance level. This confirms that the benchmark cannot be solved by a context-free shared representation alone.

Earlier ablations showed that neither additive-only nor multiplicative-only conditioning is sufficient to match the full model. The best-performing primitive combines additive and multiplicative contextual modulation:

```text
h = gamma(context) * h_basal + beta(context)
```

The separated affine dendritic variant implements the same functional primitive:

```text
h = g(context) ⊙ h_basal + a(context)
```

Across all replay budgets, `film_full` and `dendritic_affine_separate` remain statistically indistinguishable. This indicates that the relevant primitive is the combined additive–multiplicative modulation rather than the specific architectural framing.

## Sequential training without memory is fragile

In sequential training without replay, both `film_full` and `dendritic_affine_separate` reach only about 64% final average accuracy and suffer roughly 43% forgetting. This shows that contextual modulation alone is not sufficient to preserve earlier tasks under sequential feature-sign interference.

The oldest task is especially affected. For `film_full`, task 0 falls to approximately 28% final accuracy after the full sequence, despite being learned well initially. This task is also part of an antagonistic mirror pair with task 3.

## Micro-replay nearly closes the gap to joint training

A replay buffer containing only 2% of each task’s training set reduces forgetting from approximately 43% to nearly 1%, while raising final accuracy from about 64% to 95.4%, nearly matching joint training.

For `film_full`, task 0 recovers from 28% to 94% with a 2% replay buffer. This shows that the memory buffer directly counteracts the destructive sequential interference induced by feature-sign reversal.

Increasing the buffer beyond 2% yields only marginal gains:

- 2% replay: approximately 95.4% accuracy
- 5% replay: approximately 95.9% accuracy
- 10% replay: approximately 96.0% accuracy

Thus, most of the benefit is obtained at the smallest nonzero replay budget.

## Architecture equivalence under replay

The comparison between `film_full` and `dendritic_affine_separate` remains nearly identical at every replay level:

- 0%
- 2%
- 5%
- 10%

This confirms that the affine dendritic variant successfully recovers the performance of FiLM-style contextual modulation. The remaining bottleneck is not the architectural realization of the affine primitive, but whether the contextual solution is preserved during sequential learning.

## Gate-similarity diagnostics

Gate-similarity diagnostics show only modest changes under replay. This suggests that replay primarily stabilizes an already useful contextual routing structure rather than creating an entirely new routing organization.

The performance recovery is therefore best interpreted as a stabilization effect: contextual routing provides the right computational degrees of freedom, and micro-replay prevents sequential updates from destroying the solution.

## Summary

The Dendritic v2 experiments establish three main results:

1. Contextual affine modulation is necessary to solve feature-conflict tasks.
2. A separated affine dendritic implementation is functionally equivalent to FiLM in this regime.
3. A very small replay buffer is sufficient to preserve the contextual solution under sequential learning.

In short:

> Contextual affine modulation solves the feature-conflict structure; micro-replay preserves it across sequential learning.