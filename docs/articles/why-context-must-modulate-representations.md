# Why Context Must Modulate Representations, Not Just Be Concatenated

Context is often treated as one more input feature. If a model needs to know which task is active, the practical default is simple: append a task identifier, user state, or environment label to the input and let the network learn the conditional behavior.

That default is useful, but it hides a sharper problem. Sometimes context is not just extra information. Sometimes it decides what the same feature means.

This repository studies that problem through SDFC, a compact Same-Dimension Feature Conflict benchmark. In SDFC, every task uses the same input dimensions. What changes is the rule that maps those features to labels. A feature that is positive evidence in one task can become negative evidence in another. Two pairs of tasks are mirror conflicts: task 0 versus task 3, and task 1 versus task 2.

That structure turns an abstract representation-learning question into a controlled test:

> When does context need to transform a representation, rather than merely be appended to it?

The current answer is narrow but useful. In SDFC, contextual affine modulation solves the feature-conflict structure. Under sequential training, that solution is fragile without memory. A 2 percent replay buffer, however, nearly closes the gap to joint training.

## The Failure Mode

Most benchmarks let a model discover features whose meaning is reasonably stable. Continual learning makes this harder because tasks arrive over time, but the input dimensions often still point toward compatible semantics. SDFC removes that comfort. The same dimensions stay active, while the correct interpretation changes by task.

That is why concatenation is an incomplete answer. Concatenating context gives the network access to the task identifier, but it does not guarantee that context can reshape the intermediate representation where the conflict occurs. A large network may learn an equivalent computation indirectly, but SDFC asks for the mechanism explicitly.

The useful operation is affine contextual modulation:

```text
h = gamma(context) * h_basal + beta(context)
```

The input pathway computes a basal representation. The context pathway computes multiplicative and additive terms. Those terms transform the hidden representation before the shared head makes a prediction.

The dendritic-inspired implementation in this repository uses the equivalent form:

```text
h = g(context) * h_basal + a(context)
```

The point is not that this dendritic framing beats FiLM. The final results show functional equivalence between `film_full` and `dendritic_affine_separate` in this benchmark. The contribution is more precise: SDFC isolates a setting where context must modulate hidden representations, and the dendritic-inspired model recovers the same affine primitive through separated basal and contextual pathways.

## What The Results Show

The final curated tables are in `results/main_tables/`. In joint training, both contextual affine models solve the benchmark at about 96.3 percent mean final accuracy over 5 seeds.

Sequential learning without replay is very different. With 0 percent replay:

| Model | Final accuracy | Forgetting |
|---|---:|---:|
| `film_full` | 63.91% | 43.16% |
| `dendritic_affine_separate` | 63.83% | 43.20% |

The oldest mirror-conflicted task shows the damage clearly. In `film_full`, task 0 falls to 27.98 percent final accuracy while task 3, its later mirror antagonist, remains at 96.29 percent.

With 2 percent replay, the picture changes:

| Model | Final accuracy | Forgetting |
|---|---:|---:|
| `film_full` | 95.44% | 1.05% |
| `dendritic_affine_separate` | 95.41% | 1.06% |

For `film_full`, task 0 recovers from 27.98 percent to 94.26 percent. For `dendritic_affine_separate`, it recovers from 28.35 percent to 94.19 percent. Increasing replay to 5 or 10 percent gives smaller marginal gains, so the main transition is between no replay and the 2 percent buffer.

## What The Results Do Not Show

The results should not be oversold. They do not show that the dendritic-inspired model is better than FiLM. They do not establish compute efficiency, sparsity, or scaling advantages. They do not solve continual learning in general.

They show something cleaner: when tasks reuse the same features with conflicting meanings, context must be able to transform the representation, and a small replay buffer can preserve that contextual solution under sequential learning.

That is already a useful scientific point. It shifts the discussion from "which label do we give the architecture?" to "which computation must context be able to perform?"

## A Link To Parametric Recall

A related Google Research paper, ["Thinking to Recall: How Reasoning Unlocks Parametric Knowledge in LLMs"](https://research.google/blog/thinking-to-recall-how-reasoning-unlocks-parametric-knowledge-in-llms/), argues that reasoning traces can unlock facts already stored in model parameters. The authors identify two mechanisms: extra computation through a reasoning buffer, and factual priming through related intermediate facts. They also show that hallucinated intermediate facts can increase final-answer hallucinations.

That work is not evidence for SDFC, but it opens a useful bridge. Their setting is closed-book factual recall in language models. Our setting is controlled feature conflict in small neural networks. The common theme is latent access: useful information may be present in the system but inaccessible unless the right context, computation, or memory trace makes it reachable.

For future SDFC work, this suggests a new branch:

- Can neutral extra computation help recover old task rules?
- Can prototype or anchor facts replace some replay examples?
- Can a model verify an intermediate contextual cue before trusting it?
- Is a forgotten rule actually gone, or merely hard to access?

This branch could be called contextual recall or latent access. It would connect replay, context modulation, and test-time computation without pretending they are the same mechanism.

## Reproducibility

The repository includes fixed benchmark artifacts, source code, curated CSVs, figures, and tests. The quick path is:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m src.main --make-benchmark --benchmark-seed 12345
python .\scripts\make_readme_figures.py
python -m pytest -q
```

The notebook `notebooks/quick_sdfc_demo.ipynb` is a short CPU demonstration. It is not meant to reproduce the final multi-seed numbers. The final claims come from the curated CSVs and reproduction scripts.

## Short Version

SDFC makes context functionally necessary because the same input dimensions carry conflicting meanings across tasks. In this setting, the useful primitive is:

```text
h = gamma(context) * h_basal + beta(context)
```

FiLM and the dendritic affine implementation both instantiate that primitive and perform nearly identically. Sequential training damages the solution without replay. A 2 percent replay buffer nearly restores it.

The durable claim is simple: when context changes what features mean, it should modulate the representation used for prediction.
