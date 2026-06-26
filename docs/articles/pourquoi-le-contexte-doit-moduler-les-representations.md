# Pourquoi le contexte doit moduler les representations, pas seulement etre concatene

Dans beaucoup de systemes de machine learning, le contexte est traite comme une feature de plus. Si le modele doit savoir quelle tache est active, on ajoute un identifiant de tache a l'entree et on laisse le reseau apprendre le comportement conditionnel.

Cette approche est utile, mais elle cache un probleme plus dur. Parfois, le contexte n'est pas seulement une information supplementaire. Parfois, il decide ce que les memes features veulent dire.

Ce depot etudie ce probleme avec SDFC, un benchmark compact de Same-Dimension Feature Conflict. Dans SDFC, toutes les taches utilisent les memes dimensions d'entree. Ce qui change, c'est la regle qui relie ces features au label. Une feature qui est un indice positif dans une tache peut devenir un indice negatif dans une autre. Deux paires de taches sont des conflits miroir: tache 0 contre tache 3, et tache 1 contre tache 2.

La question devient concrete:

> Quand le contexte doit-il transformer une representation, au lieu d'etre seulement ajoute a l'entree?

La reponse actuelle est limitee mais utile. Dans SDFC, la modulation affine contextuelle resout la structure de conflit. En apprentissage sequentiel, cette solution est fragile sans memoire. Avec un buffer de replay de 2 pour cent, l'ecart avec l'apprentissage joint est presque ferme.

## Le mode d'echec

Dans beaucoup de benchmarks, le sens des features reste relativement stable. L'apprentissage continu complique deja les choses, car les taches arrivent dans le temps. Mais les dimensions d'entree pointent souvent vers des semantiques compatibles.

SDFC retire cette facilite. Les memes dimensions restent actives, mais leur interpretation correcte change selon la tache. C'est pour cela que la concatenation est une reponse incomplete. Ajouter le contexte donne au modele l'information de tache, mais cela ne garantit pas que cette information puisse remodeler la representation intermediaire au bon endroit.

L'operation utile est une modulation affine contextuelle:

```text
h = gamma(contexte) * h_basal + beta(contexte)
```

Le chemin d'entree calcule une representation basale. Le chemin contextuel calcule un terme multiplicatif et un terme additif. Ces termes transforment la representation cachee avant la prediction.

L'implementation inspiree des dendrites utilise une forme equivalente:

```text
h = g(contexte) * h_basal + a(contexte)
```

Le point important n'est pas de dire que cette version bat FiLM. Les resultats montrent au contraire une equivalence fonctionnelle entre `film_full` et `dendritic_affine_separate` dans ce benchmark. La contribution est plus precise: SDFC isole un cas ou le contexte doit moduler les representations, et le modele inspire des dendrites retrouve cette meme primitive affine avec des chemins basal et contextuel separes.

## Ce que montrent les resultats

Les tables finales sont dans `results/main_tables/`. En apprentissage joint, les deux modeles contextuels affines resolvent le benchmark avec environ 96,3 pour cent de precision finale moyenne sur 5 seeds.

En apprentissage sequentiel sans replay, les performances chutent:

| Modele | Precision finale | Oubli |
|---|---:|---:|
| `film_full` | 63.91% | 43.16% |
| `dendritic_affine_separate` | 63.83% | 43.20% |

La tache 0 rend le phenomene tres visible. Pour `film_full`, elle tombe a 27.98 pour cent de precision finale, alors que la tache 3, son antagoniste miroir appris plus tard, reste a 96.29 pour cent.

Avec 2 pour cent de replay, la situation change:

| Modele | Precision finale | Oubli |
|---|---:|---:|
| `film_full` | 95.44% | 1.05% |
| `dendritic_affine_separate` | 95.41% | 1.06% |

Pour `film_full`, la tache 0 remonte de 27.98 pour cent a 94.26 pour cent. Pour `dendritic_affine_separate`, elle remonte de 28.35 pour cent a 94.19 pour cent. Les budgets de replay plus grands apportent surtout des gains marginaux.

## Ce que les resultats ne montrent pas

Il faut garder le cadrage propre. Ces experiences ne montrent pas une superiorite dendritique sur FiLM. Elles ne prouvent pas une efficacite de calcul, une parcimonie, ou une meilleure scalabilite. Elles ne resolvent pas l'apprentissage continu en general.

Elles montrent quelque chose de plus net: quand plusieurs taches reutilisent les memes features avec des significations conflictuelles, le contexte doit pouvoir transformer la representation, et une petite memoire de replay peut preserver cette solution.

Ce resultat deplace la discussion. La question n'est pas seulement le nom de l'architecture. La question est: quelle computation le contexte doit-il pouvoir effectuer?

## Lien avec le rappel parametrique

Un papier Google Research, ["Thinking to Recall: How Reasoning Unlocks Parametric Knowledge in LLMs"](https://research.google/blog/thinking-to-recall-how-reasoning-unlocks-parametric-knowledge-in-llms/), montre que des traces de raisonnement peuvent debloquer des connaissances deja presentes dans les poids d'un modele de langage. Les auteurs distinguent deux mecanismes: un buffer computationnel, et un amorcage factuel par des faits intermediaires lies. Ils montrent aussi que des faits intermediaires hallucines peuvent augmenter les erreurs finales.

Ce papier ne prouve rien directement pour SDFC. Mais il ouvre un pont conceptuel. Eux etudient le rappel factuel dans les LLMs. Ici, nous etudions le conflit de features dans un cadre controle. Le point commun est l'acces latent: une information utile peut etre presente mais difficile a atteindre sans le bon contexte, la bonne computation, ou la bonne trace memoire.

Cela suggere une branche future:

- Est-ce qu'un calcul supplementaire neutre aide a recuperer une ancienne regle de tache?
- Est-ce que des prototypes ou anchors peuvent remplacer une partie du replay?
- Est-ce qu'un modele peut verifier un indice contextuel intermediaire avant de l'utiliser?
- Est-ce qu'une regle oubliee est detruite, ou seulement difficile d'acces?

On peut appeler cette branche contextual recall ou latent access.

## Reproductibilite

Le depot contient les artefacts fixes du benchmark, le code, les CSV finaux, les figures et les tests. Le chemin rapide est:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m src.main --make-benchmark --benchmark-seed 12345
python .\scripts\make_readme_figures.py
python -m pytest -q
```

Le notebook `notebooks/quick_sdfc_demo.ipynb` est une demonstration courte sur CPU. Il ne remplace pas les resultats multi-seed finaux.

## Version courte

SDFC rend le contexte necessaire car les memes dimensions d'entree ont des significations conflictuelles selon la tache. La primitive utile est:

```text
h = gamma(contexte) * h_basal + beta(contexte)
```

FiLM et l'implementation dendritique affine realisent cette primitive et se comportent presque pareil. Sans replay, l'apprentissage sequentiel abime la solution. Avec 2 pour cent de replay, elle est presque restauree.

Conclusion: quand le contexte change le sens des features, il doit moduler la representation utilisee pour predire.
