# Dendritic v2 — final micro-buffer sweep tables

## Main performance table

| model                     | replay   |   replay_size_per_task |   accuracy_% |   forgetting_% |     bwt |   n |
|:--------------------------|:---------|-----------------------:|-------------:|---------------:|--------:|----:|
| dendritic_affine_separate | 0%       |                      0 |        63.83 |          43.2  | -0.432  |   5 |
| dendritic_affine_separate | 2%       |                    200 |        95.41 |           1.06 | -0.0106 |   5 |
| dendritic_affine_separate | 5%       |                    500 |        95.91 |           0.42 | -0.0033 |   5 |
| dendritic_affine_separate | 10%      |                   1000 |        96.02 |           0.36 | -0.0021 |   5 |
| film_full                 | 0%       |                      0 |        63.91 |          43.16 | -0.4316 |   5 |
| film_full                 | 2%       |                    200 |        95.44 |           1.05 | -0.0105 |   5 |
| film_full                 | 5%       |                    500 |        95.91 |           0.49 | -0.0041 |   5 |
| film_full                 | 10%      |                   1000 |        96.02 |           0.31 | -0.0019 |   5 |

## Final per-task accuracy

| model                     | replay   |   replay_size_per_task |   eval_task |   accuracy_% |
|:--------------------------|:---------|-----------------------:|------------:|-------------:|
| dendritic_affine_separate | 0%       |                      0 |           0 |        28.35 |
| dendritic_affine_separate | 0%       |                      0 |           1 |        54.92 |
| dendritic_affine_separate | 0%       |                      0 |           2 |        75.82 |
| dendritic_affine_separate | 0%       |                      0 |           3 |        96.24 |
| dendritic_affine_separate | 2%       |                    200 |           0 |        94.19 |
| dendritic_affine_separate | 2%       |                    200 |           1 |        95.73 |
| dendritic_affine_separate | 2%       |                    200 |           2 |        95.85 |
| dendritic_affine_separate | 2%       |                    200 |           3 |        95.86 |
| dendritic_affine_separate | 5%       |                    500 |           0 |        95.42 |
| dendritic_affine_separate | 5%       |                    500 |           1 |        96.23 |
| dendritic_affine_separate | 5%       |                    500 |           2 |        96.29 |
| dendritic_affine_separate | 5%       |                    500 |           3 |        95.69 |
| dendritic_affine_separate | 10%      |                   1000 |           0 |        95.65 |
| dendritic_affine_separate | 10%      |                   1000 |           1 |        96.25 |
| dendritic_affine_separate | 10%      |                   1000 |           2 |        96.4  |
| dendritic_affine_separate | 10%      |                   1000 |           3 |        95.78 |
| film_full                 | 0%       |                      0 |           0 |        27.98 |
| film_full                 | 0%       |                      0 |           1 |        55.03 |
| film_full                 | 0%       |                      0 |           2 |        76.35 |
| film_full                 | 0%       |                      0 |           3 |        96.29 |
| film_full                 | 2%       |                    200 |           0 |        94.26 |
| film_full                 | 2%       |                    200 |           1 |        95.69 |
| film_full                 | 2%       |                    200 |           2 |        95.86 |
| film_full                 | 2%       |                    200 |           3 |        95.93 |
| film_full                 | 5%       |                    500 |           0 |        95.49 |
| film_full                 | 5%       |                    500 |           1 |        96.19 |
| film_full                 | 5%       |                    500 |           2 |        96.22 |
| film_full                 | 5%       |                    500 |           3 |        95.75 |
| film_full                 | 10%      |                   1000 |           0 |        95.67 |
| film_full                 | 10%      |                   1000 |           1 |        96.3  |
| film_full                 | 10%      |                   1000 |           2 |        96.35 |
| film_full                 | 10%      |                   1000 |           3 |        95.74 |

## Mirror-pair final accuracy

| model                     | replay   |   replay_size |   mirror_pair_0_3_accuracy_% |   mirror_pair_1_2_accuracy_% |
|:--------------------------|:---------|--------------:|-----------------------------:|-----------------------------:|
| film_full                 | 0%       |             0 |                        62.14 |                        65.69 |
| film_full                 | 2%       |           200 |                        95.1  |                        95.78 |
| film_full                 | 5%       |           500 |                        95.62 |                        96.21 |
| film_full                 | 10%      |          1000 |                        95.7  |                        96.32 |
| dendritic_affine_separate | 0%       |             0 |                        62.3  |                        65.37 |
| dendritic_affine_separate | 2%       |           200 |                        95.02 |                        95.79 |
| dendritic_affine_separate | 5%       |           500 |                        95.56 |                        96.26 |
| dendritic_affine_separate | 10%      |          1000 |                        95.71 |                        96.32 |

## Gate similarity summary used for Figure 3

| model                     |   replay_fraction_per_task | pair_type   |   mean_cosine_similarity |   std_cosine_similarity |
|:--------------------------|---------------------------:|:------------|-------------------------:|------------------------:|
| dendritic_affine_separate |                       0    | mirror      |                 0.82091  |              0.0195532  |
| dendritic_affine_separate |                       0    | non-mirror  |                 0.824731 |              0.00708413 |
| dendritic_affine_separate |                       0.02 | mirror      |                 0.808485 |              0.0145441  |
| dendritic_affine_separate |                       0.02 | non-mirror  |                 0.809807 |              0.0117884  |
| film_full                 |                       0    | mirror      |                 0.813569 |              0.0165747  |
| film_full                 |                       0    | non-mirror  |                 0.817259 |              0.00866436 |
| film_full                 |                       0.02 | mirror      |                 0.802622 |              0.0139235  |
| film_full                 |                       0.02 | non-mirror  |                 0.80494  |              0.0134905  |
