$Out = ".\results_film_dissection_smoke"
python -m src.main --make-benchmark --benchmark-seed 12345
python -m src.main --benchmark sdfc --scenario shared_head --training-regime sequential --tasks 4 --epochs 1 --batch-size 128 --model film_full --apical-init-gain 4 --seed 0 --output-dir $Out --run-name "seq_film_full_smoke"
python -m src.main --benchmark sdfc --scenario shared_head --training-regime sequential --tasks 4 --epochs 1 --batch-size 128 --model film_additive_only --apical-init-gain 4 --seed 0 --output-dir $Out --run-name "seq_film_addonly_smoke"
python -m src.main --benchmark sdfc --scenario shared_head --training-regime sequential --tasks 4 --epochs 1 --batch-size 128 --model film_multiplicative_only --apical-init-gain 4 --seed 0 --output-dir $Out --run-name "seq_film_mulonly_smoke"
python -m src.main --benchmark sdfc --scenario shared_head --training-regime sequential --tasks 4 --epochs 1 --batch-size 128 --model dendritic_unlock_centered --apical-lr-mult 10 --apical-init-gain 4 --seed 0 --output-dir $Out --run-name "seq_unlock_smoke"
python -m src.main --benchmark sdfc --scenario shared_head --training-regime sequential --tasks 4 --epochs 1 --batch-size 128 --model dendritic_affine_separate --apical-lr-mult 10 --apical-init-gain 4 --seed 0 --output-dir $Out --run-name "seq_affine_smoke"
python .\scripts\aggregate_results.py --input .\results_film_dissection_smoke\runs_summary.csv --output-dir .\results_film_dissection_smoke
