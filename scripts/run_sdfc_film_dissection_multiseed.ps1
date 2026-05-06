$Seeds = 0..4
$Out = ".\results_film_dissection"
python -m src.main --make-benchmark --benchmark-seed 12345
foreach ($s in $Seeds) {
  python -m src.main --benchmark sdfc --scenario shared_head --training-regime sequential --tasks 4 --epochs 5 --batch-size 128 --model film_full --apical-init-gain 4 --seed $s --output-dir $Out --run-name "seq_film_full_seed$s"
  python -m src.main --benchmark sdfc --scenario shared_head --training-regime sequential --tasks 4 --epochs 5 --batch-size 128 --model film_additive_only --apical-init-gain 4 --seed $s --output-dir $Out --run-name "seq_film_addonly_seed$s"
  python -m src.main --benchmark sdfc --scenario shared_head --training-regime sequential --tasks 4 --epochs 5 --batch-size 128 --model film_multiplicative_only --apical-init-gain 4 --seed $s --output-dir $Out --run-name "seq_film_mulonly_seed$s"
  python -m src.main --benchmark sdfc --scenario shared_head --training-regime sequential --tasks 4 --epochs 5 --batch-size 128 --model dendritic_unlock_centered --apical-lr-mult 10 --apical-init-gain 4 --seed $s --output-dir $Out --run-name "seq_unlock_seed$s"
  python -m src.main --benchmark sdfc --scenario shared_head --training-regime sequential --tasks 4 --epochs 5 --batch-size 128 --model dendritic_affine_separate --apical-lr-mult 10 --apical-init-gain 4 --seed $s --output-dir $Out --run-name "seq_affine_seed$s"
}
python .\scripts\aggregate_results.py --input .\results_film_dissection\runs_summary.csv --output-dir .\results_film_dissection
