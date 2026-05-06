$Common = @(
  "--benchmark","sdfc",
  "--scenario","shared_head",
  "--tasks","4",
  "--epochs","5",
  "--batch-size","128",
  "--output-dir",".\results_shared"
)
python -m src.main --make-benchmark --benchmark-seed 12345
$Seeds = 0..4
foreach ($s in $Seeds) {
  python -m src.main @Common --model mlp --seed $s --run-name "sdfc_shared_mlp_seed$s"
  python -m src.main @Common --model dendritic_no_context --seed $s --run-name "sdfc_shared_noctx_seed$s"
  python -m src.main @Common --model dendritic_unlock_centered --apical-lr-mult 10 --apical-init-gain 4 --centered-gate --seed $s --run-name "sdfc_shared_dendritic_unlock_seed$s"
  python -m src.main @Common --model film --apical-lr-mult 10 --apical-init-gain 4 --seed $s --run-name "sdfc_shared_film_seed$s"
}
python .\scripts\aggregate_results.py --input .\results_shared\runs_summary.csv --output-dir .\results_shared
