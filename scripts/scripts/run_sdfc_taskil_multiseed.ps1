$Common = @(
  "--benchmark","sdfc",
  "--scenario","task_il",
  "--tasks","4",
  "--epochs","5",
  "--batch-size","128",
  "--output-dir",".\results_taskil"
)
python -m src.main --make-benchmark --benchmark-seed 12345
$Seeds = 0..4
foreach ($s in $Seeds) {
  python -m src.main @Common --model mlp --seed $s --run-name "sdfc_taskil_mlp_seed$s"
  python -m src.main @Common --model dendritic_no_context --seed $s --run-name "sdfc_taskil_noctx_seed$s"
  python -m src.main @Common --model dendritic_unlock_centered --apical-lr-mult 10 --apical-init-gain 4 --centered-gate --seed $s --run-name "sdfc_taskil_dendritic_unlock_seed$s"
  python -m src.main @Common --model film --apical-lr-mult 10 --apical-init-gain 4 --seed $s --run-name "sdfc_taskil_film_seed$s"
}
python .\scripts\aggregate_results.py --input .\results_taskil\runs_summary.csv --output-dir .\results_taskil
