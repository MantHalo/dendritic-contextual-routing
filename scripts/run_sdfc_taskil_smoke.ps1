$Common = @(
  "--benchmark","sdfc",
  "--scenario","task_il",
  "--tasks","4",
  "--epochs","1",
  "--batch-size","128",
  "--output-dir",".\results_taskil_smoke"
)
python -m src.main --make-benchmark --benchmark-seed 12345
$Runs = @(
  @("--model","mlp","--run-name","sdfc_taskil_mlp_smoke"),
  @("--model","dendritic_no_context","--run-name","sdfc_taskil_noctx_smoke"),
  @("--model","dendritic_unlock_centered","--apical-lr-mult","10","--apical-init-gain","4","--centered-gate","--run-name","sdfc_taskil_dendritic_unlock_smoke"),
  @("--model","film","--apical-lr-mult","10","--apical-init-gain","4","--run-name","sdfc_taskil_film_smoke")
)
foreach ($r in $Runs) { python -m src.main @Common @r }
python .\scripts\aggregate_results.py --input .\results_taskil_smoke\runs_summary.csv --output-dir .\results_taskil_smoke
