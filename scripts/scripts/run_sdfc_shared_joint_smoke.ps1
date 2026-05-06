$Common = @(
  "--benchmark","sdfc",
  "--scenario","shared_head",
  "--training-regime","joint",
  "--tasks","4",
  "--epochs","1",
  "--batch-size","128",
  "--output-dir","./results_shared_joint_smoke"
)

$Runs = @(
  @("--model","mlp","--seed","0","--run-name","joint_mlp_smoke"),
  @("--model","dendritic_no_context","--seed","0","--run-name","joint_noctx_smoke"),
  @("--model","dendritic_unlock_centered","--seed","0","--apical-lr-mult","10","--apical-init-gain","4","--run-name","joint_unlock_smoke"),
  @("--model","film","--seed","0","--apical-init-gain","4","--run-name","joint_film_smoke")
)

foreach ($r in $Runs) { python -m src.main @Common @r }
python .\scripts\aggregate_results.py --input .\results_shared_joint_smoke\runs_summary.csv --output-dir .\results_shared_joint_smoke
