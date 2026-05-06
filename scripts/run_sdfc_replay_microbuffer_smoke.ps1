# Fast smoke test for the replay micro-buffer sweep.
$Fractions = @(0.0, 0.02)
$Models = @("film_full", "dendritic_affine_separate")
$s = 0
foreach ($m in $Models) {
  foreach ($f in $Fractions) {
    $pct = [int]([math]::Round($f * 100))
    $short = if ($m -eq "film_full") { "film" } else { "affine" }
    python -m src.main --benchmark sdfc --scenario shared_head --training-regime sequential --tasks 4 --epochs 1 --batch-size 128 --model $m --apical-init-gain 4 --replay-fraction-per-task $f --seed $s --output-dir ./results_seq_microbuffer_smoke --run-name "smoke_seq_${short}_r${pct}pct_seed$s"
  }
}
python .\scripts\aggregate_results.py --input .\results_seq_microbuffer_smoke\runs_summary.csv --output-dir .\results_seq_microbuffer_smoke
