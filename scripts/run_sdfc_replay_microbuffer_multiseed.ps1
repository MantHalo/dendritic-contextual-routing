# Micro-buffer sweep for paper/repo finalization.
# Default n_train=10000 per task:
# 2% = 200 examples/task, 5% = 500, 10% = 1000.
# If n_train=400, this becomes 8 / 20 / 40 examples/task.
$Seeds = 0..4
$Fractions = @(0.0, 0.02, 0.05, 0.10)
$Models = @("film_full", "dendritic_affine_separate")

foreach ($s in $Seeds) {
  foreach ($m in $Models) {
    foreach ($f in $Fractions) {
      $pct = [int]([math]::Round($f * 100))
      $short = if ($m -eq "film_full") { "film" } else { "affine" }
      python -m src.main --benchmark sdfc --scenario shared_head --training-regime sequential --tasks 4 --epochs 5 --batch-size 128 --model $m --apical-init-gain 4 --replay-fraction-per-task $f --seed $s --output-dir ./results_seq_microbuffer --run-name "seq_${short}_r${pct}pct_seed$s"
    }
  }
}
python .\scripts\aggregate_results.py --input .\results_seq_microbuffer\runs_summary.csv --output-dir .\results_seq_microbuffer
