$Seeds = 0..4
foreach ($s in $Seeds) {
  python -m src.main --benchmark sdfc --scenario shared_head --training-regime joint --tasks 4 --epochs 5 --batch-size 128 --model mlp --seed $s --output-dir ./results_shared_joint --run-name "joint_mlp_seed$s"
  python -m src.main --benchmark sdfc --scenario shared_head --training-regime joint --tasks 4 --epochs 5 --batch-size 128 --model dendritic_no_context --seed $s --output-dir ./results_shared_joint --run-name "joint_noctx_seed$s"
  python -m src.main --benchmark sdfc --scenario shared_head --training-regime joint --tasks 4 --epochs 5 --batch-size 128 --model dendritic_unlock_centered --apical-lr-mult 10 --apical-init-gain 4 --seed $s --output-dir ./results_shared_joint --run-name "joint_unlock_seed$s"
  python -m src.main --benchmark sdfc --scenario shared_head --training-regime joint --tasks 4 --epochs 5 --batch-size 128 --model film --apical-init-gain 4 --seed $s --output-dir ./results_shared_joint --run-name "joint_film_seed$s"
}
python .\scripts\aggregate_results.py --input .\results_shared_joint\runs_summary.csv --output-dir .\results_shared_joint
