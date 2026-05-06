$Seeds = 0..4
foreach ($s in $Seeds) {
  python -m src.main --benchmark sdfc --scenario shared_head --training-regime joint --tasks 4 --epochs 5 --batch-size 128 --model film_full --apical-init-gain 4 --seed $s --output-dir ./results_joint_reference --run-name "joint_film_seed$s"
  python -m src.main --benchmark sdfc --scenario shared_head --training-regime joint --tasks 4 --epochs 5 --batch-size 128 --model dendritic_affine_separate --apical-init-gain 4 --seed $s --output-dir ./results_joint_reference --run-name "joint_affine_seed$s"
}
python .\scripts\aggregate_results.py --input .\results_joint_reference\runs_summary.csv --output-dir .\results_joint_reference
