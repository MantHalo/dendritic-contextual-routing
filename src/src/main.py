from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import torch
from .data.sdfc_generator import DEFAULT_BENCHMARK_SEED, ensure_projection, generate_all_splits
from .data.sdfc_dataset import build_data_bundle
from .models.dendritic import DendriticClassifier
from .models.film import FiLMClassifier
from .training.train import train_joint, train_sequential
from .training.diagnostics import summarize_epoch_diagnostics
from .utils.io import ensure_dir, append_row_csv, append_rows_csv, save_json
from .utils.seed import set_seed

MODEL_CHOICES = ["film_full", "dendritic_affine_separate"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--make-benchmark", action="store_true")
    p.add_argument("--benchmark-seed", type=int, default=DEFAULT_BENCHMARK_SEED)
    p.add_argument("--benchmark", choices=["sdfc"], default="sdfc")
    p.add_argument("--scenario", choices=["shared_head"], default="shared_head")
    p.add_argument("--training-regime", choices=["sequential", "joint"], default="sequential")
    p.add_argument("--model", choices=MODEL_CHOICES, default="film_full")
    p.add_argument("--tasks", type=int, default=4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--apical-lr-mult", type=float, default=1.0)
    p.add_argument("--apical-init-gain", type=float, default=1.0)
    p.add_argument("--gate-temperature", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--n-val", type=int, default=2000)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--replay-fraction-per-task", type=float, default=0.0)
    p.add_argument("--replay-size-per-task", type=int, default=-1)
    p.add_argument("--output-dir", type=str, default="./results")
    p.add_argument("--run-name", type=str, default="")
    return p.parse_args()


def apply_gain(model: torch.nn.Module, gain: float) -> None:
    if abs(gain - 1.0) < 1e-8:
        return
    for name, param in model.named_parameters():
        if any(k in name.lower() for k in ("apical", "gamma", "beta", "gate", "add")):
            with torch.no_grad():
                param.mul_(gain)


def build_model(args, input_dim: int):
    if args.model == "film_full":
        m = FiLMClassifier(input_dim, args.hidden_dims, args.scenario, args.tasks, gate_temperature=args.gate_temperature, mode="full")
    elif args.model == "dendritic_affine_separate":
        m = DendriticClassifier(input_dim, args.hidden_dims, args.scenario, args.tasks, gate_temperature=args.gate_temperature, affine=True)
    else:
        raise ValueError(args.model)
    apply_gain(m, args.apical_init_gain)
    return m


def effective_replay_size(args) -> int:
    if args.training_regime != "sequential":
        return 0
    if args.replay_size_per_task >= 0:
        return int(args.replay_size_per_task)
    if args.replay_fraction_per_task <= 0:
        return 0
    return int(round(args.n_train * args.replay_fraction_per_task))


def variant_name(args) -> str:
    replay_size = effective_replay_size(args)
    replay_pct = 100.0 * args.replay_fraction_per_task if args.training_regime == "sequential" else 0.0
    parts = [
        args.model,
        args.scenario,
        args.training_regime,
        f"tasks{args.tasks}",
        f"epochs{args.epochs}",
        "hd" + "-".join(map(str, args.hidden_dims)),
        f"lr{args.lr:.0e}",
        f"aplr{args.apical_lr_mult:g}",
        f"apinit{args.apical_init_gain:g}",
        f"temp{args.gate_temperature:g}",
        f"replaypct{replay_pct:g}",
        f"replaysize{replay_size}",
    ]
    return "_".join(parts)


def enrich_rows(rows, row_meta):
    return [{**r, **row_meta} for r in rows]


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    if args.make_benchmark:
        p = ensure_projection(project_root, args.benchmark_seed)
        print(f"Saved benchmark projection P with shape={p.shape} to {project_root / 'artifacts'}")
        return
    set_seed(args.seed)
    ensure_projection(project_root, args.benchmark_seed)
    output_dir = ensure_dir(args.output_dir)
    splits = generate_all_splits(project_root, args.tasks, args.n_train, args.n_val, args.n_test, args.benchmark_seed, 0.1)
    data_bundle = build_data_bundle(splits, args.tasks)
    model = build_model(args, data_bundle.input_dim)
    print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vname = variant_name(args)
    run_name = args.run_name or f"sdfc_{vname}_seed{args.seed}"
    replay_size = effective_replay_size(args)
    if args.training_regime == "joint":
        result = train_joint(model, data_bundle, args.scenario, args.tasks, args.epochs, args.batch_size, args.lr, args.apical_lr_mult, args.weight_decay, device, run_name, args.benchmark, args.seed)
    else:
        result = train_sequential(model, data_bundle, args.scenario, args.tasks, args.epochs, args.batch_size, args.lr, args.apical_lr_mult, args.weight_decay, device, run_name, args.benchmark, args.seed, replay_size_per_task=replay_size)
    print("\n=== Final summary ===")
    print(f"Average accuracy: {result.summary['average_accuracy']:.4f}")
    print(f"Average forgetting: {result.summary['average_forgetting']:.4f}")
    row = {
        "run_name": run_name,
        "variant_name": vname,
        "benchmark": args.benchmark,
        "scenario": args.scenario,
        "training_regime": args.training_regime,
        "model": args.model,
        "tasks": args.tasks,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_dims": "-".join(map(str, args.hidden_dims)),
        "lr": args.lr,
        "apical_lr_mult": args.apical_lr_mult,
        "apical_init_gain": args.apical_init_gain,
        "gate_temperature": args.gate_temperature,
        "replay_fraction_per_task": args.replay_fraction_per_task,
        "replay_size_per_task": replay_size,
        "n_train_per_task": args.n_train,
        "seed": args.seed,
        "average_accuracy": result.summary["average_accuracy"],
        "average_forgetting": result.summary["average_forgetting"],
        "backward_transfer": result.summary["backward_transfer"],
    }
    append_row_csv(output_dir / "runs_summary.csv", row)
    row_meta = {
        "model": args.model,
        "variant_name": vname,
        "training_regime": args.training_regime,
        "replay_fraction_per_task": args.replay_fraction_per_task,
        "replay_size_per_task": replay_size,
    }
    if result.epoch_rows:
        epoch_rows = enrich_rows(result.epoch_rows, row_meta)
        append_rows_csv(output_dir / "epoch_diagnostics_all.csv", epoch_rows)
        epoch_summary = summarize_epoch_diagnostics(pd.DataFrame(epoch_rows))
        append_rows_csv(output_dir / "epoch_diagnostics_summary.csv", epoch_summary.to_dict(orient="records"))
    if result.gate_rows:
        append_rows_csv(output_dir / "gate_report_layers.csv", enrich_rows(result.gate_rows, row_meta))
    if result.gate_sim_rows:
        append_rows_csv(output_dir / "gate_report_similarity.csv", enrich_rows(result.gate_sim_rows, row_meta))
    if result.matrix_rows:
        append_rows_csv(output_dir / "accuracy_matrix_rows.csv", enrich_rows(result.matrix_rows, row_meta))
    if result.buffer_rows:
        append_rows_csv(output_dir / "buffer_stats_rows.csv", enrich_rows(result.buffer_rows, row_meta))
    payload = {**row, **result.summary}
    save_json(output_dir / f"{run_name}.json", payload)
    print(f"Saved JSON results to: {output_dir / (run_name + '.json')}")
    print(f"Appended summary row to: {output_dir / 'runs_summary.csv'}")


if __name__ == "__main__":
    main()
