from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def dedup(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["run_name"], keep="last").copy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", default="")
    args = p.parse_args()
    csv_path = Path(args.input)
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find runs summary CSV at: {csv_path}")
    out_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    if df.empty:
        print("runs_summary.csv is empty; nothing to aggregate.")
        return
    dfd = dedup(df)
    dfd.to_csv(out_dir / "runs_summary_dedup.csv", index=False)
    grouped = dfd.groupby([
        "benchmark","scenario","training_regime","model","variant_name",
        "replay_fraction_per_task","replay_size_per_task"
    ], as_index=False).agg(
        mean_accuracy=("average_accuracy","mean"),
        std_accuracy=("average_accuracy","std"),
        mean_forgetting=("average_forgetting","mean"),
        std_forgetting=("average_forgetting","std"),
        mean_bwt=("backward_transfer","mean"),
        std_bwt=("backward_transfer","std"),
        n=("run_name","size"),
    ).fillna(0.0)
    grouped.to_csv(out_dir / "aggregated_summary.csv", index=False)

    matrix_path = out_dir / "accuracy_matrix_rows.csv"
    if matrix_path.exists():
        mdf = pd.read_csv(matrix_path)
        mdf = mdf.drop_duplicates(subset=["run_name","stage_label","stage_index","eval_task"], keep="last")
        mdf.to_csv(out_dir / "accuracy_matrix_rows_dedup.csv", index=False)
        matrix_summary = mdf.groupby([
            "benchmark","scenario","training_regime","model","variant_name",
            "replay_fraction_per_task","replay_size_per_task",
            "stage_label","stage_index","eval_task"
        ], as_index=False).agg(
            mean_accuracy=("accuracy","mean"),
            std_accuracy=("accuracy","std"),
            mean_positive_rate=("positive_rate","mean"),
            mean_mean_logit=("mean_logit","mean"),
            mean_std_logit=("std_logit","mean"),
            n=("run_name","size"),
        ).fillna(0.0)
        matrix_summary.to_csv(out_dir / "accuracy_matrix_summary.csv", index=False)
        final_mask = matrix_summary["stage_label"].str.startswith("after_task_")
        if final_mask.any():
            max_stage = matrix_summary.loc[final_mask, "stage_index"].max()
            final_df = matrix_summary[final_mask & (matrix_summary["stage_index"] == max_stage)].copy()
            final_df.to_csv(out_dir / "final_task_accuracy.csv", index=False)

    gate_path = out_dir / "gate_report_similarity.csv"
    if gate_path.exists():
        gdf = pd.read_csv(gate_path)
        gdf = gdf.drop_duplicates(subset=["run_name","layer_idx","task_i","task_j"], keep="last")
        pair_summary = gdf.groupby([
            "benchmark","scenario","training_regime","model","variant_name",
            "replay_fraction_per_task","replay_size_per_task",
            "layer_idx","task_i","task_j","is_mirror_pair"
        ], as_index=False).agg(
            mean_cosine_similarity=("cosine_similarity","mean"),
            std_cosine_similarity=("cosine_similarity","std"),
            n=("run_name","size"),
        ).fillna(0.0)
        pair_summary.to_csv(out_dir / "gate_similarity_by_task_pair.csv", index=False)

    buffer_path = out_dir / "buffer_stats_rows.csv"
    if buffer_path.exists():
        bdf = pd.read_csv(buffer_path)
        bdf = bdf.drop_duplicates(subset=["run_name","stage_label","buffer_task_id"], keep="last")
        buffer_summary = bdf.groupby([
            "benchmark","scenario","training_regime","model","variant_name",
            "replay_fraction_per_task","replay_size_per_task",
            "stage_label","stage_index","buffer_task_id"
        ], as_index=False).agg(
            mean_buffer_total=("buffer_total","mean"),
            mean_buffer_task_count=("buffer_task_count","mean"),
            n=("run_name","size"),
        ).fillna(0.0)
        buffer_summary.to_csv(out_dir / "buffer_stats_summary.csv", index=False)

    print(f"Saved aggregated summary to: {out_dir / 'aggregated_summary.csv'}")


if __name__ == "__main__":
    main()
