from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"
TABLE_DIR = ROOT / "results" / "main_tables"


def _setup() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "font.size": 13,
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
        }
    )


def _box(ax, xy, wh, text, face, edge="#1f2937", fontsize=13):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.025",
        linewidth=1.6,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, weight="bold")
    return patch


def _arrow(ax, start, end, color="#374151"):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "->",
            "lw": 2.0,
            "color": color,
            "shrinkA": 4,
            "shrinkB": 4,
        },
    )


def make_overview() -> Path:
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.93,
        "Context transforms the representation before the shared head",
        ha="center",
        va="center",
        fontsize=20,
        weight="bold",
    )

    _box(ax, (0.05, 0.52), (0.18, 0.18), "Input\nfeatures", "#e0f2fe", "#0369a1")
    _box(ax, (0.30, 0.52), (0.20, 0.18), "Basal\nrepresentation", "#dcfce7", "#15803d")
    _box(ax, (0.58, 0.52), (0.20, 0.18), "Affine\nmodulation", "#fef3c7", "#b45309")
    _box(ax, (0.84, 0.52), (0.12, 0.18), "Shared\nhead", "#fee2e2", "#b91c1c")

    _box(ax, (0.35, 0.18), (0.28, 0.16), "Task context\nselects gamma and beta", "#ede9fe", "#6d28d9", fontsize=12)

    _arrow(ax, (0.23, 0.61), (0.30, 0.61))
    _arrow(ax, (0.50, 0.61), (0.58, 0.61))
    _arrow(ax, (0.78, 0.61), (0.84, 0.61))
    _arrow(ax, (0.49, 0.34), (0.64, 0.52), color="#6d28d9")

    ax.text(
        0.68,
        0.43,
        "h = gamma(context) * h_basal + beta(context)",
        ha="center",
        va="center",
        fontsize=14,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )

    out = FIG_DIR / "readme_overview.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def make_feature_conflict() -> Path:
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.93, "SDFC: same features, conflicting task rules", ha="center", va="center", fontsize=20, weight="bold")

    _box(ax, (0.08, 0.58), (0.24, 0.18), "Task 0\nz0 + z1 > 0", "#dbeafe", "#1d4ed8")
    _box(ax, (0.08, 0.24), (0.24, 0.18), "Task 3\n-z0 - z1 > 0", "#fee2e2", "#b91c1c")
    _box(ax, (0.39, 0.41), (0.20, 0.18), "Same input\ndimensions", "#f8fafc", "#475569")
    _box(ax, (0.68, 0.58), (0.24, 0.18), "Context says\ninterpret as T0", "#dcfce7", "#15803d")
    _box(ax, (0.68, 0.24), (0.24, 0.18), "Context says\ninterpret as T3", "#fef3c7", "#b45309")

    _arrow(ax, (0.32, 0.67), (0.39, 0.53), color="#1d4ed8")
    _arrow(ax, (0.32, 0.33), (0.39, 0.47), color="#b91c1c")
    _arrow(ax, (0.59, 0.53), (0.68, 0.67), color="#15803d")
    _arrow(ax, (0.59, 0.47), (0.68, 0.33), color="#b45309")

    ax.text(
        0.5,
        0.12,
        "A context-free shared representation is pulled toward incompatible interpretations.",
        ha="center",
        va="center",
        fontsize=14,
    )

    out = FIG_DIR / "sdfc_feature_conflict.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def make_gate_similarity_readable() -> Path:
    csv_path = TABLE_DIR / "table_gate_similarity_mirror_vs_nonmirror.csv"
    df = pd.read_csv(csv_path)

    df = df[df["replay_fraction_per_task"].isin([0.0, 0.02])].copy()
    df["model_label"] = df["model"].map(
        {
            "film_full": "FiLM",
            "dendritic_affine_separate": "Dendritic affine",
        }
    ).fillna(df["model"])
    df["replay_label"] = df["replay_fraction_per_task"].map({0.0: "0%", 0.02: "2%"}).fillna(df["replay_fraction_per_task"].astype(str))

    groups = df[["model_label", "replay_label"]].drop_duplicates().reset_index(drop=True)
    x = np.arange(len(groups))
    width = 0.34

    fig, ax = plt.subplots(figsize=(12, 5.4))
    colors = {"mirror": "#2563eb", "non-mirror": "#f97316"}
    offsets = {"mirror": -width / 2, "non-mirror": width / 2}

    for pair_type in ["mirror", "non-mirror"]:
        means = []
        stds = []
        for _, group in groups.iterrows():
            row = df[
                (df["model_label"] == group["model_label"])
                & (df["replay_label"] == group["replay_label"])
                & (df["pair_type"] == pair_type)
            ].iloc[0]
            means.append(row["mean_cosine_similarity"])
            stds.append(row["std_cosine_similarity"])
        ax.bar(
            x + offsets[pair_type],
            means,
            width,
            label=pair_type,
            color=colors[pair_type],
            yerr=stds,
            capsize=4,
        )

    ax.set_title("Gate similarity remains high across mirror and non-mirror pairs")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_ylim(0.75, 0.90)
    ax.set_xticks(x)
    ax.set_xticklabels(groups["model_label"] + "\n" + groups["replay_label"] + " replay")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    out = FIG_DIR / "fig3_gate_similarity_readable.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def make_replay_summary() -> Path:
    csv_path = TABLE_DIR / "table_main_performance.csv"
    df = pd.read_csv(csv_path).copy()
    summary = (
        df.groupby("replay_size_per_task", as_index=False)
        .agg(
            accuracy_mean=("accuracy_%", "mean"),
            forgetting_mean=("forgetting_%", "mean"),
        )
        .sort_values("replay_size_per_task")
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), sharex=True)
    metrics = [
        ("accuracy", "Final accuracy (%)", "Accuracy rises sharply with 2% replay"),
        ("forgetting", "Average forgetting (%)", "Forgetting nearly disappears"),
    ]

    for ax, (prefix, ylabel, title) in zip(axes, metrics):
        x = summary["replay_size_per_task"]
        y = summary[f"{prefix}_mean"]
        ax.plot(x, y, marker="o", linewidth=2.6, markersize=6.0, color="#1d4ed8")
        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_xlabel("Replay examples per task")
        ax.set_ylabel(ylabel)
        ax.set_xticks([0, 200, 500, 1000])
        ax.grid(axis="y", alpha=0.28)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylim(55, 100)
    axes[1].set_ylim(0, 50)
    fig.suptitle("2% micro-replay preserves contextual routing", fontsize=15, weight="bold", y=1.02)
    fig.text(
        0.5,
        -0.02,
        "Mean of FiLM and dendritic affine; the two curves overlap almost exactly in the final sweep.",
        ha="center",
        fontsize=10,
        color="#374151",
    )

    out = FIG_DIR / "fig1_replay_summary.png"
    fig.tight_layout(pad=1.2)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    _setup()
    outputs = [make_overview(), make_feature_conflict(), make_replay_summary(), make_gate_similarity_readable()]
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
