r"""Plot transform sweep results.

Input:
    outputs/tables/clustering/transform_sweep.csv

Outputs:
    outputs/figures/transforms/transform_sweep_alignment.png
    outputs/figures/transforms/transform_sweep_geometry.png

Run from repo root:
    .\.venv\Scripts\python.exe experiments\clustering\plot_transform_sweep.py
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ORDER = [
    "raw",
    "l2",
    "centered_l2",
    "pca50_l2",
    "pca100_l2",
    "whiten50_l2",
    "whiten100_l2",
    "drop_pc1_l2",
    "drop_pc3_l2",
    "drop_pc5_l2",
    "drop_pc10_l2",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="outputs/tables/clustering/transform_sweep.csv")
    p.add_argument("--outdir", default="outputs/figures/transforms")
    return p.parse_args()


def load_rows(path: str):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate(rows, metric: str):
    by = defaultdict(list)
    for row in rows:
        by[row["transform"]].append(float(row[metric]))
    names = [name for name in ORDER if name in by]
    means = np.array([np.mean(by[name]) for name in names])
    stds = np.array([np.std(by[name]) for name in names])
    return names, means, stds


def bar_panel(ax, rows, metric: str, title: str, ylabel: str):
    names, means, stds = aggregate(rows, metric)
    colors = ["tab:blue" if "whiten" not in n else "tab:green" for n in names]
    ax.bar(np.arange(len(names)), means, yerr=stds, color=colors, alpha=0.85, capsize=3)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)


def plot_alignment(rows, outpath: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    bar_panel(axes[0], rows, "nmi", "NMI by transform", "NMI")
    bar_panel(axes[1], rows, "purity", "Purity by transform", "Purity")
    fig.suptitle("L12 KMeans topic alignment improves after PCA whitening")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath}")


def plot_geometry(rows, outpath: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    bar_panel(axes[0], rows, "silhouette", "Silhouette", "higher is better")
    bar_panel(axes[1], rows, "davies_bouldin", "Davies-Bouldin", "lower is better")
    bar_panel(axes[2], rows, "participation_ratio", "Participation ratio", "PR")
    fig.suptitle("Geometry diagnostics by transform")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath}")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_alignment(rows, outdir / "transform_sweep_alignment.png")
    plot_geometry(rows, outdir / "transform_sweep_geometry.png")


if __name__ == "__main__":
    main()
