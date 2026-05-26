r"""Plot PCA whitening dimension sweep.

Input:
    outputs/tables/whitening_dim_sweep.csv

Output:
    outputs/figures/transforms/whitening_dim_sweep.png

Run from repo root:
    .\.venv\Scripts\python.exe experiments\plot_whitening_dim_sweep.py
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "pretrained": "tab:green",
    "random_seed1": "tab:gray",
}
LABELS = {
    "pretrained": "pretrained BERT",
    "random_seed1": "random-init BERT",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="outputs/tables/whitening_dim_sweep.csv")
    p.add_argument("--outdir", default="outputs/figures/transforms")
    return p.parse_args()


def load_rows(path: str):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate(rows, metric: str):
    by = defaultdict(list)
    dims = sorted({int(row["whiten_dim"]) for row in rows})
    models = sorted({row["model"] for row in rows})
    for row in rows:
        by[(row["model"], int(row["whiten_dim"]))].append(float(row[metric]))
    return models, dims, by


def line_panel(ax, rows, metric: str, title: str, ylabel: str):
    models, dims, by = aggregate(rows, metric)
    for model in models:
        means = np.array([np.mean(by[(model, dim)]) for dim in dims])
        stds = np.array([np.std(by[(model, dim)]) for dim in dims])
        color = COLORS.get(model)
        ax.plot(dims, means, marker="o", color=color, lw=2, label=LABELS.get(model, model))
        ax.fill_between(dims, means - stds, means + stds, color=color, alpha=0.18)
    ax.set_xscale("log")
    ax.set_xticks(dims)
    ax.set_xticklabels([str(dim) for dim in dims], rotation=30, ha="right")
    ax.set_xlabel("PCA whitening dimension")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    line_panel(axes[0], rows, "nmi", "NMI vs whitening dimension", "NMI")
    line_panel(axes[1], rows, "purity", "Purity vs whitening dimension", "Purity")
    line_panel(axes[2], rows, "silhouette", "Silhouette vs whitening dimension", "Silhouette")
    axes[0].legend(loc="best")
    fig.suptitle("Fixed recipe: layer12 + spherical KMeans, K=20")
    fig.tight_layout()

    outpath = outdir / "whitening_dim_sweep.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")


if __name__ == "__main__":
    main()
