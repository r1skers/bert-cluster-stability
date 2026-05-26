r"""Plot Lloyd vs spherical KMeans comparison.

Input:
    outputs/tables/clusterer_sweep.csv

Output:
    outputs/figures/transforms/clusterer_sweep_alignment.png
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ORDER = ["l2", "centered_l2", "pca100_l2", "whiten50_l2", "whiten100_l2"]
CLUSTERERS = [
    "lloyd",
    "spherical",
    "agglo_cosine",
    "agglo_ward",
    "gmm_diag",
    "gmm_full",
]
COLORS = {
    "lloyd": "tab:blue",
    "spherical": "tab:orange",
    "agglo_cosine": "tab:purple",
    "agglo_ward": "tab:red",
    "gmm_diag": "tab:brown",
    "gmm_full": "tab:pink",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="outputs/tables/clusterer_sweep.csv")
    p.add_argument("--outdir", default="outputs/figures/transforms")
    p.add_argument("--filename", default="clusterer_sweep_alignment.png")
    return p.parse_args()


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def grouped(rows, metric):
    by = defaultdict(list)
    for row in rows:
        by[(row["transform"], row["clusterer"])].append(float(row[metric]))
    return by


def panel(ax, rows, metric, ylabel):
    by = grouped(rows, metric)
    transforms = [tr for tr in ORDER if any(row["transform"] == tr for row in rows)]
    clusterers = [c for c in CLUSTERERS if any(row["clusterer"] == c for row in rows)]
    x = np.arange(len(transforms))
    width = 0.25
    offsets = (np.arange(len(clusterers)) - (len(clusterers) - 1) / 2) * width
    for i, clusterer in enumerate(clusterers):
        means = []
        stds = []
        for tr in transforms:
            vals = by[(tr, clusterer)]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        ax.bar(
            x + offsets[i],
            means,
            width,
            yerr=stds,
            label=clusterer,
            color=COLORS[clusterer],
            alpha=0.85,
            capsize=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(transforms, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)


def main():
    args = parse_args()
    rows = load_rows(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    panel(axes[0], rows, "nmi", "NMI")
    axes[0].set_title("NMI by transform and clusterer")
    panel(axes[1], rows, "purity", "Purity")
    axes[1].set_title("Purity by transform and clusterer")
    axes[0].legend()
    fig.suptitle("Clusterer comparison after representation transforms")
    fig.tight_layout()
    outpath = outdir / args.filename
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")


if __name__ == "__main__":
    main()
