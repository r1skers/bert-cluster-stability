r"""Plot cached layer-pooling sweep results.

Input:
    outputs/tables/clustering/pooling_sweep.csv

Output:
    outputs/figures/transforms/pooling_sweep_alignment.png

Run from repo root:
    .\.venv\Scripts\python.exe experiments\clustering\plot_pooling_sweep.py
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ORDER = ["layer12", "last4_mean", "last4_concat"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="outputs/tables/clustering/pooling_sweep.csv")
    p.add_argument("--outdir", default="outputs/figures/transforms")
    return p.parse_args()


def load_rows(path: str):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate(rows, metric: str):
    by = defaultdict(list)
    for row in rows:
        by[row["pooling"]].append(float(row[metric]))
    names = [name for name in ORDER if name in by]
    means = np.array([np.mean(by[name]) for name in names])
    stds = np.array([np.std(by[name]) for name in names])
    return names, means, stds


def bar_panel(ax, rows, metric: str, title: str, ylabel: str):
    names, means, stds = aggregate(rows, metric)
    colors = ["tab:green" if name == "layer12" else "tab:gray" for name in names]
    ax.bar(np.arange(len(names)), means, yerr=stds, color=colors, alpha=0.85, capsize=3)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    bar_panel(axes[0], rows, "nmi", "NMI by pooling", "NMI")
    bar_panel(axes[1], rows, "purity", "Purity by pooling", "Purity")
    fig.suptitle("Layer 12 remains best after fixed whitening + spherical KMeans")
    fig.tight_layout()

    outpath = outdir / "pooling_sweep_alignment.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")


if __name__ == "__main__":
    main()
