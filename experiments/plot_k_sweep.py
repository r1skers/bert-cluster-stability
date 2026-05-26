r"""Plot K sweep under fixed representation recipe.

Input:
    outputs/tables/k_sweep.csv

Output:
    outputs/figures/transforms/k_sweep_alignment.png

Run from repo root:
    .\.venv\Scripts\python.exe experiments\plot_k_sweep.py
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_COLORS = {
    "pretrained": "tab:green",
    "random_seed1": "tab:gray",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="outputs/tables/k_sweep.csv")
    p.add_argument("--outdir", default="outputs/figures/transforms")
    return p.parse_args()


def load_rows(path: str):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate(rows, metric: str):
    by = defaultdict(list)
    ks = sorted({int(row["k"]) for row in rows})
    models = sorted({row["model"] for row in rows})
    for row in rows:
        by[(row["model"], int(row["k"]))].append(float(row[metric]))
    return models, ks, by


def line_panel(ax, rows, metric: str, title: str, ylabel: str):
    models, ks, by = aggregate(rows, metric)
    for model in models:
        means = np.array([np.mean(by[(model, k)]) for k in ks])
        stds = np.array([np.std(by[(model, k)]) for k in ks])
        color = MODEL_COLORS.get(model, None)
        ax.plot(ks, means, marker="o", label=model, color=color)
        ax.fill_between(ks, means - stds, means + stds, color=color, alpha=0.18)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("K")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    line_panel(axes[0], rows, "nmi", "NMI vs K", "NMI")
    line_panel(axes[1], rows, "purity", "Purity vs K", "Purity")
    line_panel(axes[2], rows, "silhouette", "Silhouette vs K", "Silhouette")
    axes[0].legend()
    fig.suptitle("Fixed recipe: layer12 + whiten100_l2 + spherical KMeans")
    fig.tight_layout()

    outpath = outdir / "k_sweep_alignment.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")


if __name__ == "__main__":
    main()
