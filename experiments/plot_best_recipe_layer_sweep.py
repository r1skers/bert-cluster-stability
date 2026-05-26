r"""Plot layer sweep under the current best clustering recipe.

Input:
    outputs/tables/layer_sweep_best_recipe.csv

Output:
    outputs/figures/transforms/best_recipe_layer_sweep.png

Run from repo root:
    .\.venv\Scripts\python.exe experiments\plot_best_recipe_layer_sweep.py
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LAYERS = list(range(13))
MODELS = ["pretrained", "random_seed1"]
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
    p.add_argument("--csv", default="outputs/tables/layer_sweep_best_recipe.csv")
    p.add_argument("--outdir", default="outputs/figures/transforms")
    return p.parse_args()


def load_rows(path: str):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def per_layer(rows, model: str, metric: str):
    by_layer = defaultdict(list)
    for row in rows:
        if row["model"] == model:
            by_layer[int(row["layer"])].append(float(row[metric]))
    means = np.array([np.mean(by_layer[layer]) for layer in LAYERS])
    stds = np.array([np.std(by_layer[layer]) for layer in LAYERS])
    return means, stds


def line_panel(ax, rows, metric: str, title: str, ylabel: str):
    for model in MODELS:
        means, stds = per_layer(rows, model, metric)
        color = COLORS[model]
        ax.plot(LAYERS, means, marker="o", color=color, lw=2, label=LABELS[model])
        ax.fill_between(LAYERS, means - stds, means + stds, color=color, alpha=0.18)
    ax.set_xticks(LAYERS)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    line_panel(axes[0], rows, "nmi", "NMI vs layer", "NMI")
    line_panel(axes[1], rows, "purity", "Purity vs layer", "Purity")
    line_panel(axes[2], rows, "silhouette", "Silhouette vs layer", "Silhouette")
    axes[0].legend(loc="upper left")
    fig.suptitle(
        "Layer sweep under fixed recipe: whiten100_l2 + spherical KMeans, K=20"
    )
    fig.tight_layout()

    outpath = outdir / "best_recipe_layer_sweep.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")


if __name__ == "__main__":
    main()
