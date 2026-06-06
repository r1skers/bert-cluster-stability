r"""Plot resampling stability against topic alignment.

Input:
    outputs/tables/clustering/stability_pilot.csv

Output:
    outputs/figures/transforms/stability_alignment.png
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "pretrained": "tab:blue",
    "random_seed1": "tab:gray",
}
MARKERS = {
    "baseline": "o",
    "best": "s",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="outputs/tables/clustering/stability_pilot.csv")
    p.add_argument("--outdir", default="outputs/figures/transforms")
    p.add_argument("--filename", default="stability_alignment.png")
    return p.parse_args()


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def label_for(row):
    model = "pretrained" if row["model"] == "pretrained" else "random-init"
    recipe = "baseline" if row["recipe"] == "baseline" else "best recipe"
    return f"{model}, {recipe}"


def main():
    args = parse_args()
    rows = load_rows(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    x = np.arange(len(rows))
    labels = [label_for(row) for row in rows]
    nmi_means = [float(row["run_nmi_mean"]) for row in rows]
    nmi_stds = [float(row["run_nmi_std"]) for row in rows]
    ari_means = [float(row["mean_ari"]) for row in rows]
    ari_stds = [float(row["std_ari"]) for row in rows]
    colors = [COLORS[row["model"]] for row in rows]

    axes[0].bar(x - 0.18, nmi_means, 0.36, yerr=nmi_stds, color=colors, alpha=0.85, capsize=3, label="NMI")
    axes[0].bar(x + 0.18, ari_means, 0.36, yerr=ari_stds, color=colors, alpha=0.35, capsize=3, label="stability ARI")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Alignment and resampling stability")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    for row in rows:
        model = row["model"]
        recipe = row["recipe"]
        axes[1].errorbar(
            float(row["mean_ari"]),
            float(row["run_nmi_mean"]),
            xerr=float(row["std_ari"]),
            yerr=float(row["run_nmi_std"]),
            fmt=MARKERS[recipe],
            color=COLORS[model],
            markersize=8,
            capsize=3,
            label=label_for(row),
        )
    axes[1].set_xlabel("Resampling stability ARI")
    axes[1].set_ylabel("Topic alignment NMI")
    axes[1].set_title("Stability is not the same as semantic alignment")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.suptitle("Subset-resampling stability, B=50, subset=80%, K=20, layer12")
    fig.tight_layout()
    outpath = outdir / args.filename
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")


if __name__ == "__main__":
    main()
