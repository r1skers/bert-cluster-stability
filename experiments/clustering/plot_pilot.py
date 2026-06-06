r"""Plot the W1 pilot result: three figures from one CSV.

Inputs:
    outputs/tables/clustering/pilot_metrics_l2_seeds.csv
        (mode='l2', K=20, 5 KMeans seeds, two model variants)

Outputs (300 dpi PNGs):
    outputs/figures/pilot/pilot_alignment.png
        Main money-figure-in-progress: NMI(L) + purity(L),
        pretrained vs random-init, mean +/- std band over 5 seeds.

    outputs/figures/pilot/pilot_diagnostics.png
        Anisotropy(L) + PR(L), pretrained vs random-init.
        These are K-independent and computed on raw X (one value
        per (model, layer)), so no error band.

    outputs/figures/pilot/pilot_geometry_controls.png
        Silhouette + Davies-Bouldin + Calinski-Harabasz.
        Shown as geometry-only controls: these metrics do not
        explain the late-layer NMI rise as simple Euclidean cluster
        compactness. Some even favor random-init, which is the point.

Run from repo root:
    .\.venv\Scripts\python.exe experiments\clustering\plot_pilot.py
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

# fixed style choices
COLOR_PRE = "tab:blue"
COLOR_RAND = "tab:gray"
BAND_ALPHA = 0.18
LAYERS = list(range(13))
MODEL_PRE = "pretrained"
MODEL_RAND = "random_seed1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        default="outputs/tables/clustering/pilot_metrics_l2_seeds.csv",
        help="Input CSV produced by run_pilot_metrics.py with --seeds 0 1 2 3 4",
    )
    p.add_argument(
        "--outdir",
        default="outputs/figures/pilot",
        help="Directory to write the 3 PNG files into",
    )
    return p.parse_args()


# ----- data loading + per-layer aggregation -----

def load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def per_layer_mean_std(
    rows: List[Dict[str, str]], model: str, metric: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean[L], std[L]) over seeds for one (model, metric)."""
    by_layer = defaultdict(list)
    for r in rows:
        if r["model"] != model:
            continue
        by_layer[int(r["layer"])].append(float(r[metric]))
    means = np.array([np.mean(by_layer[L]) for L in LAYERS])
    stds = np.array([np.std(by_layer[L]) for L in LAYERS])
    return means, stds


# ----- single-subplot helper -----

def _plot_with_band(ax, rows, metric: str, ylabel: str, title: str) -> None:
    """Plot pretrained vs random-init for one metric, with std bands."""
    for model, color, label in [
        (MODEL_PRE, COLOR_PRE, "pretrained BERT"),
        (MODEL_RAND, COLOR_RAND, "random-init BERT (same arch.)"),
    ]:
        m, s = per_layer_mean_std(rows, model, metric)
        ax.plot(LAYERS, m, marker="o", color=color, lw=2, label=label)
        ax.fill_between(LAYERS, m - s, m + s, alpha=BAND_ALPHA, color=color)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(LAYERS)
    ax.grid(alpha=0.3)


# ----- the three figures -----

def plot_alignment(rows: List[Dict[str, str]], outpath: Path) -> None:
    """Money-figure-in-progress: NMI + purity vs layer."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    _plot_with_band(axes[0], rows, "nmi", "NMI", "NMI vs 20 NG topic labels")
    _plot_with_band(axes[1], rows, "purity", "Purity", "Purity vs 20 NG topic labels")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle(
        "Topic alignment across BERT layers  "
        "(K=20, mode=l2, mean +/- std over 5 KMeans seeds)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath}")


def plot_diagnostics(rows: List[Dict[str, str]], outpath: Path) -> None:
    """Anisotropy + participation ratio vs layer (K-independent)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    _plot_with_band(
        axes[0], rows, "anisotropy",
        "Anisotropy (mean pairwise cos)",
        "Anisotropy of raw embedding space",
    )
    _plot_with_band(
        axes[1], rows, "participation_ratio",
        "Participation ratio",
        "Effective dimensionality (PR)",
    )
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle(
        "Representation-space diagnostics across layers  "
        "(K-independent; computed on raw X)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath}")


def plot_geometry_controls(rows: List[Dict[str, str]], outpath: Path) -> None:
    """Silhouette + DB + CH as geometry-only controls."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    _plot_with_band(
        axes[0], rows, "silhouette",
        "Silhouette", "Silhouette score (higher = better)",
    )
    _plot_with_band(
        axes[1], rows, "davies_bouldin",
        "Davies-Bouldin", "Davies-Bouldin (lower = better)",
    )
    _plot_with_band(
        axes[2], rows, "calinski_harabasz",
        "Calinski-Harabasz", "Calinski-Harabasz (higher = better)",
    )
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle(
        "Geometry-only controls: compactness does not explain topic alignment  "
        "(K=20, mode=l2, mean +/- std over 5 KMeans seeds)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath}")


# ----- main -----

def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv)
    print(f"loaded {len(rows)} rows from {args.csv}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_alignment(rows, outdir / "pilot_alignment.png")
    plot_diagnostics(rows, outdir / "pilot_diagnostics.png")
    plot_geometry_controls(rows, outdir / "pilot_geometry_controls.png")
    print("done.")


if __name__ == "__main__":
    main()
