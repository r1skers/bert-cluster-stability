r"""Artifact 5 umbrella: three-view synthesis figure.

Overlays the layer-wise topic signal as seen by the three probes that
share the SAME cached BERT embeddings:

    5.1 clustering   -> NMI(L)            (unsupervised structure)
    5.2 logreg       -> CV accuracy(L)    (supervised convex optimum)
    5.3 Fisher LDA   -> CV accuracy(L)    (supervised analytical optimum)

This is the cross-probe deliverable: it answers two questions at once.

Panel A (pretrained, normalized): do the three views AGREE on *where*
in depth topic information emerges? Each metric is min-max normalized
over its own 13 layers, so only the shape is compared (magnitudes of
NMI and accuracy are not on the same scale).

Panel B (random-init, native units): the cross-view DIVERGENCE. A
supervised linear probe reads topic information well above chance from
random-init BERT and that readability decays with depth, whereas
unsupervised clustering sees essentially nothing (NMI near the floor).
Same topic signal, different probe verdicts -> the triangulation point.

Inputs:
    outputs/tables/clustering/layer_sweep_best_recipe.csv   (5.1, multi-seed)
    outputs/tables/probe/linear_probe.csv                   (5.2)
    outputs/tables/probe/lda_probe.csv                      (5.3)

Output:
    outputs/figures/synthesis/three_view_synthesis.png

Run from repo root:
    .\.venv\Scripts\python.exe experiments\synthesis\plot_three_view.py
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

MODEL_PRE = "pretrained"
MODEL_RAND = "random_seed1"
LAYERS = list(range(13))

# one color per view, shared across both panels
VIEWS = [
    ("5.1 clustering (NMI)", "tab:green", "o"),
    ("5.2 logreg (accuracy)", "tab:blue", "s"),
    ("5.3 Fisher LDA (accuracy)", "tab:orange", "^"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--clustering_csv",
        default="outputs/tables/clustering/layer_sweep_best_recipe.csv",
    )
    p.add_argument("--logreg_csv", default="outputs/tables/probe/linear_probe.csv")
    p.add_argument("--lda_csv", default="outputs/tables/probe/lda_probe.csv")
    p.add_argument("--outdir", default="outputs/figures/synthesis")
    p.add_argument("--filename", default="three_view_synthesis.png")
    return p.parse_args()


def _rows(path: str) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def layer_means(path: str, value_col: str, model: str) -> np.ndarray:
    """Mean of `value_col` per layer for one model, indexed by layer 0..12.

    Averages over any extra rows (e.g. KMeans seeds in the clustering
    sweep); for the probe CSVs there is exactly one row per (model, layer)
    so the mean is a no-op.
    """
    by_layer = defaultdict(list)
    for r in _rows(path):
        if r["model"] != model:
            continue
        by_layer[int(r["layer"])].append(float(r[value_col]))
    return np.array([np.mean(by_layer[L]) for L in LAYERS])


def minmax(v: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]; flat input maps to all-zeros."""
    span = v.max() - v.min()
    if span == 0:
        return np.zeros_like(v)
    return (v - v.min()) / span


def collect(args: argparse.Namespace, model: str) -> List[Tuple[str, str, str, np.ndarray]]:
    """Return [(label, color, marker, values[L])] for the three views."""
    sources = [
        (args.clustering_csv, "nmi"),
        (args.logreg_csv, "accuracy_mean"),
        (args.lda_csv, "accuracy_mean"),
    ]
    out = []
    for (label, color, marker), (path, col) in zip(VIEWS, sources):
        out.append((label, color, marker, layer_means(path, col, model)))
    return out


def main() -> None:
    args = parse_args()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

    # ----- Panel A: pretrained, normalized shape -----
    for label, color, marker, v in collect(args, MODEL_PRE):
        axA.plot(LAYERS, minmax(v), marker=marker, color=color, lw=2, label=label)
    axA.set_title("A. Pretrained BERT — normalized layer profile\n(min-max per metric; shape only)")
    axA.set_ylabel("normalized signal (0 = layer min, 1 = layer max)")
    axA.set_ylim(-0.05, 1.05)
    axA.legend(loc="lower right", fontsize=9)

    # ----- Panel B: random-init, native units -----
    chance = float(_rows(args.logreg_csv)[0]["chance"])
    for label, color, marker, v in collect(args, MODEL_RAND):
        axB.plot(LAYERS, v, marker=marker, color=color, lw=2, label=label)
    axB.axhline(chance, ls=":", lw=1, color="black", alpha=0.6,
                label=f"chance ({chance:.2f})")
    axB.set_title("B. Random-init BERT — native units\n(supervised probes read it; clustering does not)")
    axB.set_ylabel("score (NMI / accuracy)")
    axB.set_ylim(0, 0.5)
    axB.legend(loc="upper right", fontsize=9)

    for ax in (axA, axB):
        ax.set_xlabel("Layer")
        ax.set_xticks(LAYERS)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Artifact 5 — three probes on the same BERT layers: "
        "unsupervised clustering vs supervised linear separability",
        fontsize=12,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / args.filename
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath}")
    print("done.")


if __name__ == "__main__":
    main()
