r"""Plot a supervised linear-probe accuracy(L) curve.

Shared by Artifact 5.2 (logistic regression) and 5.3 (Fisher LDA):
the figure title is derived from the CSV's `probe` column, so the same
plotter serves both views.

Input:
    outputs/tables/probe/<probe>_probe.csv   (from run_linear_probe.py)

Output (300 dpi PNG):
    outputs/figures/probe/<filename>
        Probe accuracy(L), pretrained vs random-init BERT, with the
        CV std as a shaded band, plus chance (1/20) and majority
        baselines as horizontal references.

Run from repo root:
    .\.venv\Scripts\python.exe experiments\probe\plot_linear_probe.py
    .\.venv\Scripts\python.exe experiments\probe\plot_linear_probe.py ^
        --csv outputs/tables/probe/lda_probe.csv --filename lda_probe_accuracy.png
"""
import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

# shared style with plot_pilot.py
COLOR_PRE = "tab:blue"
COLOR_RAND = "tab:gray"
BAND_ALPHA = 0.18
MODEL_PRE = "pretrained"
MODEL_RAND = "random_seed1"

# human-readable probe names for the figure title (5.2 / 5.3 share this plot)
PROBE_LABEL = {
    "logreg": "multinomial logistic regression",
    "lda": "Fisher LDA (shrinkage)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="outputs/tables/probe/linear_probe.csv")
    p.add_argument("--outdir", default="outputs/figures/probe")
    p.add_argument("--filename", default="linear_probe_accuracy.png")
    return p.parse_args()


def load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def series(
    rows: List[Dict[str, str]], model: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (layers, accuracy_mean, accuracy_std) for one model, sorted."""
    sub = sorted(
        (r for r in rows if r["model"] == model), key=lambda r: int(r["layer"])
    )
    layers = np.array([int(r["layer"]) for r in sub])
    mean = np.array([float(r["accuracy_mean"]) for r in sub])
    std = np.array([float(r["accuracy_std"]) for r in sub])
    return layers, mean, std


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv)
    print(f"loaded {len(rows)} rows from {args.csv}")

    chance = float(rows[0]["chance"])
    majority = float(rows[0]["majority"])
    probe = rows[0].get("probe", "logreg")
    probe_name = PROBE_LABEL.get(probe, probe)

    fig, ax = plt.subplots(figsize=(8, 5))
    for model, color, label in [
        (MODEL_PRE, COLOR_PRE, "pretrained BERT"),
        (MODEL_RAND, COLOR_RAND, "random-init BERT (same arch.)"),
    ]:
        layers, mean, std = series(rows, model)
        ax.plot(layers, mean, marker="o", color=color, lw=2, label=label)
        ax.fill_between(layers, mean - std, mean + std, alpha=BAND_ALPHA, color=color)

    ax.axhline(majority, ls="--", lw=1, color="black", alpha=0.6,
               label=f"majority ({majority:.2f})")
    ax.axhline(chance, ls=":", lw=1, color="black", alpha=0.6,
               label=f"chance ({chance:.2f})")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Linear-probe accuracy (5-fold CV)")
    ax.set_title(
        "Topic linear-separability across BERT layers\n"
        f"({probe_name}, mean +/- std over 5 folds)"
    )
    ax.set_xticks(range(13))
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)

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
