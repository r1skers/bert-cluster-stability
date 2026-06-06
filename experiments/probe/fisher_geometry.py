r"""Artifact 5.3: Fisher separability geometry across BERT layers.

Where 5.2's logistic probe and 5.3's LDA *classifier* report how
accurately a linear rule separates the topics, this script reports the
underlying GEOMETRY that makes such separation easy or hard:

    Fisher trace ratio   J(L) = tr(S_B) / tr(S_W)
    between-class frac    η²(L) = J / (J + 1) = tr(S_B) / tr(S_T)

J answers "are the class means far apart relative to within-class
spread?" — i.e. is the rise in linear decodability (5.2/5.3-LDA)
driven by classes becoming more compact and better separated?

Reuses the SAME cached embeddings as 5.1/5.2; no BERT forward here.
Features are standardized per layer (matching the LDA probe pipeline)
before the geometry is measured.

Outputs:
    outputs/tables/probe/fisher_geometry.csv
    outputs/figures/probe/fisher_geometry.png

Run from repo root:
    .\.venv\Scripts\python.exe experiments\probe\fisher_geometry.py
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from src.cache import embedding_cache_path
from src.diagnostics import fisher_trace_ratio

COLOR_PRE = "tab:blue"
COLOR_RAND = "tab:gray"
MODEL_PRE = "pretrained"
LAYERS = list(range(13))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n_docs", type=int, default=2000)
    p.add_argument("--sample_seed", type=int, default=42)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--min_chars", type=int, default=30)
    p.add_argument("--model_name", type=str, default="bert-base-uncased")
    p.add_argument("--random_init_seed", type=int, default=1)
    p.add_argument("--output", type=str, default="outputs/tables/probe/fisher_geometry.csv")
    p.add_argument("--outdir", type=str, default="outputs/figures/probe")
    p.add_argument("--filename", type=str, default="fisher_geometry.png")
    return p.parse_args()


def load_cached(tag: str, args: argparse.Namespace):
    path = embedding_cache_path(
        tag, args.model_name, args.n_docs,
        args.sample_seed, args.max_length, args.min_chars,
    )
    print(f"  loading {path}")
    with np.load(path) as d:
        return d["embeddings"], d["labels"]


def main() -> None:
    args = parse_args()

    print("[1/3] loading cached embeddings...")
    pretrained_emb, labels = load_cached("pretrained", args)
    random_emb, labels_check = load_cached(f"random_seed{args.random_init_seed}", args)
    assert np.array_equal(labels, labels_check), "labels mismatch between caches"
    print(f"  shape: {pretrained_emb.shape}")

    print("[2/3] computing Fisher geometry per layer...")
    rows = []
    series = {}
    for model_tag, embeds in [
        ("pretrained", pretrained_emb),
        (f"random_seed{args.random_init_seed}", random_emb),
    ]:
        eta2_curve = []
        for L in LAYERS:
            X = StandardScaler().fit_transform(embeds[:, L, :])
            J = fisher_trace_ratio(X, labels)
            eta2 = J / (J + 1.0)
            eta2_curve.append(eta2)
            rows.append({
                "model": model_tag, "layer": L,
                "fisher_ratio": J, "between_var_frac": eta2,
            })
            print(f"  [{model_tag:>12s} L={L:2d}] J={J:6.3f}  eta2={eta2:.3f}")
        series[model_tag] = eta2_curve

    print(f"[3/3] writing CSV + figure...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    rand_tag = f"random_seed{args.random_init_seed}"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(LAYERS, series[MODEL_PRE], marker="o", color=COLOR_PRE, lw=2,
            label="pretrained BERT")
    ax.plot(LAYERS, series[rand_tag], marker="o", color=COLOR_RAND, lw=2,
            label="random-init BERT (same arch.)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Between-class variance fraction  η² = tr(S_B)/tr(S_T)")
    ax.set_title(
        "Fisher separability geometry across BERT layers\n"
        "(supervised; higher = topics more compact & separated)"
    )
    ax.set_xticks(LAYERS)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / args.filename
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {args.output}")
    print(f"  wrote {outpath}")
    print("done.")


if __name__ == "__main__":
    main()
