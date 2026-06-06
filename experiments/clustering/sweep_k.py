r"""Sweep K under fixed representation and clustering choices.

Default question:
    If we keep the current best recipe fixed
        layer12 + whiten100_l2 + spherical KMeans
    does K=20 really look best, or does the representation prefer a
    coarser/finer semantic level?

Run from repo root:
    .\.venv\Scripts\python.exe experiments\clustering\sweep_k.py
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cache import embedding_cache_path
from src.clusterers import cluster_labels
from src.diagnostics import anisotropy, participation_ratio
from src.metrics import nmi, purity
from src.pooling import apply_pooling
from src.transforms import apply_transform


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n_docs", type=int, default=2000)
    p.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20, 50])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--pooling", default="layer12")
    p.add_argument("--transform", default="whiten100_l2")
    p.add_argument(
        "--clusterer",
        default="spherical",
        choices=[
            "lloyd",
            "spherical",
            "agglo_cosine",
            "agglo_ward",
            "gmm_diag",
            "gmm_full",
        ],
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=["pretrained"],
        choices=["pretrained", "random"],
        help="Model variants to compare under the same recipe.",
    )
    p.add_argument("--sample_seed", type=int, default=42)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--min_chars", type=int, default=30)
    p.add_argument("--model_name", type=str, default="bert-base-uncased")
    p.add_argument("--random_init_seed", type=int, default=1)
    p.add_argument("--output", type=str, default="outputs/tables/clustering/k_sweep.csv")
    return p.parse_args()


def model_tag(name: str, random_init_seed: int) -> str:
    if name == "pretrained":
        return "pretrained"
    if name == "random":
        return f"random_seed{random_init_seed}"
    raise ValueError(f"unknown model name: {name}")


def load_cached(tag: str, args: argparse.Namespace):
    path = embedding_cache_path(
        tag,
        args.model_name,
        args.n_docs,
        args.sample_seed,
        args.max_length,
        args.min_chars,
    )
    print(f"loading {path}")
    with np.load(path) as d:
        return d["embeddings"], d["labels"]


def compute_scores(X: np.ndarray, labels: np.ndarray, k: int, seed: int, clusterer: str):
    pred = cluster_labels(X, k=k, seed=seed, clusterer=clusterer)
    return {
        "silhouette": float(silhouette_score(X, pred)),
        "davies_bouldin": float(davies_bouldin_score(X, pred)),
        "calinski_harabasz": float(calinski_harabasz_score(X, pred)),
        "nmi": nmi(labels, pred),
        "purity": purity(labels, pred),
        "min_cluster_size": int(np.bincount(pred, minlength=k).min()),
        "max_cluster_size": int(np.bincount(pred, minlength=k).max()),
    }


def main() -> None:
    args = parse_args()

    rows = []
    reference_labels = None
    for model in args.models:
        tag = model_tag(model, args.random_init_seed)
        embeddings, labels = load_cached(tag, args)
        if reference_labels is None:
            reference_labels = labels
        else:
            assert np.array_equal(reference_labels, labels), "label mismatch"

        X_pool = apply_pooling(embeddings, args.pooling)
        t_transform = time.time()
        X = apply_transform(X_pool, args.transform)
        transform_time = time.time() - t_transform
        ani = anisotropy(X, sample_size=2000, seed=0)
        pr = participation_ratio(X)

        for k in args.ks:
            for seed in args.seeds:
                t0 = time.time()
                scores = compute_scores(X, labels, k, seed, args.clusterer)
                elapsed = time.time() - t0
                row = {
                    "model": tag,
                    "pooling": args.pooling,
                    "transform": args.transform,
                    "clusterer": args.clusterer,
                    "k": k,
                    "seed": seed,
                    "n_features": X.shape[1],
                    "transform_time_s": transform_time,
                    **scores,
                    "anisotropy": ani,
                    "participation_ratio": pr,
                }
                rows.append(row)
                print(
                    f"  [{tag:>12s} K={k:2d} s={seed}] "
                    f"nmi={scores['nmi']:.3f} "
                    f"pur={scores['purity']:.3f} "
                    f"size={scores['min_cluster_size']}-{scores['max_cluster_size']} "
                    f"({elapsed:.1f}s)"
                )

    print(f"writing {args.output}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("done.")


if __name__ == "__main__":
    main()
