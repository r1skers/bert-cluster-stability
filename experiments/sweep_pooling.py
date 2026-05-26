r"""Sweep cached layer-pooling variants.

This is isolated from token-level pooling: it only recombines cached
layer vectors, so it does not rerun BERT.

Default comparison:
    layer12 vs last4_mean vs last4_concat
    transform = whiten100_l2
    clusterer = spherical

Run from repo root:
    .\.venv\Scripts\python.exe experiments\sweep_pooling.py
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cache import embedding_cache_path
from src.clusterers import cluster_labels
from src.diagnostics import anisotropy, participation_ratio
from src.metrics import nmi, purity
from src.pooling import apply_pooling, pooling_registry
from src.transforms import apply_transform


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n_docs", type=int, default=2000)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument(
        "--poolings",
        nargs="+",
        default=["layer12", "last4_mean", "last4_concat"],
        choices=list(pooling_registry().keys()),
    )
    p.add_argument(
        "--transforms",
        nargs="+",
        default=["whiten100_l2"],
        help="Transforms to apply after pooling.",
    )
    p.add_argument(
        "--clusterers",
        nargs="+",
        default=["spherical"],
        choices=[
            "lloyd",
            "spherical",
            "agglo_cosine",
            "agglo_ward",
            "gmm_diag",
            "gmm_full",
        ],
    )
    p.add_argument("--sample_seed", type=int, default=42)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--min_chars", type=int, default=30)
    p.add_argument("--model_name", type=str, default="bert-base-uncased")
    p.add_argument("--output", type=str, default="outputs/tables/pooling_sweep.csv")
    return p.parse_args()


def load_cached(args: argparse.Namespace):
    path = embedding_cache_path(
        "pretrained",
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
    }


def main() -> None:
    args = parse_args()
    embeddings, labels = load_cached(args)

    rows = []
    for pooling_name in args.poolings:
        t_pool = time.time()
        X_pool = apply_pooling(embeddings, pooling_name)
        pool_time = time.time() - t_pool

        for transform_name in args.transforms:
            t_transform = time.time()
            X = apply_transform(X_pool, transform_name)
            transform_time = time.time() - t_transform
            ani = anisotropy(X, sample_size=2000, seed=0)
            pr = participation_ratio(X)

            for clusterer in args.clusterers:
                for seed in args.seeds:
                    t0 = time.time()
                    scores = compute_scores(X, labels, args.k, seed, clusterer)
                    elapsed = time.time() - t0
                    row = {
                        "pooling": pooling_name,
                        "transform": transform_name,
                        "clusterer": clusterer,
                        "k": args.k,
                        "seed": seed,
                        "n_features": X.shape[1],
                        "pool_time_s": pool_time,
                        "transform_time_s": transform_time,
                        **scores,
                        "anisotropy": ani,
                        "participation_ratio": pr,
                    }
                    rows.append(row)
                    print(
                        f"  [{pooling_name:>12s} {transform_name:>14s} "
                        f"{clusterer:>9s} s={seed}] "
                        f"nmi={scores['nmi']:.3f} "
                        f"pur={scores['purity']:.3f} "
                        f"sil={scores['silhouette']:+.3f} "
                        f"features={X.shape[1]} ({elapsed:.1f}s)"
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
