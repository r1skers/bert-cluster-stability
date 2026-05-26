r"""Sweep PCA whitening dimensionality under a fixed clustering recipe.

Question:
    Is whiten100_l2 special, or is there a broad good range?

Fixed recipe:
    pooling = layer12
    clusterer = spherical
    K = 20

Variable:
    PCA whitening dimensionality d

Run from repo root:
    .\.venv\Scripts\python.exe experiments\sweep_whitening_dims.py
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
from src.metrics import center, l2_normalize, nmi, purity
from src.pooling import apply_pooling


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n_docs", type=int, default=2000)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--dims", type=int, nargs="+", default=[10, 20, 50, 100, 150, 200, 300, 500, 768])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--pooling", default="layer12")
    p.add_argument("--clusterer", default="spherical", choices=["lloyd", "spherical"])
    p.add_argument(
        "--models",
        nargs="+",
        default=["pretrained", "random"],
        choices=["pretrained", "random"],
    )
    p.add_argument("--sample_seed", type=int, default=42)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--min_chars", type=int, default=30)
    p.add_argument("--model_name", type=str, default="bert-base-uncased")
    p.add_argument("--random_init_seed", type=int, default=1)
    p.add_argument("--output", type=str, default="outputs/tables/whitening_dim_sweep.csv")
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


def centered_svd(X: np.ndarray):
    Xc = center(X)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc, s, vt


def whiten_from_svd(
    Xc: np.ndarray,
    s: np.ndarray,
    vt: np.ndarray,
    dim: int,
    eps: float = 1e-12,
) -> np.ndarray:
    if dim <= 0 or dim > vt.shape[0]:
        raise ValueError(f"invalid whitening dim {dim}; max is {vt.shape[0]}")
    scores = Xc @ vt[:dim].T
    Xw = scores / np.maximum(s[:dim], eps)
    return l2_normalize(Xw)


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

        print(f"[{tag}] SVD for {args.pooling}...")
        t_svd = time.time()
        Xc, s, vt = centered_svd(X_pool)
        svd_time = time.time() - t_svd
        max_dim = vt.shape[0]
        dims = [d for d in args.dims if d <= max_dim]
        skipped = [d for d in args.dims if d > max_dim]
        if skipped:
            print(f"  skipping dims above max {max_dim}: {skipped}")

        for dim in dims:
            t_transform = time.time()
            X = whiten_from_svd(Xc, s, vt, dim)
            transform_time = time.time() - t_transform
            ani = anisotropy(X, sample_size=2000, seed=0)
            pr = participation_ratio(X)

            for seed in args.seeds:
                t0 = time.time()
                scores = compute_scores(X, labels, args.k, seed, args.clusterer)
                elapsed = time.time() - t0
                row = {
                    "model": tag,
                    "pooling": args.pooling,
                    "clusterer": args.clusterer,
                    "k": args.k,
                    "seed": seed,
                    "whiten_dim": dim,
                    "n_features": X.shape[1],
                    "svd_time_s": svd_time,
                    "transform_time_s": transform_time,
                    **scores,
                    "anisotropy": ani,
                    "participation_ratio": pr,
                }
                rows.append(row)
                print(
                    f"  [{tag:>12s} d={dim:3d} s={seed}] "
                    f"nmi={scores['nmi']:.3f} "
                    f"pur={scores['purity']:.3f} "
                    f"sil={scores['silhouette']:+.3f} "
                    f"pr={pr:6.1f} ({elapsed:.1f}s)"
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
