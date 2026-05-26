r"""Sweep representation transforms before KMeans.

This script answers a practical question:
    Can simple post-processing make the KMeans clusters more
    topic-aligned than the current l2 baseline?

Default run is intentionally small:
    pretrained BERT, layer 12, K=20, seeds 0..4, all transforms

Run from repo root:
    .\.venv\Scripts\python.exe experiments\sweep_transforms.py
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
from src.transforms import apply_transform, transform_registry


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n_docs", type=int, default=2000)
    p.add_argument("--layers", type=int, nargs="+", default=[12])
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument(
        "--clusterers",
        nargs="+",
        default=["lloyd"],
        choices=[
            "lloyd",
            "spherical",
            "agglo_cosine",
            "agglo_ward",
            "gmm_diag",
            "gmm_full",
        ],
        help="Clustering backends to compare.",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=["pretrained"],
        choices=["pretrained", "random"],
        help="Model variants to sweep.",
    )
    p.add_argument(
        "--transforms",
        nargs="+",
        default=list(transform_registry().keys()),
        help="Named transforms from src.transforms.",
    )
    p.add_argument("--sample_seed", type=int, default=42)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--min_chars", type=int, default=30)
    p.add_argument("--model_name", type=str, default="bert-base-uncased")
    p.add_argument("--random_init_seed", type=int, default=1)
    p.add_argument("--output", type=str, default="outputs/tables/transform_sweep.csv")
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
    print(f"  loading {path}")
    with np.load(path) as d:
        return d["embeddings"], d["labels"]


def compute_one(
    X: np.ndarray,
    labels: np.ndarray,
    k: int,
    seed: int,
    clusterer: str,
):
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

    print("[1/3] loading caches...")
    loaded = {}
    reference_labels = None
    for name in args.models:
        tag = model_tag(name, args.random_init_seed)
        emb, labels = load_cached(tag, args)
        if reference_labels is None:
            reference_labels = labels
        else:
            assert np.array_equal(reference_labels, labels), "label mismatch"
        loaded[tag] = emb
    labels = reference_labels

    print("[2/3] sweeping transforms...")
    rows = []
    for tag, embeddings in loaded.items():
        for layer in args.layers:
            X_raw = embeddings[:, layer, :]
            for transform_name in args.transforms:
                t0 = time.time()
                X = apply_transform(X_raw, transform_name)
                transform_time = time.time() - t0
                ani = anisotropy(X, sample_size=2000, seed=0)
                pr = participation_ratio(X)

                for clusterer in args.clusterers:
                    for seed in args.seeds:
                        t1 = time.time()
                        scores = compute_one(X, labels, args.k, seed, clusterer)
                        elapsed = time.time() - t1
                        row = {
                            "model": tag,
                            "layer": layer,
                            "k": args.k,
                            "seed": seed,
                            "clusterer": clusterer,
                            "transform": transform_name,
                            "n_features": X.shape[1],
                            "transform_time_s": transform_time,
                            **scores,
                            "anisotropy": ani,
                            "participation_ratio": pr,
                        }
                        rows.append(row)
                        print(
                            f"  [{tag:>12s} L={layer:2d} {transform_name:>14s} "
                            f"{clusterer:>9s} s={seed}] "
                            f"nmi={scores['nmi']:.3f} "
                            f"pur={scores['purity']:.3f} "
                            f"sil={scores['silhouette']:+.3f} "
                            f"pr={pr:6.1f} ({elapsed:.1f}s)"
                        )

    print(f"[3/3] writing {args.output}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("done.")


if __name__ == "__main__":
    main()
