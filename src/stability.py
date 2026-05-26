"""Resampling stability for clustering recipes.

The protocol is Lange/Ben-Hur style:

    1. Draw many 80% subsets.
    2. Fit the recipe on each subset.
    3. Predict labels for the full dataset.
    4. Compare all resulting partitions with pairwise ARI.

This measures partition reproducibility under sampling perturbation.
It is intentionally separate from topic alignment metrics such as NMI:
alignment asks whether clusters match labels, while stability asks
whether repeated clustering runs recover the same partition.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import re
from typing import Literal

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from src.clusterers import spherical_kmeans_fit, spherical_kmeans_predict
from src.metrics import l2_normalize


PredictiveClusterer = Literal["lloyd", "spherical"]
TransformFitScope = Literal["subset", "full"]


@dataclass(frozen=True)
class StabilitySummary:
    """Summary statistics for one resampling stability run."""

    mean_ari: float
    std_ari: float
    min_ari: float
    max_ari: float
    n_pairs: int
    n_runs: int
    subset_fraction: float


def mean_pairwise_ari(label_runs: np.ndarray) -> StabilitySummary:
    """Compute pairwise adjusted Rand index across label vectors."""
    if label_runs.ndim != 2:
        raise ValueError("label_runs must have shape (n_runs, n_samples)")
    n_runs = label_runs.shape[0]
    if n_runs < 2:
        raise ValueError("at least two label runs are required")

    scores = [
        adjusted_rand_score(label_runs[i], label_runs[j])
        for i, j in combinations(range(n_runs), 2)
    ]
    arr = np.asarray(scores, dtype=np.float64)
    return StabilitySummary(
        mean_ari=float(arr.mean()),
        std_ari=float(arr.std(ddof=0)),
        min_ari=float(arr.min()),
        max_ari=float(arr.max()),
        n_pairs=int(arr.size),
        n_runs=int(n_runs),
        subset_fraction=float("nan"),
    )


def subsample_indices(
    n_samples: int,
    subset_fraction: float,
    n_runs: int,
    seed: int = 0,
) -> list[np.ndarray]:
    """Draw reproducible without-replacement subsets."""
    if not 0.0 < subset_fraction <= 1.0:
        raise ValueError("subset_fraction must be in (0, 1]")
    if n_runs < 2:
        raise ValueError("n_runs must be at least 2")
    subset_size = int(round(n_samples * subset_fraction))
    if subset_size < 2:
        raise ValueError("subset is too small")

    rng = np.random.default_rng(seed)
    return [
        np.sort(rng.choice(n_samples, size=subset_size, replace=False))
        for _ in range(n_runs)
    ]


def transform_for_resample(
    X_all: np.ndarray,
    train_idx: np.ndarray,
    transform: str,
    *,
    fit_scope: TransformFitScope = "subset",
    eps: float = 1e-12,
) -> np.ndarray:
    """Apply a recipe transform, fitting data-dependent pieces as needed.

    For stability, `fit_scope="subset"` is the stricter default:
    each resample refits PCA whitening from only the sampled rows, then
    projects all rows through that sampled coordinate system.
    """
    if transform == "l2":
        return l2_normalize(X_all)

    match = re.fullmatch(r"whiten(\d+)_l2", transform)
    if match is None:
        raise ValueError(
            "stability currently supports transforms 'l2' and "
            "'whiten{dim}_l2'"
        )

    n_components = int(match.group(1))
    X_fit = X_all if fit_scope == "full" else X_all[train_idx]
    mean = X_fit.mean(axis=0, keepdims=True)
    X_fit_centered = X_fit - mean
    _, s, vt = np.linalg.svd(X_fit_centered, full_matrices=False)
    if n_components <= 0 or n_components > vt.shape[0]:
        raise ValueError(f"invalid n_components: {n_components}")

    scores = (X_all - mean) @ vt[:n_components].T
    Xw = scores / np.maximum(s[:n_components], eps)
    return l2_normalize(Xw)


def fit_predict_all(
    X_all: np.ndarray,
    train_idx: np.ndarray,
    *,
    k: int,
    seed: int,
    clusterer: PredictiveClusterer,
) -> np.ndarray:
    """Fit a predictive clusterer on train_idx and label every row."""
    X_train = X_all[train_idx]
    if clusterer == "lloyd":
        model = KMeans(
            n_clusters=k,
            n_init=20,
            algorithm="lloyd",
            random_state=seed,
        )
        model.fit(X_train)
        return model.predict(X_all).astype(np.int64)

    if clusterer == "spherical":
        centers, _ = spherical_kmeans_fit(X_train, k=k, seed=seed)
        return spherical_kmeans_predict(X_all, centers)

    raise ValueError(f"unsupported predictive clusterer: {clusterer!r}")


def resampling_stability(
    X: np.ndarray,
    *,
    k: int,
    transform: str,
    clusterer: PredictiveClusterer,
    subset_fraction: float = 0.8,
    n_runs: int = 50,
    subset_seed: int = 0,
    cluster_seed: int = 0,
    transform_fit_scope: TransformFitScope = "subset",
) -> tuple[StabilitySummary, np.ndarray]:
    """Run resampling stability and return summary plus label matrix."""
    subsets = subsample_indices(
        len(X),
        subset_fraction=subset_fraction,
        n_runs=n_runs,
        seed=subset_seed,
    )
    label_runs = []
    for run, train_idx in enumerate(subsets):
        X_run = transform_for_resample(
            X,
            train_idx,
            transform,
            fit_scope=transform_fit_scope,
        )
        labels = fit_predict_all(
            X_run,
            train_idx,
            k=k,
            seed=cluster_seed + run,
            clusterer=clusterer,
        )
        label_runs.append(labels)

    label_matrix = np.vstack(label_runs)
    summary = mean_pairwise_ari(label_matrix)
    summary = StabilitySummary(
        mean_ari=summary.mean_ari,
        std_ari=summary.std_ari,
        min_ari=summary.min_ari,
        max_ari=summary.max_ari,
        n_pairs=summary.n_pairs,
        n_runs=summary.n_runs,
        subset_fraction=subset_fraction,
    )
    return summary, label_matrix
