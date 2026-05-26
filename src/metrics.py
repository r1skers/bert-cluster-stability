"""Clustering metrics and the KMeans wrapper that produces them.

Three concerns live here:

    1. Representation preprocessing.
       `prepare(X, mode)` dispatches to `l2_normalize` (project to
       unit sphere) or `center` (subtract mean) or identity. The
       primary protocol uses mode='l2' because BERT representations
       are anisotropic (Ethayarajh 2019); raw KMeans on anisotropic
       vectors collapses toward the dominant axis. L2 makes the
       k-means effectively cosine-distance-based.

    2. KMeans with fixed, reproducible parameters.
       `kmeans_labels(X, k, seed)` hard-codes n_init=20 and
       algorithm='lloyd'. We do NOT rely on sklearn defaults
       because those change across versions.

    3. Five clustering metrics, all computed from a single KMeans
       fit per (X, k):
         - clarity:    silhouette, Davies-Bouldin, Calinski-Harabasz
         - alignment:  NMI, purity   (vs ground-truth labels)
       Returned as a dict for easy CSV serialization downstream.
"""
from typing import Dict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


# ----- representation preprocessing -----

def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalize. Output rows have unit L2 norm.

    `eps` guards against zero-norm rows (would be a degenerate
    BERT embedding; shouldn't occur in practice).
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def center(X: np.ndarray) -> np.ndarray:
    """Subtract the column mean — removes the dominant-direction
    anisotropy component without rescaling row magnitudes.
    """
    return X - X.mean(axis=0, keepdims=True)


def prepare(X: np.ndarray, mode: str = "l2") -> np.ndarray:
    """Dispatch table for representation preprocessing.

    Modes
    -----
    'l2'       : row-wise L2 normalize (primary protocol)
    'centered' : subtract column mean
    'raw'      : identity (use as ablation only)
    """
    if mode == "l2":
        return l2_normalize(X)
    if mode == "centered":
        return center(X)
    if mode == "raw":
        return X
    raise ValueError(f"unknown prepare mode: {mode!r}")


# ----- KMeans wrapper -----

def kmeans_labels(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Fit Lloyd KMeans with fixed reproducible parameters.

    Returns
    -------
    np.ndarray of shape (N,)
        Cluster assignment for each row in X.

    Notes
    -----
    `n_init=20` runs KMeans 20 times with different centroid
    initializations and keeps the best (lowest inertia). This is
    crucial — KMeans is non-convex and `n_init=1` can land in
    bad local minima, especially for high-K runs on BERT-scale
    data. Sklearn's recent default `n_init='auto'` is 1 for many
    cases, which is why we pin it explicitly.
    """
    km = KMeans(
        n_clusters=k,
        n_init=20,
        algorithm="lloyd",
        random_state=seed,
    )
    return km.fit_predict(X)


# ----- alignment metrics not in sklearn -----

def purity(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Cluster purity: per-cluster majority-vote accuracy.

    For each cluster, count its most common ground-truth label;
    sum those counts and divide by N. Range [0, 1], higher better.

    Purity has a known flaw: trivially 1.0 if every point is its
    own cluster. We always pair it with NMI (which penalizes
    over-clustering) so the pair is informative.
    """
    n = len(true_labels)
    k = int(pred_labels.max()) + 1
    c = int(true_labels.max()) + 1
    contingency = np.zeros((k, c), dtype=np.int64)
    for t, p in zip(true_labels, pred_labels):
        contingency[p, t] += 1
    return float(contingency.max(axis=1).sum() / n)


def nmi(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Normalized mutual information, arithmetic average normalization.

    Thin wrapper over sklearn's `normalized_mutual_info_score`
    so callers have a one-line, dependency-free signature.
    """
    return float(
        normalized_mutual_info_score(
            true_labels, pred_labels, average_method="arithmetic"
        )
    )


# ----- combined "fit once, compute many" entry point -----

def cluster_metrics(
    X: np.ndarray,
    true_labels: np.ndarray,
    k: int,
    seed: int = 0,
    mode: str = "l2",
) -> Dict[str, float]:
    """Compute clarity + alignment metrics for one (X, k) combo.

    Pipeline
    --------
    1. `prepare(X, mode)`               — representation preprocessing
    2. `kmeans_labels(X', k, seed)`     — one KMeans fit, n_init=20
    3. silhouette / DB / CH on (X', labels)
    4. NMI / purity on (true_labels, labels)

    Returns
    -------
    dict with keys:
        silhouette, davies_bouldin, calinski_harabasz, nmi, purity

    All values are Python floats so the dict serializes cleanly
    to CSV downstream.
    """
    X_prep = prepare(X, mode=mode)
    pred = kmeans_labels(X_prep, k=k, seed=seed)
    return {
        "silhouette": float(silhouette_score(X_prep, pred)),
        "davies_bouldin": float(davies_bouldin_score(X_prep, pred)),
        "calinski_harabasz": float(calinski_harabasz_score(X_prep, pred)),
        "nmi": nmi(true_labels, pred),
        "purity": purity(true_labels, pred),
    }
