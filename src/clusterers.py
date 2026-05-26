"""Alternative clustering backends with a common labels interface."""
from typing import Literal

import numpy as np
from sklearn.cluster import AgglomerativeClustering, kmeans_plusplus
from sklearn.mixture import GaussianMixture

from src.metrics import kmeans_labels, l2_normalize


ClustererName = Literal[
    "lloyd",
    "spherical",
    "agglo_cosine",
    "agglo_ward",
    "gmm_diag",
    "gmm_full",
]


def spherical_kmeans_labels(
    X: np.ndarray,
    k: int,
    seed: int = 0,
    n_init: int = 20,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """Cluster rows by spherical KMeans.

    Spherical KMeans uses cosine similarity: rows are L2-normalized,
    centroids are initialized on the unit sphere, and every centroid
    update is re-normalized. The objective maximizes total cosine
    similarity to assigned centroids.
    """
    centers, labels = spherical_kmeans_fit(
        X,
        k=k,
        seed=seed,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
    )
    return labels


def spherical_kmeans_fit(
    X: np.ndarray,
    k: int,
    seed: int = 0,
    n_init: int = 20,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit spherical KMeans and return unit centroids plus train labels.

    This split fit/predict interface is useful for resampling stability:
    fit centroids on one subset, then assign labels to the full dataset.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if len(X) < k:
        raise ValueError("k cannot exceed number of rows")
    if n_init <= 0:
        raise ValueError("n_init must be positive")

    Xn = l2_normalize(X).astype(np.float64, copy=False)
    n = Xn.shape[0]
    best_centers = None
    best_labels = None
    best_score = -np.inf

    rng = np.random.default_rng(seed)
    init_seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_init)

    for init_seed in init_seeds:
        centers, _ = kmeans_plusplus(Xn, n_clusters=k, random_state=int(init_seed))
        centers = l2_normalize(centers)
        old_labels = None

        for _ in range(max_iter):
            sim = Xn @ centers.T
            labels = np.argmax(sim, axis=1)

            new_centers = np.zeros_like(centers)
            for c in range(k):
                mask = labels == c
                if mask.any():
                    new_centers[c] = Xn[mask].mean(axis=0)
                else:
                    # Re-seed empty clusters with the point currently
                    # least similar to any centroid.
                    worst = int(np.argmin(sim.max(axis=1)))
                    new_centers[c] = Xn[worst]
            new_centers = l2_normalize(new_centers)

            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if old_labels is not None and np.array_equal(labels, old_labels):
                break
            if shift < tol:
                break
            old_labels = labels

        sim = Xn @ centers.T
        labels = np.argmax(sim, axis=1)
        score = float(sim[np.arange(n), labels].sum())
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
            best_labels = labels.copy()

    return best_centers, best_labels.astype(np.int64)


def spherical_kmeans_predict(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Assign rows to the nearest spherical KMeans centroid by cosine."""
    if centers.ndim != 2:
        raise ValueError("centers must have shape (k, d)")
    Xn = l2_normalize(X).astype(np.float64, copy=False)
    Cn = l2_normalize(centers).astype(np.float64, copy=False)
    return np.argmax(Xn @ Cn.T, axis=1).astype(np.int64)


def agglomerative_cosine_average_labels(X: np.ndarray, k: int) -> np.ndarray:
    """Cluster rows by agglomerative clustering with cosine/average linkage.

    This is a non-centroid control for KMeans-style probes. It builds a
    hierarchy from pairwise cosine distances and cuts it to exactly k
    clusters, without random initialization.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if len(X) < k:
        raise ValueError("k cannot exceed number of rows")

    model = AgglomerativeClustering(
        n_clusters=k,
        metric="cosine",
        linkage="average",
    )
    return model.fit_predict(l2_normalize(X)).astype(np.int64)


def agglomerative_ward_labels(X: np.ndarray, k: int) -> np.ndarray:
    """Cluster rows by agglomerative clustering with Ward linkage.

    Ward linkage is the standard Euclidean hierarchical clustering
    control. It greedily merges clusters to minimize the increase in
    within-cluster variance, making it closer in spirit to KMeans than
    cosine/average linkage.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if len(X) < k:
        raise ValueError("k cannot exceed number of rows")

    model = AgglomerativeClustering(
        n_clusters=k,
        metric="euclidean",
        linkage="ward",
    )
    return model.fit_predict(X).astype(np.int64)


def gaussian_mixture_labels(
    X: np.ndarray,
    k: int,
    seed: int = 0,
    covariance_type: Literal["diag", "full"] = "diag",
    n_init: int = 5,
) -> np.ndarray:
    """Cluster rows with a Gaussian mixture model.

    GMM is a soft, probabilistic control for KMeans. `diag` allows each
    component to have axis-wise variance; `full` allows a full covariance
    matrix per component. We return hard labels by maximum posterior
    component probability.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if len(X) < k:
        raise ValueError("k cannot exceed number of rows")

    model = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        n_init=n_init,
        random_state=seed,
        reg_covar=1e-6,
        max_iter=200,
    )
    return model.fit_predict(X).astype(np.int64)


def cluster_labels(
    X: np.ndarray,
    k: int,
    seed: int = 0,
    clusterer: ClustererName = "lloyd",
) -> np.ndarray:
    """Return cluster labels using one named backend."""
    if clusterer == "lloyd":
        return kmeans_labels(X, k=k, seed=seed)
    if clusterer == "spherical":
        return spherical_kmeans_labels(X, k=k, seed=seed)
    if clusterer == "agglo_cosine":
        return agglomerative_cosine_average_labels(X, k=k)
    if clusterer == "agglo_ward":
        return agglomerative_ward_labels(X, k=k)
    if clusterer == "gmm_diag":
        return gaussian_mixture_labels(X, k=k, seed=seed, covariance_type="diag")
    if clusterer == "gmm_full":
        return gaussian_mixture_labels(X, k=k, seed=seed, covariance_type="full")
    raise ValueError(f"unknown clusterer: {clusterer!r}")
