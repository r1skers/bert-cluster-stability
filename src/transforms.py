"""Representation transforms for clustering sweeps.

These functions are deliberately independent from KMeans. They take
one layer matrix X with shape (N, d) and return a transformed matrix
that can be passed to clustering code.

The goal is to isolate representation choices:
    raw / l2 / centered
    remove top principal components
    PCA projection
    PCA whitening
"""
from typing import Callable, Dict

import numpy as np

from src.metrics import center, l2_normalize


def _centered_svd(X: np.ndarray):
    """Return centered X and its compact SVD."""
    Xc = center(X)
    u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc, u, s, vt


def centered_l2(X: np.ndarray) -> np.ndarray:
    """Center columns, then row-wise L2 normalize."""
    return l2_normalize(center(X))


def remove_top_pcs(X: np.ndarray, n_remove: int, *, normalize: bool = True) -> np.ndarray:
    """Remove the first `n_remove` principal directions from X.

    This is the "all-but-the-top" style post-processing often used
    for sentence embeddings: subtract the dominant common directions,
    then optionally L2-normalize before clustering.
    """
    if n_remove < 0:
        raise ValueError("n_remove must be non-negative")
    Xc, _, _, vt = _centered_svd(X)
    if n_remove == 0:
        out = Xc
    else:
        pcs = vt[:n_remove]
        out = Xc - (Xc @ pcs.T) @ pcs
    return l2_normalize(out) if normalize else out


def pca_project(X: np.ndarray, n_components: int, *, normalize: bool = True) -> np.ndarray:
    """Project centered X onto its first `n_components` PCs."""
    Xc, _, _, vt = _centered_svd(X)
    if n_components <= 0 or n_components > vt.shape[0]:
        raise ValueError(f"invalid n_components: {n_components}")
    out = Xc @ vt[:n_components].T
    return l2_normalize(out) if normalize else out


def pca_whiten(
    X: np.ndarray,
    n_components: int,
    *,
    normalize: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """Project to PCA coordinates and divide each PC by its singular value.

    Whitening makes retained principal directions have comparable
    scale, which can stop KMeans from being dominated by the largest
    variance axes.
    """
    Xc, _, s, vt = _centered_svd(X)
    if n_components <= 0 or n_components > vt.shape[0]:
        raise ValueError(f"invalid n_components: {n_components}")
    scores = Xc @ vt[:n_components].T
    out = scores / np.maximum(s[:n_components], eps)
    return l2_normalize(out) if normalize else out


def transform_registry() -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
    """Named transforms used by sweep scripts."""
    return {
        "raw": lambda X: X,
        "l2": l2_normalize,
        "centered": center,
        "centered_l2": centered_l2,
        "drop_pc1_l2": lambda X: remove_top_pcs(X, 1, normalize=True),
        "drop_pc3_l2": lambda X: remove_top_pcs(X, 3, normalize=True),
        "drop_pc5_l2": lambda X: remove_top_pcs(X, 5, normalize=True),
        "drop_pc10_l2": lambda X: remove_top_pcs(X, 10, normalize=True),
        "pca50_l2": lambda X: pca_project(X, 50, normalize=True),
        "pca100_l2": lambda X: pca_project(X, 100, normalize=True),
        "whiten50_l2": lambda X: pca_whiten(X, 50, normalize=True),
        "whiten100_l2": lambda X: pca_whiten(X, 100, normalize=True),
    }


def apply_transform(X: np.ndarray, name: str) -> np.ndarray:
    """Apply one named transform."""
    registry = transform_registry()
    if name not in registry:
        known = ", ".join(sorted(registry))
        raise ValueError(f"unknown transform {name!r}; known: {known}")
    return registry[name](X)
