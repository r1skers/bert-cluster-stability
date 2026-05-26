"""K-independent representation-shape diagnostics.

These functions characterize the geometry of a representation
matrix X of shape (N, d) WITHOUT any clustering — they describe
the embedding space itself.

Why they matter for this project:

    * `anisotropy(X)` quantifies the Ethayarajh 2019 finding that
      BERT representations live in a narrow cone in the embedding
      space. High anisotropy is a known warning sign that any
      clustering result is at risk of being dominated by the
      dominant direction.

    * `participation_ratio(X)` measures the effective dimensionality
      via the covariance eigenvalue spectrum. PR=d means perfectly
      isotropic; PR=1 means collapsed to a line.

Both are pure numpy and computed once per (model, layer) — not
per K — so they sit outside `src/metrics.py`.
"""
from typing import Optional

import numpy as np


def anisotropy(
    X: np.ndarray,
    sample_size: Optional[int] = 2000,
    seed: int = 0,
) -> float:
    """Mean pairwise cosine similarity between rows of X.

    Range: [-1, 1]. Higher = more anisotropic (vectors clustered
    along a common direction). For isotropic gaussian X, the
    population value is 0 (with finite-sample noise).

    Parameters
    ----------
    X : (N, d) array
    sample_size : int or None
        If `N > sample_size`, randomly sub-sample to keep the
        O(N²) cosine-pairs computation tractable. None disables
        sub-sampling. Default 2000 → at most 2e6 pairs, ~ms.
    seed : int
        RNG seed for the optional sub-sample.

    Notes
    -----
    We avoid materializing the full N×N cosine matrix; instead we
    L2-normalize once, then the dot-product `Xn @ Xn.T` gives the
    cosine matrix in one BLAS call. We then mean over the strict
    upper triangle (excluding the diagonal of 1s).
    """
    n = X.shape[0]
    if sample_size is not None and n > sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=sample_size, replace=False)
        X = X[idx]
        n = sample_size

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.maximum(norms, 1e-12)
    cos = Xn @ Xn.T                                # (n, n)
    iu = np.triu_indices(n, k=1)                   # strict upper tri
    return float(cos[iu].mean())


def participation_ratio(X: np.ndarray) -> float:
    """Effective dimensionality from the covariance eigenspectrum.

    Definition
    ----------
        PR = (Σ λ_i)² / Σ λ_i²
        where λ_i are eigenvalues of the (centered) covariance.

    Equivalently, PR = tr(C)² / tr(C²). We use the trace form so
    we never explicitly compute the eigendecomposition.

    Range: [1, d].
        PR ≈ 1   → all variance on one axis (collapsed to a line)
        PR ≈ d   → variance spread evenly over all axes (isotropic)

    Interpretation: it's the number of axes you'd need to capture
    the variance if it were uniformly distributed.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    # Covariance matrix C = (Xc.T @ Xc) / (N - 1); the constant
    # cancels in tr(C)² / tr(C²), so we work with S = Xc.T @ Xc.
    # tr(S)   = ||Xc||_F²
    # tr(S²)  = ||Xc.T @ Xc||_F² = ||Xc @ Xc.T||_F² (Frobenius cyclic)
    # The d×d form is cheaper when N > d; we use it directly.
    S = Xc.T @ Xc
    tr_S = np.trace(S)
    tr_S2 = np.sum(S * S)                          # = tr(S @ S)
    return float(tr_S ** 2 / max(tr_S2, 1e-30))
