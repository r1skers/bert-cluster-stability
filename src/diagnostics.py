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


def fisher_trace_ratio(X: np.ndarray, y: np.ndarray) -> float:
    """Fisher separability criterion (trace form): tr(S_B) / tr(S_W).

    This is the *supervised* geometry counterpart to anisotropy /
    participation ratio: it uses the labels `y` to ask how separated
    the classes are, in a way that needs NO matrix inversion.

    Definitions
    -----------
        tr(S_W) = Σ_c Σ_{i∈c} ||x_i − μ_c||²     within-class scatter
        tr(S_B) = Σ_c n_c ||μ_c − μ||²            between-class scatter
        tr(S_T) = Σ_i ||x_i − μ||² = tr(S_B) + tr(S_W)

        Fisher ratio  J = tr(S_B) / tr(S_W)

    Interpretation
    --------------
        J large  → class means far apart relative to within-class spread
                   (classes compact and well separated).
        J small  → classes overlap; means close vs the scatter.

    The bounded sibling — between-class variance fraction — is a
    one-liner from J:  η² = tr(S_B)/tr(S_T) = J / (J + 1) ∈ [0, 1].

    Why the *trace* form (not the full Fisher eigenproblem): the full
    criterion needs S_W^{-1}, but S_W (d×d) estimated from few samples
    per class is near-singular at BERT scale (768 dims, ~70-114 rows
    per class). The trace ratio is the well-conditioned, inversion-free
    separability scalar — the geometric diagnostic, distinct from the
    LDA *classifier* (which handles the singularity via shrinkage).

    Parameters
    ----------
    X : (N, d) array
        Representation matrix. Caller decides preprocessing (e.g.
        standardize) — this function is pure geometry on the given X.
    y : (N,) int array
        Class labels.

    Returns
    -------
    float
        tr(S_B) / tr(S_W). Non-negative; 0 only if all class means
        coincide.
    """
    mu = X.mean(axis=0)
    sw = 0.0
    sb = 0.0
    for c in np.unique(y):
        Xc = X[y == c]
        mu_c = Xc.mean(axis=0)
        sw += float(((Xc - mu_c) ** 2).sum())
        sb += float(len(Xc) * ((mu_c - mu) ** 2).sum())
    return float(sb / max(sw, 1e-30))
