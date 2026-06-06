"""Tests for src/diagnostics.py.

Two extreme structures pin down the expected behavior:
  - Isotropic gaussian: anisotropy ≈ 0, PR ≈ d
  - Rank-1 (collapsed to a line): anisotropy ≈ 1, PR ≈ 1
"""
import numpy as np

from src.diagnostics import anisotropy, fisher_trace_ratio, participation_ratio


# ----- anisotropy -----

def test_anisotropy_isotropic_is_near_zero():
    """Random gaussian in 50D → mean pairwise cosine ≈ 0."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 50))
    a = anisotropy(X)
    assert abs(a) < 0.05


def test_anisotropy_collapsed_to_line_is_near_one():
    """All vectors along +x → every pair cosine = 1."""
    X = np.zeros((100, 10))
    X[:, 0] = np.linspace(1.0, 2.0, 100)  # all positive scalars on x-axis
    a = anisotropy(X)
    assert a > 0.99


def test_anisotropy_antipodal_is_negative():
    """Half on +x, half on -x → cosine = -1 for cross-group pairs,
    +1 for same-group; with equal sizes the mean is around 0
    (with -1/(n-1) bias) — should be CLEARLY less than the
    collapsed-to-line case above."""
    X = np.zeros((100, 5))
    X[:50, 0] = 1.0
    X[50:, 0] = -1.0
    a = anisotropy(X)
    # Cross-group pairs (50*50) contribute -1; same-group pairs contribute +1.
    # Strict upper tri excludes diagonal. Result is essentially 0 minus 1/(n-1) ish.
    assert a < 0.1


def test_anisotropy_subsamples_when_large():
    """sample_size limit should cap the effective N (smoke test:
    just ensure the function returns a finite scalar quickly)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, 20))
    a = anisotropy(X, sample_size=500, seed=0)
    assert np.isfinite(a)
    assert abs(a) < 0.1


# ----- participation_ratio -----

def test_participation_ratio_isotropic_is_near_d():
    """For isotropic gaussian in d-D, PR ≈ d (within sample noise)."""
    rng = np.random.default_rng(0)
    d = 20
    X = rng.normal(size=(2000, d))
    pr = participation_ratio(X)
    # With N=2000, d=20 sample covariance is very close to identity → PR ≈ d
    assert abs(pr - d) < 2.0


def test_participation_ratio_rank_one_is_near_one():
    """All variance on one axis → PR = 1 exactly (up to fp tol)."""
    X = np.zeros((100, 8))
    X[:, 0] = np.linspace(-1.0, 1.0, 100)
    pr = participation_ratio(X)
    assert abs(pr - 1.0) < 1e-6


def test_participation_ratio_rank_two_is_about_two():
    """Variance equally split across two axes → PR ≈ 2."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(1000, 2))
    X = np.zeros((1000, 10))
    X[:, :2] = base
    pr = participation_ratio(X)
    assert abs(pr - 2.0) < 0.2


# ----- fisher_trace_ratio -----

def _three_blobs(n_per=80, d=10, sep=10.0, std=0.5, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array(
        [[sep] + [0.0] * (d - 1),
         [0.0, sep] + [0.0] * (d - 2),
         [0.0, 0.0, sep] + [0.0] * (d - 3)],
        dtype=np.float64,
    )
    X = np.vstack([rng.normal(c, std, size=(n_per, d)) for c in centers])
    y = np.repeat(np.arange(3), n_per)
    return X, y


def test_fisher_ratio_large_for_separated_blobs():
    """Well-separated blobs: between-class scatter >> within → J >> 1."""
    X, y = _three_blobs()
    assert fisher_trace_ratio(X, y) > 20.0


def test_fisher_ratio_near_zero_for_random_labels():
    """Random labels on the same X → class means coincide → J ≈ 0."""
    X, _ = _three_blobs()
    rng = np.random.default_rng(1)
    y_rand = rng.integers(0, 3, size=X.shape[0])
    assert fisher_trace_ratio(X, y_rand) < 0.2


def test_fisher_ratio_eta2_identity():
    """η² = J/(J+1) should equal tr(S_B)/tr(S_T) computed directly."""
    X, y = _three_blobs()
    J = fisher_trace_ratio(X, y)
    eta2 = J / (J + 1)
    # direct: between-var fraction of total var
    mu = X.mean(0)
    sb = sum(len(X[y == c]) * ((X[y == c].mean(0) - mu) ** 2).sum()
             for c in np.unique(y))
    st = ((X - mu) ** 2).sum()
    assert abs(eta2 - sb / st) < 1e-9
