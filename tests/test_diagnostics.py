"""Tests for src/diagnostics.py.

Two extreme structures pin down the expected behavior:
  - Isotropic gaussian: anisotropy ≈ 0, PR ≈ d
  - Rank-1 (collapsed to a line): anisotropy ≈ 1, PR ≈ 1
"""
import numpy as np

from src.diagnostics import anisotropy, participation_ratio


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
