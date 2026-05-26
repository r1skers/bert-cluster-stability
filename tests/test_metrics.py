"""Tests for src/metrics.py.

We use synthetic data with KNOWN structure so the test assertions
are unambiguous: three well-separated gaussian clusters in 10D
should give high silhouette and near-perfect NMI; randomly
shuffled labels should give near-zero NMI; etc.
"""
import numpy as np

from src.metrics import (
    cluster_metrics,
    kmeans_labels,
    l2_normalize,
    nmi,
    prepare,
    purity,
)


def make_clusters(n_per: int = 100, d: int = 10, seed: int = 0):
    """Generate 3 well-separated isotropic gaussian clusters in d-D."""
    rng = np.random.default_rng(seed)
    centers = np.array(
        [[10.0] + [0.0] * (d - 1),
         [0.0, 10.0] + [0.0] * (d - 2),
         [0.0, 0.0, 10.0] + [0.0] * (d - 3)],
        dtype=np.float64,
    )
    X = np.vstack([rng.normal(c, 0.5, size=(n_per, d)) for c in centers])
    y = np.repeat(np.arange(3), n_per)
    return X, y


# ----- l2_normalize -----

def test_l2_normalize_unit_norm():
    """After L2 normalize, every row has norm 1.0 (within tol)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 8))
    Xn = l2_normalize(X)
    norms = np.linalg.norm(Xn, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_prepare_modes():
    """Each mode does what its name says; bad mode raises."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.array_equal(prepare(X, "raw"), X)
    centered = prepare(X, "centered")
    assert np.allclose(centered.mean(axis=0), 0.0)
    l2 = prepare(X, "l2")
    np.testing.assert_allclose(np.linalg.norm(l2, axis=1), 1.0, atol=1e-10)


def test_prepare_rejects_bad_mode():
    import pytest
    with pytest.raises(ValueError):
        prepare(np.zeros((3, 3)), mode="bogus")


# ----- KMeans wrapper -----

def test_kmeans_determinism():
    """Same seed → identical labels (n_init runs deterministic too)."""
    X, _ = make_clusters()
    a = kmeans_labels(X, k=3, seed=0)
    b = kmeans_labels(X, k=3, seed=0)
    assert np.array_equal(a, b)


def test_kmeans_recovers_3_clusters():
    """3 well-separated gaussian blobs are recovered: NMI > 0.95."""
    X, y = make_clusters()
    pred = kmeans_labels(X, k=3, seed=0)
    assert nmi(y, pred) > 0.95


# ----- alignment metrics -----

def test_nmi_near_one_for_clean_recovery():
    """KMeans on well-separated clusters → NMI ≈ 1."""
    X, y = make_clusters()
    pred = kmeans_labels(X, k=3, seed=0)
    assert nmi(y, pred) > 0.95


def test_nmi_near_zero_for_random_labels():
    """Randomly shuffled labels have ~no mutual info with truth."""
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(3), 100)
    pred = rng.integers(0, 3, size=300)
    assert nmi(y, pred) < 0.1


def test_purity_perfect_when_clusters_match():
    """Same labels = purity 1.0."""
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    assert purity(y, y) == 1.0


def test_purity_one_third_for_random_3way():
    """Random uniform 3-way prediction over 3 balanced classes
    has expected purity ≈ 1/3, but realized value depends on the
    specific draw. We just check it's clearly below 0.6."""
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(3), 200)
    pred = rng.integers(0, 3, size=600)
    assert purity(y, pred) < 0.6


# ----- combined entry point -----

def test_cluster_metrics_returns_all_five_keys():
    """`cluster_metrics` should return the five named float metrics."""
    X, y = make_clusters()
    out = cluster_metrics(X, y, k=3, seed=0, mode="raw")
    assert set(out.keys()) == {
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "nmi",
        "purity",
    }
    for v in out.values():
        assert isinstance(v, float)


def test_cluster_metrics_clean_data_is_good():
    """On clean data, silhouette > 0.5 and NMI > 0.9."""
    X, y = make_clusters()
    out = cluster_metrics(X, y, k=3, seed=0, mode="raw")
    assert out["silhouette"] > 0.5
    assert out["nmi"] > 0.9
    assert out["purity"] > 0.9


def test_cluster_metrics_l2_mode_works():
    """L2 mode should also recover the clusters here (they're
    radially separable in the synthetic setup)."""
    X, y = make_clusters()
    out = cluster_metrics(X, y, k=3, seed=0, mode="l2")
    assert out["nmi"] > 0.9
