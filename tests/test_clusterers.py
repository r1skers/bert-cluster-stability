"""Tests for alternative clusterers."""
import numpy as np
import pytest

from src.clusterers import (
    agglomerative_cosine_average_labels,
    agglomerative_ward_labels,
    cluster_labels,
    gaussian_mixture_labels,
    spherical_kmeans_labels,
)
from src.metrics import nmi


def make_directional_clusters(n_per=80, d=12, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.eye(3, d)
    chunks = []
    for c in centers:
        x = c + rng.normal(scale=0.05, size=(n_per, d))
        chunks.append(x)
    X = np.vstack(chunks)
    y = np.repeat(np.arange(3), n_per)
    return X, y


def test_spherical_kmeans_recovers_directional_clusters():
    X, y = make_directional_clusters()
    pred = spherical_kmeans_labels(X, k=3, seed=0, n_init=5)
    assert pred.shape == y.shape
    assert nmi(y, pred) > 0.95


def test_spherical_kmeans_is_deterministic_for_same_seed():
    X, _ = make_directional_clusters()
    a = spherical_kmeans_labels(X, k=3, seed=7, n_init=5)
    b = spherical_kmeans_labels(X, k=3, seed=7, n_init=5)
    np.testing.assert_array_equal(a, b)


def test_cluster_labels_dispatches_lloyd_and_spherical():
    X, y = make_directional_clusters()
    a = cluster_labels(X, k=3, seed=0, clusterer="lloyd")
    b = cluster_labels(X, k=3, seed=0, clusterer="spherical")
    assert nmi(y, a) > 0.95
    assert nmi(y, b) > 0.95


def test_agglomerative_cosine_recovers_directional_clusters():
    X, y = make_directional_clusters()
    pred = agglomerative_cosine_average_labels(X, k=3)
    assert pred.shape == y.shape
    assert nmi(y, pred) > 0.95


def test_agglomerative_ward_recovers_directional_clusters():
    X, y = make_directional_clusters()
    pred = agglomerative_ward_labels(X, k=3)
    assert pred.shape == y.shape
    assert nmi(y, pred) > 0.95


def test_gaussian_mixture_diag_recovers_directional_clusters():
    X, y = make_directional_clusters()
    pred = gaussian_mixture_labels(X, k=3, seed=0, covariance_type="diag", n_init=2)
    assert pred.shape == y.shape
    assert nmi(y, pred) > 0.95


def test_gaussian_mixture_full_recovers_directional_clusters():
    X, y = make_directional_clusters()
    pred = gaussian_mixture_labels(X, k=3, seed=0, covariance_type="full", n_init=2)
    assert pred.shape == y.shape
    assert nmi(y, pred) > 0.95


def test_cluster_labels_dispatches_agglomerative_cosine():
    X, y = make_directional_clusters()
    pred = cluster_labels(X, k=3, clusterer="agglo_cosine")
    assert nmi(y, pred) > 0.95


def test_cluster_labels_dispatches_agglomerative_ward():
    X, y = make_directional_clusters()
    pred = cluster_labels(X, k=3, clusterer="agglo_ward")
    assert nmi(y, pred) > 0.95


def test_cluster_labels_dispatches_gmm_diag_and_full():
    X, y = make_directional_clusters()
    diag = cluster_labels(X, k=3, seed=0, clusterer="gmm_diag")
    full = cluster_labels(X, k=3, seed=0, clusterer="gmm_full")
    assert nmi(y, diag) > 0.95
    assert nmi(y, full) > 0.95


def test_cluster_labels_rejects_unknown_backend():
    with pytest.raises(ValueError):
        cluster_labels(np.zeros((5, 2)), k=2, clusterer="mystery")
