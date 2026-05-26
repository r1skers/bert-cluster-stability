"""Tests for resampling stability helpers."""
import numpy as np
import pytest

from src.stability import (
    mean_pairwise_ari,
    resampling_stability,
    subsample_indices,
    transform_for_resample,
)


def make_blobs(n_per=40, d=8, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.eye(3, d) * 5.0
    X = np.vstack([
        center + rng.normal(scale=0.2, size=(n_per, d))
        for center in centers
    ])
    y = np.repeat(np.arange(3), n_per)
    return X, y


def test_mean_pairwise_ari_is_one_for_identical_partitions():
    labels = np.array([
        [0, 0, 1, 1],
        [7, 7, 3, 3],
        [2, 2, 9, 9],
    ])
    summary = mean_pairwise_ari(labels)
    assert summary.mean_ari == pytest.approx(1.0)
    assert summary.n_pairs == 3


def test_subsample_indices_are_reproducible_and_expected_size():
    a = subsample_indices(100, subset_fraction=0.8, n_runs=3, seed=5)
    b = subsample_indices(100, subset_fraction=0.8, n_runs=3, seed=5)
    assert len(a) == 3
    assert all(len(idx) == 80 for idx in a)
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left, right)


def test_transform_for_resample_supports_l2_and_whitening():
    X, _ = make_blobs()
    train_idx = np.arange(80)
    X_l2 = transform_for_resample(X, train_idx, "l2")
    X_w = transform_for_resample(X, train_idx, "whiten5_l2")
    assert X_l2.shape == X.shape
    assert X_w.shape == (len(X), 5)
    np.testing.assert_allclose(np.linalg.norm(X_l2, axis=1), 1.0, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(X_w, axis=1), 1.0, atol=1e-10)


def test_resampling_stability_is_high_for_clear_blobs_lloyd():
    X, _ = make_blobs()
    summary, label_runs = resampling_stability(
        X,
        k=3,
        transform="l2",
        clusterer="lloyd",
        subset_fraction=0.8,
        n_runs=5,
        subset_seed=0,
        cluster_seed=0,
    )
    assert label_runs.shape == (5, len(X))
    assert summary.mean_ari > 0.95


def test_resampling_stability_is_high_for_clear_blobs_spherical():
    X, _ = make_blobs()
    summary, label_runs = resampling_stability(
        X,
        k=3,
        transform="whiten5_l2",
        clusterer="spherical",
        subset_fraction=0.8,
        n_runs=5,
        subset_seed=0,
        cluster_seed=0,
    )
    assert label_runs.shape == (5, len(X))
    assert summary.mean_ari > 0.8
