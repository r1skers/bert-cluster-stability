"""Tests for src/transforms.py."""
import numpy as np
import pytest

from src.transforms import (
    apply_transform,
    centered_l2,
    pca_project,
    pca_whiten,
    remove_top_pcs,
    transform_registry,
)


def test_centered_l2_zero_mean_direction_and_unit_rows():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 12)) + 5.0
    out = centered_l2(X)
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_remove_top_pc_reduces_dominant_projection():
    rng = np.random.default_rng(0)
    n = 200
    signal = rng.normal(size=(n, 1)) * 10.0
    noise = rng.normal(scale=0.1, size=(n, 4))
    X = np.hstack([signal, noise])

    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    pc1 = vt[0]
    before = np.linalg.norm(Xc @ pc1)

    out = remove_top_pcs(X, 1, normalize=False)
    after = np.linalg.norm(out @ pc1)
    assert after < before * 1e-6


def test_pca_project_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 20))
    out = pca_project(X, 7, normalize=False)
    assert out.shape == (50, 7)


def test_pca_whiten_shape_and_unit_rows_when_normalized():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 15))
    out = pca_whiten(X, 8, normalize=True)
    assert out.shape == (60, 8)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-10)


def test_registry_contains_expected_baselines():
    names = set(transform_registry())
    assert {"raw", "l2", "centered", "centered_l2"}.issubset(names)
    assert {"drop_pc1_l2", "pca50_l2", "whiten50_l2"}.issubset(names)


def test_apply_transform_rejects_unknown():
    with pytest.raises(ValueError):
        apply_transform(np.zeros((3, 3)), "not_a_transform")
