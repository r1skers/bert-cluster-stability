"""Tests for cached layer pooling variants."""
import numpy as np
import pytest

from src.pooling import apply_pooling, last4_concat, last4_mean, layer_representation


def make_embeddings():
    # Shape (N=3, layers=13, H=2). Values make layer identity visible.
    E = np.zeros((3, 13, 2), dtype=float)
    for layer in range(13):
        E[:, layer, :] = layer
    return E


def test_layer_representation_shape_and_values():
    E = make_embeddings()
    out = layer_representation(E, 12)
    assert out.shape == (3, 2)
    assert np.all(out == 12)


def test_last4_mean_averages_layers_9_to_12():
    E = make_embeddings()
    out = last4_mean(E)
    assert out.shape == (3, 2)
    assert np.all(out == (9 + 10 + 11 + 12) / 4)


def test_last4_concat_shape_and_order():
    E = make_embeddings()
    out = last4_concat(E)
    assert out.shape == (3, 8)
    np.testing.assert_array_equal(out[0], np.array([9, 9, 10, 10, 11, 11, 12, 12]))


def test_apply_pooling_rejects_unknown():
    with pytest.raises(ValueError):
        apply_pooling(make_embeddings(), "nope")
