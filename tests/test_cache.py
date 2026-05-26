"""Tests for src/cache.py.

These tests do NOT touch BERT — the cache layer is generic numpy-
array I/O. We use a synthetic `compute_fn` and pytest's `tmp_path`
fixture so each test gets an isolated scratch directory.
"""
import os

import numpy as np

from src.cache import compute_or_load, embedding_cache_path


def test_embedding_cache_path_format():
    """Sanity that the path-naming convention is what we expect."""
    p = embedding_cache_path(
        "pretrained",
        "bert-base-uncased",
        n_docs=2000,
        sample_seed=42,
        max_length=512,
        min_chars=30,
        cache_dir="outputs/cache",
    )
    assert p.endswith(
        "pretrained_bert-base-uncased_n2000_s42_len512_min30.npz"
    )
    assert "outputs" in p and "cache" in p


def test_cache_miss_then_hit(tmp_path):
    """First call computes and writes; second call loads without
    re-invoking the compute function."""
    path = str(tmp_path / "x.npz")
    call_counter = {"n": 0}

    def compute():
        call_counter["n"] += 1
        return {
            "embeddings": np.arange(12).reshape(3, 4).astype(np.float32),
            "labels": np.array([0, 1, 2]),
        }

    # cache miss -> compute called
    out1 = compute_or_load(path, compute)
    assert call_counter["n"] == 1
    assert os.path.exists(path)
    assert out1["embeddings"].shape == (3, 4)

    # cache hit -> compute NOT called again
    out2 = compute_or_load(path, compute)
    assert call_counter["n"] == 1  # unchanged
    np.testing.assert_array_equal(out1["embeddings"], out2["embeddings"])
    np.testing.assert_array_equal(out1["labels"], out2["labels"])


def test_force_recomputes(tmp_path):
    """`force=True` must bypass an existing cache file."""
    path = str(tmp_path / "y.npz")
    call_counter = {"n": 0}

    def compute():
        call_counter["n"] += 1
        return {"a": np.array([float(call_counter["n"])])}

    compute_or_load(path, compute)              # writes a=1.0
    out2 = compute_or_load(path, compute)        # cache hit, a still 1.0
    out3 = compute_or_load(path, compute, force=True)  # forced, a=2.0
    assert call_counter["n"] == 2
    assert out2["a"][0] == 1.0
    assert out3["a"][0] == 2.0


def test_creates_parent_dir(tmp_path):
    """compute_or_load must create non-existent parent directories."""
    nested = tmp_path / "a" / "b" / "c.npz"
    assert not nested.parent.exists()

    def compute():
        return {"x": np.zeros(3)}

    compute_or_load(str(nested), compute)
    assert nested.exists()
