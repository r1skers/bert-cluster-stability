"""Tests for src/data.py."""
import numpy as np
import pytest

from src.data import NewsgroupsData, filter_short_docs, load_20newsgroups


def test_load_smoke():
    """load_20newsgroups returns aligned docs and labels for 20 topics."""
    data = load_20newsgroups(subset="train")
    assert len(data.docs) == len(data.labels)
    assert len(data.target_names) == 20
    assert data.labels.min() >= 0
    assert data.labels.max() < 20


def test_filter_drops_short_docs():
    """filter_short_docs removes empty and short docs but keeps long ones."""
    fake = NewsgroupsData(
        docs=["", "tiny", "a" * 100, "b" * 200],
        labels=np.array([0, 1, 2, 3]),
        target_names=["t0", "t1", "t2", "t3"],
    )
    out = filter_short_docs(fake, min_chars=30)
    assert len(out.docs) == 2
    assert list(out.labels) == [2, 3]


def test_filter_preserves_label_alignment():
    """A no-op filter (min_chars=0) must not reorder docs and labels."""
    data = load_20newsgroups(subset="train")
    first_doc_before = data.docs[0]
    first_label_before = int(data.labels[0])

    filtered = filter_short_docs(data, min_chars=0)

    assert filtered.docs[0] == first_doc_before
    assert int(filtered.labels[0]) == first_label_before
    assert len(filtered) == len(data)
