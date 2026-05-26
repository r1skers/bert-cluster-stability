"""Tests for src/interpret.py.

We use tiny synthetic data with known structure so the assertions
are exact.
"""
import numpy as np

from src.interpret import (
    build_contingency,
    cluster_summary,
    top_keywords_per_cluster,
    top_keywords_per_cluster_ctfidf,
)


def test_contingency_basic():
    """3 clusters × 3 topics, known assignments → exact counts."""
    # cluster_labels:  [0, 0, 1, 1, 2, 2]
    # true_labels:     [0, 0, 1, 2, 2, 2]
    cl = np.array([0, 0, 1, 1, 2, 2])
    tl = np.array([0, 0, 1, 2, 2, 2])
    c = build_contingency(cl, tl, k=3, n_topics=3)
    expected = np.array([
        [2, 0, 0],  # cluster 0 → 2 in topic 0
        [0, 1, 1],  # cluster 1 → 1 in topic 1, 1 in topic 2
        [0, 0, 2],  # cluster 2 → 2 in topic 2
    ], dtype=np.int64)
    np.testing.assert_array_equal(c, expected)


def test_contingency_handles_missing_clusters():
    """A k larger than max cluster id still returns correctly-shaped output."""
    cl = np.array([0, 0, 0])  # cluster 1 and 2 are empty
    tl = np.array([1, 1, 2])
    c = build_contingency(cl, tl, k=3, n_topics=3)
    assert c.shape == (3, 3)
    assert c[1].sum() == 0
    assert c[2].sum() == 0


def test_summary_picks_majority_topic():
    """A cluster with 8/10 in topic A → major_topic=A, fraction=0.8."""
    # 10 points all in cluster 0; 8 topic A, 2 topic B
    cl = np.zeros(10, dtype=int)
    tl = np.array([0]*8 + [1]*2)
    out = cluster_summary(cl, tl, target_names=["topicA", "topicB", "topicC"])
    assert len(out) == 1
    row = out[0]
    assert row["cluster_id"] == 0
    assert row["size"] == 10
    assert row["major_topic"] == "topicA"
    assert abs(row["major_topic_fraction"] - 0.8) < 1e-9


def test_summary_sorted_by_size_desc():
    """Larger clusters appear first in the output."""
    # cluster 0 has 2 points, cluster 1 has 5, cluster 2 has 3
    cl = np.array([0, 0, 1, 1, 1, 1, 1, 2, 2, 2])
    tl = np.zeros(10, dtype=int)
    out = cluster_summary(cl, tl, target_names=["topicA"])
    sizes = [r["size"] for r in out]
    assert sizes == [5, 3, 2]


def test_top_keywords_basic():
    """Two clusters, each defined by distinctive vocabulary."""
    docs = [
        "the cat sat on the mat",          # cluster 0
        "a cat plays with a mat",          # cluster 0
        "neural networks learn weights",   # cluster 1
        "deep networks have many weights", # cluster 1
    ]
    cl = np.array([0, 0, 1, 1])
    kw = top_keywords_per_cluster(docs, cl, k=2, top_n=3)
    # cluster 0 should contain "cat" and/or "mat"
    assert any(w in {"cat", "mat"} for w in kw[0])
    # cluster 1 should contain "networks" and/or "weights"
    assert any(w in {"networks", "weights"} for w in kw[1])


def test_ctfidf_suppresses_cross_cluster_fillers():
    """c-TF-IDF should prefer cluster-specific terms over shared filler."""
    docs = [
        "people people people kidney doctor cancer",
        "people people people doctor kidney physician",
        "people people people engine ford volvo",
        "people people people ford engine cars",
    ]
    cl = np.array([0, 0, 1, 1])

    kw = top_keywords_per_cluster_ctfidf(docs, cl, k=2, top_n=4)

    assert "people" not in kw[0]
    assert "people" not in kw[1]
    assert any(w in {"kidney", "doctor", "cancer", "physician"} for w in kw[0])
    assert any(w in {"engine", "ford", "volvo", "cars"} for w in kw[1])
