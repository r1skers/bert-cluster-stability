"""Cluster interpretability helpers.

After clustering BERT embeddings, we want to know WHAT each cluster
contains. These three pure functions answer that:

    build_contingency        :  cluster × topic count matrix
    cluster_summary          :  per-cluster {size, major_topic, ...}
    top_keywords_per_cluster :  per-cluster TF-IDF top words

All three are pure data transformations — no I/O, no plotting —
so they live in src/ and are easy to test.
"""
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


EXTENDED_STOPWORDS = set(ENGLISH_STOP_WORDS) | {
    # Conversational fillers common in 20 Newsgroups posts.
    "also",
    "does",
    "did",
    "don",
    "dont",
    "going",
    "got",
    "just",
    "know",
    "like",
    "people",
    "really",
    "right",
    "said",
    "says",
    "think",
    "use",
    "using",
    "want",
    "way",
    # Email/news metadata leftovers.
    "address",
    "article",
    "com",
    "edu",
    "lines",
    "mail",
    "organization",
    "posting",
    "subject",
    "thanks",
    "university",
    # Common 20NG signature / name artifacts that survive sklearn cleanup.
    "baden",
    "bison",
    "cadre",
    "chastity",
    "cobb",
    "deane",
    "dobbs",
    "dsl",
    "frode",
    "geb",
    "ihno",
    "intellect",
    "loser",
    "pitt",
    "skepticism",
}


def build_contingency(
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
    k: int,
    n_topics: int,
) -> np.ndarray:
    """Build a k × n_topics integer count matrix.

    `c[i, j]` = number of points in cluster i that have true label j.

    Parameters
    ----------
    cluster_labels : (N,) int array, values in [0, k)
    true_labels    : (N,) int array, values in [0, n_topics)
    """
    c = np.zeros((k, n_topics), dtype=np.int64)
    for cl, tl in zip(cluster_labels, true_labels):
        c[cl, tl] += 1
    return c


def cluster_summary(
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
    target_names: List[str],
) -> List[Dict]:
    """Per-cluster summary for the human reader.

    Returns a list of dicts (one per cluster), sorted by size desc:
        cluster_id : int
        size : int
        major_topic : str        — most frequent true label in cluster
        major_topic_fraction : float in [0, 1]
        top_3_topics : str       — "topicA(123); topicB(45); topicC(12)"

    A cluster with high `major_topic_fraction` is "pure" — its
    members are dominated by one true topic. Low fraction means
    the cluster is a mixture and KMeans found a non-topic axis.
    """
    rows: List[Dict] = []
    n_topics = len(target_names)
    for c in sorted(np.unique(cluster_labels).tolist()):
        mask = cluster_labels == c
        size = int(mask.sum())
        if size == 0:
            continue
        counts = np.bincount(true_labels[mask], minlength=n_topics)
        order = np.argsort(counts)[::-1]
        major_idx = int(order[0])
        major_frac = float(counts[major_idx] / size)
        top_3 = "; ".join(
            f"{target_names[int(order[i])]}({int(counts[order[i]])})"
            for i in range(min(3, n_topics))
            if counts[order[i]] > 0
        )
        rows.append({
            "cluster_id": int(c),
            "size": size,
            "major_topic": target_names[major_idx],
            "major_topic_fraction": major_frac,
            "top_3_topics": top_3,
        })
    rows.sort(key=lambda r: r["size"], reverse=True)
    return rows


def top_keywords_per_cluster(
    docs: List[str],
    cluster_labels: np.ndarray,
    k: int,
    top_n: int = 8,
    max_features: int = 10000,
) -> Dict[int, List[str]]:
    """Per-cluster characteristic words via mean TF-IDF.

    Pipeline
    --------
    1. Fit a `TfidfVectorizer` over all docs (English stop words,
       3+ letter alphabetic tokens, lowercased).
    2. For each cluster c, take the mean TF-IDF row over docs in c.
    3. Return top_n vocabulary terms by mean score.

    This is the simplest interpretable approach — there's a fancier
    c-TF-IDF variant (BERTopic-style) that compares within-cluster
    to outside-cluster, but for a first pass mean-within is enough.

    Returns
    -------
    dict mapping cluster_id -> list of top terms (strings).
    Empty list if a cluster has zero members.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        token_pattern=r"\b[a-z]{3,}\b",
        lowercase=True,
    )
    X = vectorizer.fit_transform(docs)              # (N, V) sparse
    vocab = vectorizer.get_feature_names_out()

    keywords: Dict[int, List[str]] = {}
    for c in range(k):
        mask = cluster_labels == c
        if not mask.any():
            keywords[c] = []
            continue
        mean_tfidf = np.asarray(X[mask].mean(axis=0)).ravel()
        top_idx = np.argsort(mean_tfidf)[::-1][:top_n]
        keywords[c] = [vocab[i] for i in top_idx]
    return keywords


def top_keywords_per_cluster_ctfidf(
    docs: List[str],
    cluster_labels: np.ndarray,
    k: int,
    top_n: int = 8,
    max_features: int = 10000,
) -> Dict[int, List[str]]:
    """Per-cluster keywords via BERTopic-style c-TF-IDF.

    Each cluster is collapsed into one "mega-document", then TF-IDF
    is fit across those k mega-documents. Terms that appear across
    many clusters get low IDF, while terms specific to one cluster
    are promoted.

    This is better than mean document-level TF-IDF for this project
    because conversational words such as "people", "think", and
    "know" can be frequent within many clusters; c-TF-IDF pushes
    those cross-cluster words down.
    """
    cluster_docs = []
    for c in range(k):
        text = " ".join(d for d, label in zip(docs, cluster_labels) if label == c)
        cluster_docs.append(text or "emptyclusterplaceholder")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=list(EXTENDED_STOPWORDS),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b",
        lowercase=True,
        max_df=0.7,
    )
    X = vectorizer.fit_transform(cluster_docs)      # (k, V) sparse
    vocab = vectorizer.get_feature_names_out()

    keywords: Dict[int, List[str]] = {}
    for c in range(k):
        scores = np.asarray(X[c].todense()).ravel()
        if scores.size == 0 or np.all(scores == 0):
            keywords[c] = []
            continue
        top_idx = np.argsort(scores)[::-1][:top_n]
        keywords[c] = [vocab[i] for i in top_idx if scores[i] > 0]
    return keywords
