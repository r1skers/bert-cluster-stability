"""20 Newsgroups data loading and preprocessing.

Pipeline:
    1. fetch_20newsgroups with header / footer / quote removal
       (else trivial features like newsgroup names in mail headers
       would let any clustering recover topic labels via leakage)
    2. filter empty / very short documents
       (after cleaning, some docs collapse to <30 chars; once
       BERT-tokenized these are mostly [PAD] and mean-pool over
       non-pad tokens degenerates toward the [CLS] embedding,
       polluting cluster geometry without carrying topic info)
    3. return a NewsgroupsData holder with docs + integer labels
       + topic name list

This module deliberately does NOT touch BERT. It produces plain
strings + integer labels; downstream modules tokenize and embed.
"""
from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.datasets import fetch_20newsgroups


@dataclass
class NewsgroupsData:
    """Container for one split of the 20 Newsgroups corpus.

    Attributes
    ----------
    docs : list of str
        Cleaned document strings.
    labels : np.ndarray of int, shape (N,)
        Integer topic id for each doc, range [0, len(target_names)).
    target_names : list of str
        Human-readable topic names, indexed by label.
    """

    docs: List[str]
    labels: np.ndarray
    target_names: List[str]

    def __len__(self) -> int:
        return len(self.docs)


def load_20newsgroups(subset: str = "all") -> NewsgroupsData:
    """Load 20 Newsgroups with canonical noise removal applied.

    Parameters
    ----------
    subset : {'train', 'test', 'all'}
        Which split to fetch. Default 'all' = 18846 docs.

    Notes
    -----
    `remove=('headers', 'footers', 'quotes')` is the standard
    20 NG anti-leakage cleanup recommended by sklearn. Without it,
    headers contain the newsgroup name and signatures contain
    affiliations, both of which give the cluster algorithm a
    trivial shortcut.
    """
    # 把外部库的数据格式转换成自己项目内部稳定的数据格式
    bunch = fetch_20newsgroups(
        subset=subset,
        remove=("headers", "footers", "quotes"),
    )
    return NewsgroupsData(
        docs=list(bunch.data),
        labels=np.asarray(bunch.target),
        target_names=list(bunch.target_names),
    )


def filter_short_docs(
    data: NewsgroupsData, min_chars: int = 30
) -> NewsgroupsData:
    """Drop documents whose stripped length is below `min_chars`.

    The default threshold 30 is a pragmatic choice: roughly 5-6
    English words, below which a "doc" carries essentially no
    topic signal after BERT mean-pool.

    Returns a new NewsgroupsData; does not mutate the input.
    """
    keep = np.array([len(d.strip()) >= min_chars for d in data.docs])
    return NewsgroupsData(
        docs=[d for d, k in zip(data.docs, keep) if k],
        labels=data.labels[keep],
        target_names=data.target_names,
    )


def summarize(data: NewsgroupsData) -> None:
    """Print a quick sanity summary of a NewsgroupsData split."""
    N = len(data.docs)
    lengths = np.array([len(d) for d in data.docs])
    print(f"docs:        {N}")
    print(f"topics:      {len(data.target_names)}")
    print(
        f"char length: min={lengths.min()} "
        f"median={int(np.median(lengths))} "
        f"mean={lengths.mean():.0f} "
        f"max={lengths.max()}"
    )
    counts = np.bincount(data.labels, minlength=len(data.target_names))
    print("per-topic doc counts:")
    for name, c in zip(data.target_names, counts):
        print(f"  {c:5d}  {name}")
