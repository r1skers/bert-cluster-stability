"""Generic numpy-array cache backed by .npz files.

This module is deliberately decoupled from BERT. It only knows how
to:
    1. derive a cache file path from (model_tag, n_docs)
    2. either load arrays from an existing .npz, OR run a user-
       supplied compute function and save its dict-of-arrays result

Same primitive will be reused later to cache clustering labels,
stability scores, and any other expensive intermediate.
"""
import os
from typing import Callable, Dict

import numpy as np


def embedding_cache_path(
    model_tag: str,
    model_name: str,
    n_docs: int,
    sample_seed: int,
    max_length: int,
    min_chars: int,
    cache_dir: str = "outputs/cache",
) -> str:
    """Compose the on-disk cache path for an embedding run.

    Filename encodes every parameter that, if changed, would
    invalidate previously-cached embeddings:

        {model_tag}_{model_name}_n{n_docs}_s{sample_seed}
        _len{max_length}_min{min_chars}.npz

    Example
    -------
    embedding_cache_path(
        "pretrained", "bert-base-uncased",
        n_docs=2000, sample_seed=42, max_length=512, min_chars=30,
    )
        -> outputs/cache/pretrained_bert-base-uncased_n2000_s42_len512_min30.npz

    Note
    ----
    Even though the params are in the filename, callers should
    ALSO save them as `meta_*` keys inside the .npz so the file
    is self-describing if a future analyst inspects it directly.
    """
    name = (
        f"{model_tag}_{model_name}"
        f"_n{n_docs}_s{sample_seed}"
        f"_len{max_length}_min{min_chars}.npz"
    )
    return os.path.join(cache_dir, name)


def compute_or_load(
    path: str,
    compute_fn: Callable[[], Dict[str, np.ndarray]],
    *,
    force: bool = False,
) -> Dict[str, np.ndarray]:
    """Load cached arrays from `path`, or compute + save them.

    Parameters
    ----------
    path : str
        .npz file path. Parent directories are created if missing.
    compute_fn : zero-arg callable
        Returns a dict mapping string keys to numpy arrays. Called
        only on cache miss (or when `force=True`).
    force : bool
        If True, ignore any existing cache file and recompute.

    Returns
    -------
    dict[str, np.ndarray]
        Same shape as compute_fn's return value. When loading from
        cache, returns a plain dict (not a NpzFile, which has lazy
        loading and would surprise callers).
    """
    if not force and os.path.exists(path):
        with np.load(path) as npz:
            return {k: npz[k] for k in npz.files}

    result = compute_fn()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, **result)
    return result
