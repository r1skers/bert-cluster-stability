"""Pooling variants over cached layer-wise embeddings.

The cache stores one vector per document per BERT layer:
    embeddings shape = (N, 13, H)

These helpers combine the layer axis without rerunning BERT. They are
kept separate from token-level pooling (CLS, no-special-token mean,
IDF-weighted mean), which would require a new forward pass.
"""
from typing import Callable, Dict

import numpy as np


def layer_representation(embeddings: np.ndarray, layer: int) -> np.ndarray:
    """Return one cached layer as (N, H)."""
    if embeddings.ndim != 3:
        raise ValueError("embeddings must have shape (N, layers, H)")
    if layer < 0 or layer >= embeddings.shape[1]:
        raise ValueError(f"layer out of range: {layer}")
    return embeddings[:, layer, :]


def last4_mean(embeddings: np.ndarray) -> np.ndarray:
    """Average encoder layers 9..12 into one (N, H) representation."""
    if embeddings.shape[1] < 13:
        raise ValueError("expected 13 BERT layers including embedding layer")
    return embeddings[:, 9:13, :].mean(axis=1)


def last4_concat(embeddings: np.ndarray) -> np.ndarray:
    """Concatenate encoder layers 9..12 into one (N, 4H) representation."""
    if embeddings.shape[1] < 13:
        raise ValueError("expected 13 BERT layers including embedding layer")
    n = embeddings.shape[0]
    return embeddings[:, 9:13, :].reshape(n, -1)


def pooling_registry() -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
    """Named pooling variants available without rerunning BERT."""
    return {
        "layer12": lambda E: layer_representation(E, 12),
        "layer11": lambda E: layer_representation(E, 11),
        "layer10": lambda E: layer_representation(E, 10),
        "layer9": lambda E: layer_representation(E, 9),
        "last4_mean": last4_mean,
        "last4_concat": last4_concat,
    }


def apply_pooling(embeddings: np.ndarray, name: str) -> np.ndarray:
    """Apply one named pooling variant to cached embeddings."""
    registry = pooling_registry()
    if name not in registry:
        known = ", ".join(sorted(registry))
        raise ValueError(f"unknown pooling {name!r}; known: {known}")
    return registry[name](embeddings)
