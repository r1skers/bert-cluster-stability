"""Tests for src/probe.py.

Mirrors the synthetic-data philosophy of test_metrics.py: build data
with KNOWN linear separability so assertions are unambiguous. Three
well-separated gaussian blobs are perfectly linearly separable, so any
working linear probe should approach accuracy 1.0; random labels over
the same X should collapse to chance.
"""
import numpy as np
import pytest

from src.probe import (
    chance_accuracy,
    evaluate_layer,
    majority_accuracy,
    make_lda_probe,
    make_logreg_probe,
)


def make_separable(n_per: int = 80, d: int = 10, seed: int = 0):
    """3 well-separated isotropic gaussian blobs in d-D (linearly separable)."""
    rng = np.random.default_rng(seed)
    centers = np.array(
        [[10.0] + [0.0] * (d - 1),
         [0.0, 10.0] + [0.0] * (d - 2),
         [0.0, 0.0, 10.0] + [0.0] * (d - 3)],
        dtype=np.float64,
    )
    X = np.vstack([rng.normal(c, 0.5, size=(n_per, d)) for c in centers])
    y = np.repeat(np.arange(3), n_per)
    return X, y


# ----- estimator factories -----

def test_factory_returns_fresh_estimators():
    """Each call to the factory yields a distinct, unfitted estimator,
    so CV folds never re-fit a shared object."""
    factory = make_logreg_probe()
    a, b = factory(), factory()
    assert a is not b
    # an unfitted pipeline's classifier has no learned coef_ yet
    assert not hasattr(a.named_steps["clf"], "coef_")


def test_logreg_factory_respects_C():
    factory = make_logreg_probe(C=0.25)
    est = factory()
    assert est.named_steps["clf"].C == 0.25


# ----- baselines -----

def test_chance_accuracy():
    y = np.repeat(np.arange(20), 5)
    assert chance_accuracy(y) == pytest.approx(1.0 / 20)


def test_majority_accuracy_skewed():
    """Majority baseline = largest class share."""
    y = np.array([0, 0, 0, 0, 1, 1])  # class 0 is 4/6
    assert majority_accuracy(y) == pytest.approx(4 / 6)


# ----- evaluation loop -----

def test_evaluate_layer_returns_expected_keys():
    X, y = make_separable()
    out = evaluate_layer(X, y, make_logreg_probe(), cv=3)
    assert set(out.keys()) == {
        "accuracy_mean", "accuracy_std",
        "macro_f1_mean", "macro_f1_std",
        "chance", "majority",
    }
    for v in out.values():
        assert isinstance(v, float)


def test_logreg_recovers_separable_data():
    """Linearly separable blobs → near-perfect probe accuracy."""
    X, y = make_separable()
    out = evaluate_layer(X, y, make_logreg_probe(), cv=5)
    assert out["accuracy_mean"] > 0.95
    assert out["macro_f1_mean"] > 0.95


def test_lda_recovers_separable_data():
    """The shared harness works identically for the 5.3 LDA factory."""
    X, y = make_separable()
    out = evaluate_layer(X, y, make_lda_probe(), cv=5)
    assert out["accuracy_mean"] > 0.95


def test_probe_collapses_to_chance_on_random_labels():
    """Random labels carry no linear signal → accuracy near chance (1/3)."""
    X, _ = make_separable()
    rng = np.random.default_rng(1)
    y_rand = rng.integers(0, 3, size=X.shape[0])
    out = evaluate_layer(X, y_rand, make_logreg_probe(), cv=5)
    # comfortably below the separable case; near 1/3 chance
    assert out["accuracy_mean"] < 0.5
