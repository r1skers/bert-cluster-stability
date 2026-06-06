"""Supervised linear-probe harness for BERT layer representations.

This module powers Artifact 5.2 (logistic-regression probe) and is
designed to be reused verbatim by Artifact 5.3 (Fisher / LDA probe).
Both views ask the same question —

    "How linearly separable are the 20 Newsgroups topics in BERT
     layer L's document-segment representation?"

— and differ ONLY in which linear estimator answers it:

    5.2  logistic regression   -> convex (SGD-style) separating optimum
    5.3  LDA / Fisher          -> Gaussian analytical separating optimum

so the entire cross-validated evaluation loop, baseline computation,
and downstream plotting are shared. A "view" is just an estimator
factory passed into `evaluate_layer`.

Design contract
---------------
* An *estimator factory* is a zero-arg callable returning a FRESH,
  unfitted sklearn estimator (so each CV fold / each layer gets its
  own model — never a re-fit of a shared object).
* Any feature scaling lives INSIDE the returned Pipeline, so it is
  fit on the training fold only. `cross_validate` then guarantees no
  train/test leakage through the scaler statistics.

Contrast with 5.1 (clustering)
------------------------------
5.1 found that *unsupervised* clustering is highly sensitive to
representation preprocessing (whitening was decisive) because it reads
whatever geometry dominates. A supervised linear probe is, in
contrast, approximately invariant to invertible global linear maps
(logistic regression can absorb centering / PCA / whitening into its
own weight matrix). The only preprocessing that genuinely changes a
logistic probe is per-row L2 normalization or feature standardization
under finite regularization — which is why we standardize and pin a
fixed C rather than sweeping preprocessing the way 5.1 had to.
"""
from typing import Callable, Dict

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# An estimator factory: () -> fresh unfitted sklearn estimator.
EstimatorFactory = Callable[[], object]


# ----- estimator factories (one per probe "view") -----

def make_logreg_probe(C: float = 1.0, max_iter: int = 2000) -> EstimatorFactory:
    """Factory for the Artifact 5.2 logistic-regression probe.

    Returns a zero-arg callable that builds a fresh
    `StandardScaler -> multinomial LogisticRegression` pipeline.

    Parameters
    ----------
    C : float
        Inverse L2 regularization strength. With ~1400 train rows in
        768 dims the problem is over-parameterized, so regularization
        is load-bearing; C=1.0 is the sklearn default and a sane,
        reproducible pin. Sweep it via the run script if needed.
    max_iter : int
        lbfgs iteration cap. 2000 is comfortably above what BERT-scale
        standardized features need to converge.

    Notes
    -----
    StandardScaler is inside the pipeline on purpose: under
    cross-validation it is fit on each training fold only, so the
    held-out fold never sees its own mean/variance (no leakage).
    """
    def factory() -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    # sklearn >=1.7 dropped the `multi_class` arg; lbfgs is
                    # multinomial (softmax) by default for multiclass y.
                    LogisticRegression(
                        C=C,
                        max_iter=max_iter,
                        solver="lbfgs",
                    ),
                ),
            ]
        )

    return factory


def make_lda_probe(shrinkage: str = "auto") -> EstimatorFactory:
    """Factory for the Artifact 5.3 Fisher / LDA probe.

    Provided here (rather than in a separate 5.3 module) to make the
    shared-harness contract concrete: 5.3's run script only needs to
    pass THIS factory into `evaluate_layer` — the evaluation loop,
    baselines, CSV schema, and plots are identical to 5.2.

    Uses the `lsqr` solver with automatic Ledoit-Wolf shrinkage, which
    is the well-conditioned analytical Fisher discriminant in the
    high-dim / low-sample regime (768 dims, ~70-114 rows per class).
    Plain LDA without shrinkage would invert a singular within-class
    scatter matrix here.
    """
    def factory() -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LinearDiscriminantAnalysis(
                        solver="lsqr", shrinkage=shrinkage
                    ),
                ),
            ]
        )

    return factory


# ----- baselines -----

def chance_accuracy(y: np.ndarray) -> float:
    """Uniform-random accuracy = 1 / n_classes.

    The floor a probe must clear to demonstrate ANY readable signal.
    """
    return 1.0 / len(np.unique(y))


def majority_accuracy(y: np.ndarray) -> float:
    """Accuracy of always predicting the most frequent class.

    A stricter, label-distribution-aware baseline than `chance`. For
    the roughly balanced 20 NG it sits just above 1/20, but reporting
    it makes the floor honest if a subsample is skewed.
    """
    counts = np.bincount(y)
    return float(counts.max() / len(y))


# ----- the shared evaluation loop -----

def evaluate_layer(
    X: np.ndarray,
    y: np.ndarray,
    estimator_factory: EstimatorFactory,
    *,
    cv: int = 5,
    seed: int = 0,
) -> Dict[str, float]:
    """Cross-validate one linear probe on one layer's representation.

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
        One layer's document-segment vectors (e.g. ``emb[:, L, :]``).
    y : np.ndarray, shape (N,)
        Integer topic labels.
    estimator_factory : EstimatorFactory
        Zero-arg callable returning a fresh estimator. `cross_validate`
        clones it for every fold, so the factory is consulted once and
        each fold trains an independent model.
    cv : int
        Number of stratified folds. 5 mirrors the 5-seed error-band
        protocol used in 5.1's figures.
    seed : int
        Fold-shuffling seed; fixed for reproducibility.

    Returns
    -------
    dict with keys:
        accuracy_mean, accuracy_std,
        macro_f1_mean, macro_f1_std,
        chance, majority

    Accuracy is the primary metric (the question is "probe accuracy(L)");
    macro-F1 is reported alongside so a per-class collapse on the
    roughly balanced 20 NG would still surface.
    """
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    scores = cross_validate(
        estimator_factory(),
        X,
        y,
        cv=splitter,
        scoring=("accuracy", "f1_macro"),
        n_jobs=-1,
    )
    return {
        "accuracy_mean": float(scores["test_accuracy"].mean()),
        "accuracy_std": float(scores["test_accuracy"].std()),
        "macro_f1_mean": float(scores["test_f1_macro"].mean()),
        "macro_f1_std": float(scores["test_f1_macro"].std()),
        "chance": chance_accuracy(y),
        "majority": majority_accuracy(y),
    }
