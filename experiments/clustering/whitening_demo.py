r"""Synthetic demo for why PCA whitening can help clustering.

This is not a BERT experiment. It is a small geometric sanity check for
the note's explanation:

    when a representation is dominated by a common / high-variance
    direction unrelated to labels, KMeans can produce a stable but
    semantically wrong partition. Whitening down-weights that dominant
    direction and can recover lower-energy cluster structure.

Run from repo root:
    .\.venv\Scripts\python.exe experiments\clustering\whitening_demo.py
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diagnostics import anisotropy, participation_ratio
from src.metrics import l2_normalize
from src.transforms import pca_whiten


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--n_per_cluster", type=int, default=180)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--nuisance_mean", type=float, default=20.0)
    p.add_argument("--nuisance_scale", type=float, default=10.0)
    p.add_argument("--signal_scale", type=float, default=2.0)
    p.add_argument("--cluster_noise", type=float, default=0.45)
    p.add_argument("--extra_noise_dims", type=int, default=7)
    p.add_argument("--extra_noise_scale", type=float, default=0.35)
    p.add_argument("--outfig", default="outputs/figures/transforms/whitening_demo.png")
    p.add_argument("--outcsv", default="outputs/tables/clustering/whitening_demo.csv")
    return p.parse_args()


def make_anisotropic_mixture(args):
    """Return X, y where labels live in low-energy signal directions."""
    rng = np.random.default_rng(args.seed)
    signal_centers = np.array(
        [
            [0.0, args.signal_scale, 0.0],
            [0.0, -args.signal_scale, 0.0],
            [0.0, 0.0, args.signal_scale],
        ]
    )

    chunks = []
    labels = []
    for label, center in enumerate(signal_centers):
        base = center + rng.normal(
            scale=args.cluster_noise,
            size=(args.n_per_cluster, 3),
        )
        # This coordinate is deliberately unrelated to the true labels.
        # Its variance dominates Euclidean clustering before whitening.
        base[:, 0] = args.nuisance_mean + rng.normal(
            scale=args.nuisance_scale,
            size=args.n_per_cluster,
        )
        extra = rng.normal(
            scale=args.extra_noise_scale,
            size=(args.n_per_cluster, args.extra_noise_dims),
        )
        chunks.append(np.hstack([base, extra]))
        labels.extend([label] * args.n_per_cluster)

    return np.vstack(chunks), np.asarray(labels, dtype=np.int64)


def kmeans_labels(X, k, seed):
    model = KMeans(
        n_clusters=k,
        n_init=20,
        algorithm="lloyd",
        random_state=seed,
    )
    return model.fit_predict(X)


def scores(true_labels, pred_labels):
    return {
        "ari": float(adjusted_rand_score(true_labels, pred_labels)),
        "nmi": float(normalized_mutual_info_score(true_labels, pred_labels)),
    }


def write_rows(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def scatter_panel(ax, X_signal, labels, title):
    ax.scatter(
        X_signal[:, 0],
        X_signal[:, 1],
        c=labels,
        s=16,
        cmap="tab10",
        alpha=0.78,
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xlabel("signal dim 1")
    ax.set_ylabel("signal dim 2")
    ax.grid(alpha=0.2)


def main():
    args = parse_args()
    X, y = make_anisotropic_mixture(args)

    X_l2 = l2_normalize(X)
    X_whiten = pca_whiten(X, n_components=X.shape[1], normalize=True)

    pred_l2 = kmeans_labels(X_l2, args.k, args.seed)
    pred_whiten = kmeans_labels(X_whiten, args.k, args.seed)

    l2_scores = scores(y, pred_l2)
    whiten_scores = scores(y, pred_whiten)
    rows = [
        {
            "space": "l2",
            **l2_scores,
            "anisotropy": anisotropy(X_l2, sample_size=len(X_l2), seed=args.seed),
            "participation_ratio": participation_ratio(X_l2),
        },
        {
            "space": "whiten_l2",
            **whiten_scores,
            "anisotropy": anisotropy(X_whiten, sample_size=len(X_whiten), seed=args.seed),
            "participation_ratio": participation_ratio(X_whiten),
        },
    ]
    write_rows(args.outcsv, rows)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True, sharey=True)
    X_signal = X[:, 1:3]
    scatter_panel(
        axes[0],
        X_signal,
        y,
        "True labels in the low-energy signal plane",
    )
    scatter_panel(
        axes[1],
        X_signal,
        pred_l2,
        f"L2 + Lloyd\nARI={l2_scores['ari']:.3f}, NMI={l2_scores['nmi']:.3f}",
    )
    scatter_panel(
        axes[2],
        X_signal,
        pred_whiten,
        f"PCA whitening + L2 + Lloyd\nARI={whiten_scores['ari']:.3f}, NMI={whiten_scores['nmi']:.3f}",
    )
    fig.suptitle(
        "Synthetic anisotropy demo: whitening can reveal low-energy cluster structure"
    )
    fig.tight_layout()

    outfig = Path(args.outfig)
    outfig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfig, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {args.outcsv}")
    print(f"wrote {args.outfig}")
    for row in rows:
        print(
            f"{row['space']:>10s}: "
            f"ARI={row['ari']:.3f}, NMI={row['nmi']:.3f}, "
            f"anisotropy={row['anisotropy']:.3f}, "
            f"PR={row['participation_ratio']:.1f}"
        )


if __name__ == "__main__":
    main()
