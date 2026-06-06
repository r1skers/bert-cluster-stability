r"""Run clustering metrics over cached BERT embeddings, write CSV.

For each (model variant, layer L) and a single K (default 20),
compute:
    clarity:     silhouette, Davies-Bouldin, Calinski-Harabasz
    alignment:   NMI, purity vs 20 NG topic labels
    diagnostics: anisotropy, participation_ratio   (K-independent)

Inputs:  cached embeddings from `extract_embeddings.py`
Output:  outputs/tables/clustering/pilot_metrics.csv

Run from repo root:
    .\.venv\Scripts\python.exe experiments\clustering\run_pilot_metrics.py
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cache import embedding_cache_path
from src.diagnostics import anisotropy, participation_ratio
from src.metrics import cluster_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n_docs", type=int, default=2000)
    p.add_argument("--k", type=int, default=20,
                   help="KMeans cluster count (K=20 = 20 NG topic count)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0],
                   help="KMeans seeds; pass multiple for variance sensitivity, "
                        "e.g. --seeds 0 1 2 3 4")
    p.add_argument("--mode", type=str, default="l2",
                   choices=["l2", "centered", "raw"])
    p.add_argument("--sample_seed", type=int, default=42)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--min_chars", type=int, default=30)
    p.add_argument("--model_name", type=str, default="bert-base-uncased")
    p.add_argument("--random_init_seed", type=int, default=1)
    p.add_argument("--output", type=str, default="outputs/tables/clustering/pilot_metrics.csv")
    return p.parse_args()


def load_cached(tag: str, args: argparse.Namespace):
    """Return (embeddings, labels) from the cache file matching args."""
    path = embedding_cache_path(
        tag, args.model_name, args.n_docs,
        args.sample_seed, args.max_length, args.min_chars,
    )
    print(f"  loading {path}")
    with np.load(path) as d:
        return d["embeddings"], d["labels"]


def main() -> None:
    args = parse_args()

    print(f"[1/3] loading cached embeddings...")
    pretrained_emb, labels = load_cached("pretrained", args)
    random_emb, labels_check = load_cached(
        f"random_seed{args.random_init_seed}", args
    )
    assert np.array_equal(labels, labels_check), (
        "labels mismatch between pretrained and random-init caches "
        "— were they extracted with the same sample_seed?"
    )
    n_layers = pretrained_emb.shape[1]
    print(f"  shape: {pretrained_emb.shape} (N, layers, hidden)")

    print(
        f"[2/3] computing metrics: {n_layers} layers × 2 models, "
        f"K={args.k}, mode={args.mode}, seeds={args.seeds}"
    )

    rows = []
    for model_tag, embeds in [
        ("pretrained", pretrained_emb),
        (f"random_seed{args.random_init_seed}", random_emb),
    ]:
        for L in range(n_layers):
            X = embeds[:, L, :]  # (N, hidden_dim)

            # diagnostics are computed once per (model, layer):
            # they don't depend on KMeans seed or representation mode.
            # done on RAW X — anisotropy / PR are properties of the
            # original embedding space, not of any preprocessed version.
            ani = anisotropy(X, sample_size=2000, seed=0)
            pr = participation_ratio(X)

            for s in args.seeds:
                t0 = time.time()
                cm = cluster_metrics(
                    X, labels, k=args.k, seed=s, mode=args.mode
                )
                dt = time.time() - t0

                row = {
                    "model": model_tag,
                    "layer": L,
                    "k": args.k,
                    "mode": args.mode,
                    "seed": s,
                    **cm,
                    "anisotropy": ani,
                    "participation_ratio": pr,
                }
                rows.append(row)
                print(
                    f"  [{model_tag:>12s} L={L:2d} s={s}] "
                    f"sil={cm['silhouette']:+.3f}  "
                    f"nmi={cm['nmi']:.3f}  "
                    f"pur={cm['purity']:.3f}  "
                    f"ani={ani:+.3f}  "
                    f"pr={pr:6.1f}  "
                    f"({dt:.1f}s)"
                )

    print(f"[3/3] writing CSV: {args.output}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("done.")


if __name__ == "__main__":
    main()
