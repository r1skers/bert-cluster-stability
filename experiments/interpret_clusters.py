r"""Open the black box of KMeans on BERT layer embeddings.

For each requested layer, fit clustering with the requested K and produce three
human-readable artifacts:

    outputs/figures/interpret/cluster_topic_heatmap_K{k}_{transform}_{clusterer}_L{L}.png
        Row-normalized heatmap: rows = clusters (with size in label),
        cols = 20 NG topics, color = P(topic | cluster).

    outputs/interpret/cluster_summary_K{k}_{transform}_{clusterer}_L{L}.csv
        Per-cluster size, major topic, purity, top-3 topics.

    outputs/interpret/cluster_keywords_K{k}_{transform}_{clusterer}_L{L}.txt
        Top-N c-TF-IDF words per cluster.

Default `--layers 0 12` produces the L0 vs L12 contrast: at L0 the
heatmap should be blurry / mixed; at L12 it should show banded
rows where clusters concentrate on specific topics. That visual
contrast is what NMI(L) = 0.07 -> 0.36 looks like in pixels.

Run from repo root:
    .\.venv\Scripts\python.exe experiments\interpret_clusters.py
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cache import embedding_cache_path
from src.clusterers import cluster_labels
from src.data import filter_short_docs, load_20newsgroups
from src.interpret import (
    build_contingency,
    cluster_summary,
    top_keywords_per_cluster_ctfidf,
)
from src.transforms import apply_transform


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, nargs="+", default=[0, 12])
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clusterer", type=str, default="lloyd",
                   choices=[
                       "lloyd",
                       "spherical",
                       "agglo_cosine",
                       "agglo_ward",
                       "gmm_diag",
                       "gmm_full",
                   ])
    p.add_argument(
        "--transform",
        type=str,
        default="l2",
        help="Named transform from src.transforms, e.g. l2 or whiten100_l2",
    )
    p.add_argument("--n_docs", type=int, default=2000)
    p.add_argument("--sample_seed", type=int, default=42)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--min_chars", type=int, default=30)
    p.add_argument("--model_name", type=str, default="bert-base-uncased")
    p.add_argument("--top_keywords", type=int, default=8)
    p.add_argument("--outdir", type=str, default="outputs/interpret")
    p.add_argument("--figures_dir", type=str, default="outputs/figures/interpret")
    return p.parse_args()


def reload_docs_and_labels(args):
    """Re-derive (docs, labels, target_names) using the same params
    as `extract_embeddings.py`. We need the raw text strings for
    TF-IDF, which the cached npz does not store."""
    data = load_20newsgroups(subset="all")
    data = filter_short_docs(data, min_chars=args.min_chars)
    n_total = len(data)
    if args.n_docs >= n_total:
        idx = np.arange(n_total)
    else:
        rng = np.random.default_rng(args.sample_seed)
        idx = np.sort(rng.choice(n_total, size=args.n_docs, replace=False))
    docs = [data.docs[i] for i in idx]
    labels = data.labels[idx]
    return docs, labels, data.target_names


def load_cached_embeddings(args):
    path = embedding_cache_path(
        "pretrained", args.model_name, args.n_docs,
        args.sample_seed, args.max_length, args.min_chars,
    )
    with np.load(path) as d:
        return d["embeddings"], d["labels"]


def plot_heatmap(
    contingency: np.ndarray,
    target_names,
    layer: int,
    k_value: int,
    transform: str,
    clusterer: str,
    outpath: Path,
) -> None:
    """Row-normalized contingency heatmap: P(topic | cluster) per row."""
    row_sums = contingency.sum(axis=1, keepdims=True)
    norm = contingency / np.maximum(row_sums, 1)
    k = contingency.shape[0]
    sizes = contingency.sum(axis=1).astype(int)

    fig_height = max(5.0, min(18.0, 0.28 * k + 2.5))
    fig, ax = plt.subplots(figsize=(11, fig_height))
    im = ax.imshow(norm, aspect="auto", cmap="viridis", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(target_names)))
    ax.set_xticklabels(target_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(k))
    ax.set_yticklabels(
        [f"c{c} (n={s})" for c, s in zip(range(k), sizes)], fontsize=8
    )
    ax.set_xlabel("20 Newsgroups topic")
    ax.set_ylabel("KMeans cluster (with size)")
    ax.set_title(
        f"Cluster-Topic alignment heatmap, Layer {layer}, K={k_value}, "
        f"{transform} + {clusterer}  "
        f"(row-normalized: P(topic | cluster))"
    )
    plt.colorbar(im, ax=ax, label="P(topic | cluster)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath}")


def write_summary_csv(rows, outpath: Path) -> None:
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {outpath}")


def write_keywords_txt(keywords, summary_rows, layer: int, outpath: Path) -> None:
    """Plain-text file: one cluster per block, sorted by size desc.

    Each block:
        cluster 7  (n=132)  major: sci.space (78%)
          top words: space, nasa, orbit, launch, moon, ...
    """
    sizes = {r["cluster_id"]: r["size"] for r in summary_rows}
    major = {r["cluster_id"]: r["major_topic"] for r in summary_rows}
    major_frac = {r["cluster_id"]: r["major_topic_fraction"] for r in summary_rows}

    order = sorted(keywords.keys(), key=lambda c: -sizes.get(c, 0))

    with open(outpath, "w", encoding="utf-8") as f:
        f.write(f"# Cluster keywords, Layer {layer}\n")
        f.write(f"# Sorted by cluster size (largest first)\n\n")
        for c in order:
            words = keywords[c]
            if not words:
                continue
            size = sizes.get(c, 0)
            mt = major.get(c, "?")
            mf = major_frac.get(c, 0.0)
            f.write(f"cluster {c:>2d}  (n={size:4d})  major: {mt} ({mf*100:.0f}%)\n")
            f.write(f"  top words: {', '.join(words)}\n\n")
    print(f"  wrote {outpath}")


def analyze_layer(
    layer: int,
    embeddings: np.ndarray,
    labels: np.ndarray,
    docs,
    target_names,
    args: argparse.Namespace,
    outdir: Path,
    figures_dir: Path,
) -> None:
    print(f"[layer {layer}]")
    X = embeddings[:, layer, :]
    X_prep = apply_transform(X, args.transform)
    pred = cluster_labels(X_prep, k=args.k, seed=args.seed, clusterer=args.clusterer)

    contingency = build_contingency(pred, labels, args.k, len(target_names))
    summary = cluster_summary(pred, labels, target_names)
    keywords = top_keywords_per_cluster_ctfidf(
        docs, pred, k=args.k, top_n=args.top_keywords
    )

    plot_heatmap(
        contingency, target_names, layer, args.k, args.transform, args.clusterer,
        figures_dir / (
            f"cluster_topic_heatmap_K{args.k}_{args.transform}_"
            f"{args.clusterer}_L{layer}.png"
        ),
    )
    write_summary_csv(
        summary,
        outdir / (
            f"cluster_summary_K{args.k}_{args.transform}_"
            f"{args.clusterer}_L{layer}.csv"
        ),
    )
    write_keywords_txt(
        keywords,
        summary,
        layer,
        outdir / (
            f"cluster_keywords_K{args.k}_{args.transform}_"
            f"{args.clusterer}_L{layer}.txt"
        ),
    )


def main() -> None:
    args = parse_args()

    print("[1/3] reloading 20 NG docs + labels...")
    docs, labels, target_names = reload_docs_and_labels(args)
    print(f"      {len(docs)} docs, {len(target_names)} topics")

    print("[2/3] loading cached pretrained embeddings...")
    embeddings, cached_labels = load_cached_embeddings(args)
    assert np.array_equal(cached_labels, labels), (
        "Cached labels do not match re-derived labels — "
        "did sample_seed / min_chars / n_docs change between runs?"
    )
    print(f"      embeddings shape: {embeddings.shape}")

    outdir = Path(args.outdir)
    figures_dir = Path(args.figures_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"[3/3] analyzing {len(args.layers)} layers: {args.layers}")
    for L in args.layers:
        analyze_layer(
            L, embeddings, labels, docs, target_names,
            args, outdir, figures_dir,
        )

    print("done.")


if __name__ == "__main__":
    main()
