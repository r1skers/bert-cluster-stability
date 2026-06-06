r"""Artifact 5.2: layer-wise logistic-regression probe over cached embeddings.

For each (model variant, layer L), cross-validate a multinomial
logistic-regression probe predicting the 20 NG topic from layer L's
document-segment representation, and record mean +/- std accuracy /
macro-F1 plus the chance and majority baselines.

Reuses the SAME cached embeddings produced for 5.1
(`extract_embeddings.py`); no BERT forward pass happens here.

Output:  outputs/tables/probe/linear_probe.csv

Run from repo root:
    .\.venv\Scripts\python.exe experiments\probe\run_linear_probe.py
    .\.venv\Scripts\python.exe experiments\probe\run_linear_probe.py --probe lda   # 5.3 reuse
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
from src.probe import evaluate_layer, make_lda_probe, make_logreg_probe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n_docs", type=int, default=2000)
    p.add_argument(
        "--probe", choices=["logreg", "lda"], default="logreg",
        help="logreg = Artifact 5.2; lda = Artifact 5.3 (shared harness)",
    )
    p.add_argument("--C", type=float, default=1.0,
                   help="logreg inverse-regularization strength")
    p.add_argument("--cv", type=int, default=5, help="stratified CV folds")
    p.add_argument("--cv_seed", type=int, default=0)
    p.add_argument("--layers", type=int, nargs="+", default=list(range(13)))
    p.add_argument("--sample_seed", type=int, default=42)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--min_chars", type=int, default=30)
    p.add_argument("--model_name", type=str, default="bert-base-uncased")
    p.add_argument("--random_init_seed", type=int, default=1)
    p.add_argument("--output", type=str, default="outputs/tables/probe/linear_probe.csv")
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

    print("[1/3] loading cached embeddings...")
    pretrained_emb, labels = load_cached("pretrained", args)
    random_emb, labels_check = load_cached(
        f"random_seed{args.random_init_seed}", args
    )
    assert np.array_equal(labels, labels_check), (
        "labels mismatch between pretrained and random-init caches "
        "— were they extracted with the same sample_seed?"
    )
    print(f"  shape: {pretrained_emb.shape} (N, layers, hidden)")

    if args.probe == "logreg":
        factory = make_logreg_probe(C=args.C)
    else:
        factory = make_lda_probe()

    print(
        f"[2/3] probing: {len(args.layers)} layers x 2 models, "
        f"probe={args.probe}, cv={args.cv}"
    )

    rows = []
    for model_tag, embeds in [
        ("pretrained", pretrained_emb),
        (f"random_seed{args.random_init_seed}", random_emb),
    ]:
        for L in args.layers:
            X = embeds[:, L, :]  # (N, hidden_dim)
            t0 = time.time()
            res = evaluate_layer(
                X, labels, factory, cv=args.cv, seed=args.cv_seed
            )
            dt = time.time() - t0
            row = {"model": model_tag, "layer": L, "probe": args.probe, **res}
            rows.append(row)
            print(
                f"  [{model_tag:>12s} L={L:2d}] "
                f"acc={res['accuracy_mean']:.3f}+/-{res['accuracy_std']:.3f}  "
                f"f1={res['macro_f1_mean']:.3f}  "
                f"(chance={res['chance']:.3f})  "
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
