r"""Run subset-resampling stability for the narrowed recipes.

This is Step A in the project plan. It compares:

    baseline: layer12 + l2 + Lloyd KMeans + K=20
    best:     layer12 + whiten100_l2 + spherical KMeans + K=20

Protocol:
    80% subset x B runs -> fit recipe on subset -> predict all docs
    -> pairwise ARI across the B full-label partitions.

Run from repo root:
    .\.venv\Scripts\python.exe experiments\run_stability.py

Quick smoke test:
    .\.venv\Scripts\python.exe experiments\run_stability.py --n_runs 5 --models pretrained
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cache import embedding_cache_path
from src.metrics import nmi, purity
from src.pooling import apply_pooling
from src.stability import resampling_stability


RECIPES = {
    "baseline": {"transform": "l2", "clusterer": "lloyd"},
    "best": {"transform": "whiten100_l2", "clusterer": "spherical"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n_docs", type=int, default=2000)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--n_runs", type=int, default=50)
    p.add_argument("--subset_fraction", type=float, default=0.8)
    p.add_argument("--subset_seed", type=int, default=0)
    p.add_argument("--cluster_seed", type=int, default=0)
    p.add_argument("--pooling", default="layer12")
    p.add_argument(
        "--recipes",
        nargs="+",
        default=["baseline", "best"],
        choices=sorted(RECIPES),
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=["pretrained", "random"],
        choices=["pretrained", "random"],
    )
    p.add_argument(
        "--transform_fit_scope",
        default="subset",
        choices=["subset", "full"],
        help="subset is stricter; full tests clustering stability after a fixed transform",
    )
    p.add_argument("--sample_seed", type=int, default=42)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--min_chars", type=int, default=30)
    p.add_argument("--model_name", type=str, default="bert-base-uncased")
    p.add_argument("--random_init_seed", type=int, default=1)
    p.add_argument("--output", type=str, default="outputs/tables/stability_pilot.csv")
    p.add_argument("--labels_output", type=str, default="outputs/tables/stability_label_runs.npz")
    return p.parse_args()


def model_tag(name: str, random_init_seed: int) -> str:
    if name == "pretrained":
        return "pretrained"
    if name == "random":
        return f"random_seed{random_init_seed}"
    raise ValueError(f"unknown model name: {name}")


def load_cached(tag: str, args: argparse.Namespace):
    path = embedding_cache_path(
        tag,
        args.model_name,
        args.n_docs,
        args.sample_seed,
        args.max_length,
        args.min_chars,
    )
    print(f"loading {path}")
    with np.load(path) as d:
        return d["embeddings"], d["labels"]


def label_alignment_stats(true_labels: np.ndarray, label_runs: np.ndarray) -> dict[str, float]:
    nmis = np.array([nmi(true_labels, pred) for pred in label_runs], dtype=np.float64)
    purities = np.array([purity(true_labels, pred) for pred in label_runs], dtype=np.float64)
    return {
        "run_nmi_mean": float(nmis.mean()),
        "run_nmi_std": float(nmis.std(ddof=0)),
        "run_nmi_min": float(nmis.min()),
        "run_nmi_max": float(nmis.max()),
        "run_purity_mean": float(purities.mean()),
        "run_purity_std": float(purities.std(ddof=0)),
        "run_purity_min": float(purities.min()),
        "run_purity_max": float(purities.max()),
    }


def main() -> None:
    args = parse_args()

    rows = []
    label_outputs = {}
    reference_labels = None

    for model in args.models:
        tag = model_tag(model, args.random_init_seed)
        embeddings, labels = load_cached(tag, args)
        if reference_labels is None:
            reference_labels = labels
            label_outputs["true_labels"] = labels
        else:
            assert np.array_equal(reference_labels, labels), "label mismatch"

        X = apply_pooling(embeddings, args.pooling)
        for recipe_name in args.recipes:
            recipe = RECIPES[recipe_name]
            print(
                f"[{tag} | {recipe_name}] "
                f"{recipe['transform']} + {recipe['clusterer']} "
                f"K={args.k}, B={args.n_runs}"
            )
            t0 = time.time()
            summary, label_runs = resampling_stability(
                X,
                k=args.k,
                transform=recipe["transform"],
                clusterer=recipe["clusterer"],
                subset_fraction=args.subset_fraction,
                n_runs=args.n_runs,
                subset_seed=args.subset_seed,
                cluster_seed=args.cluster_seed,
                transform_fit_scope=args.transform_fit_scope,
            )
            elapsed = time.time() - t0
            align = label_alignment_stats(labels, label_runs)
            row = {
                "model": tag,
                "pooling": args.pooling,
                "recipe": recipe_name,
                "transform": recipe["transform"],
                "clusterer": recipe["clusterer"],
                "k": args.k,
                "n_runs": args.n_runs,
                "subset_fraction": args.subset_fraction,
                "subset_seed": args.subset_seed,
                "cluster_seed": args.cluster_seed,
                "transform_fit_scope": args.transform_fit_scope,
                "mean_ari": summary.mean_ari,
                "std_ari": summary.std_ari,
                "min_ari": summary.min_ari,
                "max_ari": summary.max_ari,
                "n_pairs": summary.n_pairs,
                "elapsed_s": elapsed,
                **align,
            }
            rows.append(row)
            label_outputs[f"{tag}_{recipe_name}"] = label_runs
            print(
                f"  stability ARI={summary.mean_ari:.3f} +/- {summary.std_ari:.3f}; "
                f"NMI={align['run_nmi_mean']:.3f} +/- {align['run_nmi_std']:.3f}; "
                f"purity={align['run_purity_mean']:.3f} +/- {align['run_purity_std']:.3f}; "
                f"{elapsed:.1f}s"
            )

    print(f"writing {args.output}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"writing {args.labels_output}")
    np.savez(args.labels_output, **label_outputs)
    print("done.")


if __name__ == "__main__":
    main()
