r"""Extract and cache BERT layer-wise document-segment embeddings.

Run from the repo root:

    .\.venv\Scripts\python.exe experiments\extract_embeddings.py --n_docs 2000 --batch_size 16

What this does:
    1. Load 20 Newsgroups, apply the canonical cleanup + short-doc filter.
    2. Deterministically subsample `n_docs` documents (seeded; the same
       subset is used for both the pretrained and random-init runs so
       all downstream comparisons see identical inputs).
    3. For each of two models — pretrained `bert-base-uncased` and a
       random-init BERT (seed=1) of the same architecture — run a
       cached forward pass:
           - cache miss: load model, forward + mean-pool all docs,
             save (embeddings, labels, indices) to outputs/cache/.
           - cache hit: skip the forward, load straight from disk.
    4. Print timing, cache paths, and shapes.

This is the single entry point for both W1 pilot (`--n_docs 2000`)
and W2 full extraction (`--n_docs` ≈ 18129); the only difference is
the CLI flag.
"""
import argparse
import sys
import time
from pathlib import Path

# Allow direct execution from the repo root:
#   python experiments/extract_embeddings.py ...
# In that mode Python puts `experiments/` on sys.path, not the repo root,
# so `import src...` would otherwise fail.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.cache import compute_or_load, embedding_cache_path
from src.data import filter_short_docs, load_20newsgroups
from src.embed import embed_docs, load_bert


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract and cache BERT layer-wise document embeddings."
    )
    p.add_argument(
        "--n_docs",
        type=int,
        default=2000,
        help="Number of docs to subsample (pilot=2000, full ~18129)",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Forward-pass batch size (lower if CPU OOM)",
    )
    p.add_argument(
        "--min_chars",
        type=int,
        default=30,
        help="Minimum doc length after cleanup",
    )
    p.add_argument(
        "--sample_seed",
        type=int,
        default=42,
        help="RNG seed for selecting the doc subset",
    )
    p.add_argument(
        "--random_init_seed",
        type=int,
        default=1,
        help="RNG seed for the random-init BERT control",
    )
    p.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="HuggingFace model identifier",
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="WordPiece token truncation length",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing cache files and recompute",
    )
    return p.parse_args()


def subsample_indices(n_total: int, n_docs: int, seed: int) -> np.ndarray:
    """Deterministically pick `n_docs` indices out of `n_total`.

    Uses numpy's modern RNG with a fixed seed so that the same call
    twice gives the same indices, and so that switching models (but
    not seed) selects the SAME docs for fair comparison.
    """
    if n_docs >= n_total:
        return np.arange(n_total)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=n_docs, replace=False))


def run_one_model(
    *,
    tag: str,
    random_init: bool,
    model_seed: int,
    model_name: str,
    docs: list,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    n_docs: int,
    sample_seed: int,
    max_length: int,
    min_chars: int,
    force: bool,
) -> None:
    """Compute (or load from cache) embeddings for one model variant."""
    path = embedding_cache_path(
        tag, model_name, n_docs, sample_seed, max_length, min_chars
    )

    def compute():
        print(f"  cache miss -> loading {tag} BERT and running forward...")
        model, tokenizer = load_bert(
            model_name=model_name, random_init=random_init, seed=model_seed
        )
        t0 = time.time()
        embeddings = embed_docs(
            docs, model, tokenizer,
            batch_size=batch_size, max_length=max_length,
        )
        print(f"  forward took {time.time() - t0:.1f} s")
        return {
            "embeddings": embeddings,
            "labels": labels,
            "indices": indices,
            # self-describing metadata (0-d arrays)
            "meta_model_name": np.array(model_name),
            "meta_random_init": np.array(random_init),
            "meta_model_seed": np.array(model_seed),
            "meta_n_docs": np.array(n_docs),
            "meta_sample_seed": np.array(sample_seed),
            "meta_max_length": np.array(max_length),
            "meta_min_chars": np.array(min_chars),
        }

    t0 = time.time()
    result = compute_or_load(path, compute, force=force)
    print(f"  done in {time.time() - t0:.1f} s total")
    print(f"  cache file: {path}")
    print(f"  embeddings shape: {result['embeddings'].shape}")
    print(f"  embeddings dtype: {result['embeddings'].dtype}")
    print(f"  labels shape:     {result['labels'].shape}")


def main() -> None:
    args = parse_args()

    print(f"[1/3] loading 20 Newsgroups (this may take a moment first time)...")
    data = load_20newsgroups(subset="all")
    data = filter_short_docs(data, min_chars=args.min_chars)
    print(f"      after filter: {len(data)} docs across {len(data.target_names)} topics")

    print(f"[2/3] subsampling {args.n_docs} docs (seed={args.sample_seed})...")
    indices = subsample_indices(len(data), args.n_docs, args.sample_seed)
    docs = [data.docs[i] for i in indices]
    labels = data.labels[indices]
    print(f"      first 5 indices: {indices[:5].tolist()}")
    print(f"      label distribution: {np.bincount(labels, minlength=20).tolist()}")

    print(f"[3/3] extracting embeddings...")
    common = dict(
        model_name=args.model_name,
        docs=docs,
        labels=labels,
        indices=indices,
        batch_size=args.batch_size,
        n_docs=args.n_docs,
        sample_seed=args.sample_seed,
        max_length=args.max_length,
        min_chars=args.min_chars,
        force=args.force,
    )
    print(f"  -- pretrained --")
    run_one_model(
        tag="pretrained", random_init=False, model_seed=0, **common
    )
    print(f"  -- random_init seed={args.random_init_seed} --")
    run_one_model(
        tag=f"random_seed{args.random_init_seed}",
        random_init=True,
        model_seed=args.random_init_seed,
        **common,
    )

    print("done.")


if __name__ == "__main__":
    main()
