"""Tests for src/embed.py.

These tests actually load bert-base-uncased (~440MB on first run,
cached afterward at ~/.cache/huggingface/). They run on 4 tiny
docs each, so wall time after the model is loaded is small.

The `pretrained` and `random_init` fixtures are module-scoped so
each model is loaded ONCE per pytest invocation, not per test.
"""
import pytest
import torch

from src.embed import extract_layer_embeddings, load_bert, tokenize_batch

FAKE_DOCS = [
    "the cat sat on the mat",
    "deep learning revolutionizes natural language processing",
    "neural networks learn hierarchical representations",
    "transformers use self attention to mix information across positions",
]


@pytest.fixture(scope="module")
def pretrained():
    return load_bert(random_init=False)


@pytest.fixture(scope="module")
def random_init():
    return load_bert(random_init=True, seed=1)


def test_extract_shape_pretrained(pretrained):
    """Pretrained BERT-base forward on 4 docs → (4, 13, 768)."""
    model, tokenizer = pretrained
    enc = tokenize_batch(FAKE_DOCS, tokenizer)
    embeds = extract_layer_embeddings(
        model, enc["input_ids"], enc["attention_mask"]
    )
    assert embeds.shape == (4, 13, 768)


def test_extract_shape_random_init(random_init):
    """Random-init same-architecture BERT → same shape (4, 13, 768)."""
    model, tokenizer = random_init
    enc = tokenize_batch(FAKE_DOCS, tokenizer)
    embeds = extract_layer_embeddings(
        model, enc["input_ids"], enc["attention_mask"]
    )
    assert embeds.shape == (4, 13, 768)


def test_random_init_differs_from_pretrained(pretrained, random_init):
    """Two models on the same inputs must produce clearly different
    final-layer representations (so any layerwise trend we see is
    genuinely about the pretraining, not just shared architecture)."""
    pre_model, tokenizer = pretrained
    rand_model, _ = random_init
    enc = tokenize_batch(FAKE_DOCS, tokenizer)
    pre_emb = extract_layer_embeddings(
        pre_model, enc["input_ids"], enc["attention_mask"]
    )
    rand_emb = extract_layer_embeddings(
        rand_model, enc["input_ids"], enc["attention_mask"]
    )
    # Compare last layer; expect substantial divergence.
    assert not torch.allclose(pre_emb[:, -1], rand_emb[:, -1], atol=1e-2)


def test_mean_pool_ignores_padding(pretrained):
    """A doc's mean-pool must NOT depend on how much padding its
    batch-mates force onto it.

    We embed `target` alone, and embed it again alongside a long
    doc that forces padding on `target`. Both runs should give
    the same vector for `target` under mask-aware mean-pool.
    """
    model, tokenizer = pretrained
    target = "the cat sat on the mat"
    long_filler = " ".join(["word"] * 60)  # forces padding on target

    enc_alone = tokenize_batch([target], tokenizer)
    enc_mixed = tokenize_batch([target, long_filler], tokenizer)

    emb_alone = extract_layer_embeddings(
        model, enc_alone["input_ids"], enc_alone["attention_mask"]
    )
    emb_mixed = extract_layer_embeddings(
        model, enc_mixed["input_ids"], enc_mixed["attention_mask"]
    )

    assert torch.allclose(emb_alone[0], emb_mixed[0], atol=1e-5)


def test_layers_are_different(pretrained):
    """Layer 0 (embeddings) and layer 12 (final encoder) should not
    be bit-identical — sanity that the encoder stack is doing work."""
    model, tokenizer = pretrained
    enc = tokenize_batch(FAKE_DOCS, tokenizer)
    embeds = extract_layer_embeddings(
        model, enc["input_ids"], enc["attention_mask"]
    )
    assert not torch.allclose(embeds[:, 0], embeds[:, -1], atol=1e-2)


def test_embed_docs_shape(pretrained):
    """embed_docs on 8 docs with batch_size=4 must produce (8, 13, 768)
    AND must have iterated more than once (forces the batch loop)."""
    from src.embed import embed_docs

    model, tokenizer = pretrained
    # 8 docs = 2 batches at batch_size=4. Both batches must contribute.
    docs = FAKE_DOCS + [
        "convolutional networks process images via local kernels",
        "recurrent networks process sequences step by step",
        "attention mechanisms allow long range dependencies",
        "language models predict the next token in a sequence",
    ]
    out = embed_docs(
        docs, model, tokenizer, batch_size=4, show_progress=False
    )
    assert out.shape == (8, 13, 768)
    # numpy array, not torch tensor (downstream is sklearn)
    import numpy as np
    assert isinstance(out, np.ndarray)
