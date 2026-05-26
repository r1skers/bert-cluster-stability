"""BERT embedding extraction (single-batch primitives).

This module exposes three low-level functions:

    1. `load_bert(model_name, random_init, seed)`
       Returns (model, tokenizer). Model is in eval mode.

    2. `tokenize_batch(docs, tokenizer, max_length=512)`
       Returns dict of input_ids and attention_mask tensors.

    3. `extract_layer_embeddings(model, input_ids, attention_mask)`
       One forward pass with output_hidden_states=True;
       mean-pools each layer's hidden states over non-padding
       token positions; returns shape (B, 13, H).

Whole-corpus batched extraction + npz caching lives in a later
step (Step 3b); this file only handles ONE batch at a time so
each primitive is easy to test in isolation.
"""
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BertConfig, BertModel, PreTrainedTokenizerBase


def load_bert(
    model_name: str = "bert-base-uncased",
    random_init: bool = False,  # 加入随机初始化的选项，作为实验对照
    seed: int = 1,
) -> Tuple[BertModel, PreTrainedTokenizerBase]:
    """Load BERT + tokenizer.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    random_init : bool, default False
        If False, load pretrained weights.
        If True, instantiate the SAME architecture (from the
        pretrained config) but with random weights. This is the
        experimental control: it disentangles "trends caused by
        deep-stacking alone" from "trends caused by pretraining".
    seed : int
        torch random seed, used only when random_init=True.

    Notes
    -----
    The tokenizer is ALWAYS loaded from the pretrained checkpoint,
    even for random-init BERT. Random-init BERT has no meaningful
    vocabulary of its own — we just need both models to see the
    same input_ids for a given doc so the comparison is fair.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if random_init:
        config = BertConfig.from_pretrained(model_name)
        torch.manual_seed(seed)
        model = BertModel(config)
    else:
        model = BertModel.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def tokenize_batch(
    docs: List[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
) -> Dict[str, torch.Tensor]:
    """Tokenize a list of docs into padded tensors.

    Returns
    -------
    dict
        Contains `input_ids` and `attention_mask`, both of shape
        (B, L) where L is min(max_length, longest doc's token
        count after the [CLS] + tokens + [SEP] wrap).
    """
    return tokenizer(
        docs,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


@torch.no_grad()
def extract_layer_embeddings(
    model: BertModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Forward one batch and mean-pool each layer's hidden states.

    Parameters
    ----------
    model : BertModel
        BERT model already in eval mode (load_bert handles this).
    input_ids : torch.Tensor of shape (B, L)
    attention_mask : torch.Tensor of shape (B, L)
        1 for real tokens, 0 for padding.

    Returns
    -------
    torch.Tensor of shape (B, 13, H)
        Per-layer mean-pooled segment vectors.
        Index 0 = the embedding layer (token + position + segment
        embeddings, before any encoder block).
        Indices 1..12 = outputs of encoder blocks 1..12.

    Notes
    -----
    `output_hidden_states=True` makes BERT return a tuple of 13
    tensors of shape (B, L, H) — one per layer including the
    embedding layer. We stack along a new axis to get (B, 13, L, H),
    apply the attention mask to zero out padded positions, sum
    along the L axis, divide by the per-doc real-token count.

    The `.clamp(min=1)` on counts is purely defensive: after our
    `filter_short_docs` step, every doc has at least ~30 chars
    so counts are always positive in practice.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )

    # tuple of 13 (B, L, H) tensors -> stack to (B, 13, L, H)
    hidden_states = torch.stack(outputs.hidden_states, dim=1)

    # mask: (B, 1, L, 1), broadcastable over layers and hidden dim
    mask = attention_mask.unsqueeze(1).unsqueeze(-1).float()

    summed = (hidden_states * mask).sum(dim=2)            # (B, 13, H)
    counts = mask.sum(dim=2).clamp(min=1)                  # (B, 13, 1)
    return summed / counts                                  # (B, 13, H)


def embed_docs(
    docs: List[str],
    model: BertModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 16,
    max_length: int = 512,
    show_progress: bool = True,
) -> np.ndarray:
    """Embed an entire corpus through `model` in batches.

    Parameters
    ----------
    docs : list of str
        All documents to embed.
    model, tokenizer
        From `load_bert`.
    batch_size : int
        How many docs per forward pass. 16 is reasonable for
        BERT-base on a 5700X CPU; lower if you hit OOM.
    max_length : int
        Token truncation length.
    show_progress : bool
        Whether to display a tqdm progress bar.

    Returns
    -------
    np.ndarray of shape (N, 13, H)
        Per-doc, per-layer mean-pooled segment vectors.

    Notes
    -----
    We accumulate per-batch results as numpy (after `.cpu().numpy()`)
    rather than as torch tensors. This avoids holding the autograd
    graph or GPU memory for things we'll never backprop through,
    and matches the downstream sklearn / npz interface.
    """
    chunks: List[np.ndarray] = []
    n_batches = (len(docs) + batch_size - 1) // batch_size
    iterator = range(0, len(docs), batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=n_batches, desc="embedding")

    for start in iterator:
        batch = docs[start : start + batch_size]
        enc = tokenize_batch(batch, tokenizer, max_length=max_length)
        emb = extract_layer_embeddings(
            model, enc["input_ids"], enc["attention_mask"]
        )
        chunks.append(emb.cpu().numpy())

    return np.concatenate(chunks, axis=0)
