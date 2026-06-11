"""Unit tests for the VeOmni word-embedding fully_shard selection helper.

``_embedding_block_classes`` decides which leaf modules get their OWN
``fully_shard`` group so a direct ``wte(input_ids)`` lookup all-gathers the
weight to a plain tensor regardless of caller (the HI3 mixed-DTensor fix). It is
a pure function — no dist, no ``fully_shard``, no ``parallel_state`` — so it
tests on CPU/meta with no GPU and no process group.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from unirl.train.backend.veomni.wrap import _embedding_block_classes  # noqa: E402


class _FakeBlock(nn.Module):
    """Stand-in for a decoder layer (a non-embedding wrap target)."""

    def __init__(self) -> None:
        super().__init__()
        self.qkv_proj = nn.Linear(4, 4)


class _HI3Like(nn.Module):
    """``transformer.model`` shape: a ``wte`` embedding + decoder blocks."""

    def __init__(self) -> None:
        super().__init__()
        self.wte = nn.Embedding(8, 4)
        self.layers = nn.ModuleList([_FakeBlock(), _FakeBlock()])


class _QwenImageLike(nn.Module):
    """Diffusion DiT shape: an ``nn.Embedding`` NOT named wte/embed_tokens."""

    def __init__(self) -> None:
        super().__init__()
        self.addition_t_embedding = nn.Embedding(2, 4)
        self.blocks = nn.ModuleList([_FakeBlock()])


def test_hi3_wte_seeds_embedding_class() -> None:
    with torch.device("meta"):
        model = _HI3Like()
    assert _embedding_block_classes(model, tie_word_embeddings=False) == frozenset({"Embedding"})


def test_qwenimage_embedding_is_noop() -> None:
    # addition_t_embedding is NOT in {wte, embed_tokens} -> empty -> basic_modules unchanged,
    # so the validated QwenImage checksum parity is preserved.
    with torch.device("meta"):
        model = _QwenImageLike()
    assert _embedding_block_classes(model, tie_word_embeddings=False) == frozenset()


def test_tie_true_suppresses_union() -> None:
    # Tied embed/lm_head share storage and must stay in one group.
    with torch.device("meta"):
        model = _HI3Like()
    assert _embedding_block_classes(model, tie_word_embeddings=True) == frozenset()


def test_embed_tokens_leaf_also_matches() -> None:
    # qwen3 / qwen_vl naming.
    class _EmbedTokensLike(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(8, 4)

    with torch.device("meta"):
        model = _EmbedTokensLike()
    assert _embedding_block_classes(model, tie_word_embeddings=False) == frozenset({"Embedding"})


def test_non_parametric_named_wte_skipped() -> None:
    # The hasattr(weight) guard: a module named wte with no .weight is not a target.
    class _Decoy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.wte = nn.Identity()

    model = _Decoy()
    assert _embedding_block_classes(model, tie_word_embeddings=False) == frozenset()
