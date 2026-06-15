"""Unit tests for the VeOmni root-leaf fully_shard selection helper.

``_root_leaf_modules`` decides which direct-child leaf modules of the wrapped
root get their OWN ``fully_shard`` group (so the outer wrapper can call them —
HI3's ``wte`` / ``ln_f`` — outside the decoder's managed forward without meeting
a root-sharded DTensor). It is a pure function — no dist, no ``fully_shard`` —
so it tests on CPU/meta with no GPU and no process group.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from unirl.train.backend.veomni.wrap import _root_leaf_modules  # noqa: E402


def _names(model: nn.Module, tie: bool) -> set:
    return {name for name, _ in _root_leaf_modules(model, tie)}


class _FakeBlock(nn.Module):
    """Stand-in for a decoder layer (lives inside the ModuleList container)."""

    def __init__(self) -> None:
        super().__init__()
        self.qkv_proj = nn.Linear(4, 4)


class _HI3Like(nn.Module):
    """``transformer.model`` shape: wte embedding + decoder blocks + final norm."""

    def __init__(self) -> None:
        super().__init__()
        self.wte = nn.Embedding(8, 4)
        self.layers = nn.ModuleList([_FakeBlock(), _FakeBlock()])  # container: no own params
        self.ln_f = nn.LayerNorm(4)


def test_hi3like_selects_wte_and_ln_f() -> None:
    with torch.device("meta"):
        model = _HI3Like()
    # wte + ln_f carry their own params; the decoder ModuleList does not.
    assert _names(model, tie=False) == {"wte", "ln_f"}


def test_tie_excludes_embedding_keeps_norm() -> None:
    with torch.device("meta"):
        model = _HI3Like()
    # Tied embed/lm_head must stay in one group; the final norm is unaffected.
    assert _names(model, tie=True) == {"ln_f"}


def test_container_only_module_selects_nothing() -> None:
    class _ContainerOnly(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([_FakeBlock()])  # params live in the block

    with torch.device("meta"):
        model = _ContainerOnly()
    assert _names(model, tie=False) == set()


def test_embed_tokens_naming_excluded_when_tied() -> None:
    class _EmbedTokensLike(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(8, 4)
            self.norm = nn.LayerNorm(4)

    with torch.device("meta"):
        model = _EmbedTokensLike()
    assert _names(model, tie=False) == {"embed_tokens", "norm"}
    assert _names(model, tie=True) == {"norm"}
