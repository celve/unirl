"""Unit tests for :class:`PEBundle` — the composed weights container.

PEBundle is a pure container: two ``Bundle``-typed fields, no construction
logic. These tests verify field assignment and Bundle Protocol
satisfaction. No real model weights, no GPU.
"""

from __future__ import annotations

from diffusionrl.models.pe import PEBundle
from diffusionrl.models.types.bundle import Bundle


class _StubBundle:
    """Minimal stand-in for any Bundle (the Protocol is empty)."""

    def __init__(self, tag: str) -> None:
        self.tag = tag


def test_pe_bundle_holds_two_children() -> None:
    diffusion = _StubBundle("diffusion")
    llm = _StubBundle("llm")

    pe_bundle = PEBundle(diffusion=diffusion, llm=llm)

    assert pe_bundle.diffusion is diffusion
    assert pe_bundle.llm is llm


def test_pe_bundle_satisfies_bundle_protocol() -> None:
    """Bundle Protocol is empty (``@runtime_checkable Protocol`` with no
    members), so isinstance is trivially True. The check exists to catch
    a future tightening of the Protocol that would unintentionally
    exclude PEBundle."""
    pe_bundle = PEBundle(diffusion=_StubBundle("d"), llm=_StubBundle("l"))

    assert isinstance(pe_bundle, Bundle)


def test_pe_bundle_kwargs_only() -> None:
    """``PEBundle`` enforces keyword-only construction for clarity at
    call sites — positional args would let diffusion / llm be silently
    swapped."""
    import pytest

    with pytest.raises(TypeError):
        PEBundle(_StubBundle("d"), _StubBundle("l"))  # type: ignore[misc]
