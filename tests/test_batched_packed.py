"""Tests for ``packed_field`` + framework-managed ``cu_seqlens`` on ``Batched``.

Synthetic dataclasses; no real model dependency. Tests the public surface:

* ``Cls.pack(...)`` for constructing instances with per-sample tensor lists.
* Hidden ``_packed_cu_seqlens`` propagated through ``concat`` / ``slice`` /
  ``select`` / ``clone`` / ``to_device``.
* ``cu_seqlens`` and ``lengths`` read-only properties on the instance.
* ``batch_size`` falls back to cu_seqlens-derived count when no concat field
  is populated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest
import torch

from diffusionrl.utils.batched import (
    Batched,
    FieldKind,
    concat_field,
    packed_field,
    shared_field,
)

# ---------------------------------------------------------------------------
# Synthetic dataclasses
# ---------------------------------------------------------------------------


@dataclass
class _PackedOnly(Batched):
    """Minimal: one packed field, no concat neighbor."""

    tokens: Optional[torch.Tensor] = packed_field(default=None)


@dataclass
class _TwoPacked(Batched):
    """Two packed fields sharing the framework-managed cu_seqlens."""

    tokens: Optional[torch.Tensor] = packed_field(default=None)
    log_probs: Optional[torch.Tensor] = packed_field(default=None)


@dataclass
class _MixedFields(Batched):
    """Packed field alongside a regular concat field and a shared field."""

    sample_indices: Optional[torch.Tensor] = concat_field(default=None)
    tokens: Optional[torch.Tensor] = packed_field(default=None)
    schedule: Optional[torch.Tensor] = shared_field(default=None)


# ---------------------------------------------------------------------------
# Cls.pack(...) — user-facing constructor
# ---------------------------------------------------------------------------


def test_pack_from_per_sample_lists() -> None:
    seg = _PackedOnly.pack(
        tokens=[
            torch.tensor([1, 2]),
            torch.tensor([3, 4, 5]),
            torch.tensor([6, 7, 8, 9]),
        ]
    )
    assert seg.tokens is not None
    assert seg.tokens.tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert seg.tokens.shape == (9,)
    assert seg.cu_seqlens is not None
    assert seg.cu_seqlens.tolist() == [0, 2, 5, 9]


def test_pack_two_packed_fields_must_match_sizes() -> None:
    with pytest.raises(ValueError, match="don't match earlier packed-field sizes"):
        _TwoPacked.pack(
            tokens=[torch.tensor([1, 2]), torch.tensor([3, 4, 5])],
            log_probs=[torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
        )


def test_pack_with_concat_field_neighbor() -> None:
    seg = _MixedFields.pack(
        sample_indices=torch.arange(3),
        tokens=[
            torch.tensor([1, 2]),
            torch.tensor([3, 4, 5]),
            torch.tensor([6, 7, 8, 9]),
        ],
        schedule=torch.linspace(0.0, 1.0, 5),
    )
    assert seg.sample_indices.tolist() == [0, 1, 2]
    assert seg.tokens.tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert seg.cu_seqlens.tolist() == [0, 2, 5, 9]
    assert seg.schedule.shape == (5,)


def test_pack_with_none_packed_fields() -> None:
    seg = _TwoPacked.pack(tokens=None, log_probs=None)
    assert seg.tokens is None
    assert seg.log_probs is None
    assert seg.cu_seqlens is None
    assert seg.lengths is None


def test_pack_rejects_already_packed_tensor() -> None:
    with pytest.raises(TypeError, match="not an already-packed Tensor"):
        _PackedOnly.pack(tokens=torch.tensor([1, 2, 3, 4, 5]))


def test_pack_rejects_non_sequence() -> None:
    with pytest.raises(TypeError, match="expects a Sequence"):
        _PackedOnly.pack(tokens=42)  # type: ignore[arg-type]


def test_default_init_does_not_set_cu_seqlens() -> None:
    seg = _PackedOnly(tokens=torch.tensor([1, 2, 3, 4, 5]))
    assert seg.tokens is not None
    assert seg.cu_seqlens is None
    assert seg.lengths is None


# ---------------------------------------------------------------------------
# concat
# ---------------------------------------------------------------------------


def test_concat_two_shards() -> None:
    a = _PackedOnly.pack(tokens=[torch.tensor([1, 2]), torch.tensor([3, 4, 5])])
    b = _PackedOnly.pack(tokens=[torch.tensor([6, 7, 8, 9]), torch.tensor([10, 11])])
    merged = _PackedOnly.concat([a, b])
    assert merged.tokens.tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # lengths [2, 3, 4, 2] → cu [0, 2, 5, 9, 11]
    assert merged.cu_seqlens.tolist() == [0, 2, 5, 9, 11]


def test_concat_three_shards_offset_shift() -> None:
    a = _PackedOnly.pack(tokens=[torch.tensor([1, 2])])
    b = _PackedOnly.pack(tokens=[torch.tensor([3, 4, 5])])
    c = _PackedOnly.pack(tokens=[torch.tensor([6, 7, 8, 9])])
    merged = _PackedOnly.concat([a, b, c])
    assert merged.tokens.tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert merged.cu_seqlens.tolist() == [0, 2, 5, 9]


def test_concat_propagates_two_packed_fields() -> None:
    a = _TwoPacked.pack(
        tokens=[torch.tensor([1, 2]), torch.tensor([3, 4, 5])],
        log_probs=[torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4, 0.5])],
    )
    b = _TwoPacked.pack(
        tokens=[torch.tensor([6, 7])],
        log_probs=[torch.tensor([0.6, 0.7])],
    )
    merged = _TwoPacked.concat([a, b])
    assert merged.tokens.tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert merged.log_probs.tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert merged.cu_seqlens.tolist() == [0, 2, 5, 7]


# ---------------------------------------------------------------------------
# slice
# ---------------------------------------------------------------------------


def _make_basic_packed() -> _PackedOnly:
    return _PackedOnly.pack(
        tokens=[
            torch.tensor([1, 2]),
            torch.tensor([3, 4, 5]),
            torch.tensor([6, 7, 8, 9]),
        ]
    )


def test_slice_basic() -> None:
    seg = _make_basic_packed()

    s02 = seg.slice(0, 2)
    assert s02.tokens.tolist() == [1, 2, 3, 4, 5]
    assert s02.cu_seqlens.tolist() == [0, 2, 5]

    s13 = seg.slice(1, 3)
    assert s13.tokens.tolist() == [3, 4, 5, 6, 7, 8, 9]
    assert s13.cu_seqlens.tolist() == [0, 3, 7]


def test_slice_re_zeroes_cu() -> None:
    seg = _make_basic_packed()
    sub = seg.slice(1, 3)
    assert sub.cu_seqlens[0].item() == 0


def test_slice_empty_range() -> None:
    seg = _make_basic_packed()
    sub = seg.slice(1, 1)
    assert sub.tokens.shape == (0,)
    assert sub.cu_seqlens.tolist() == [0]


def test_slice_full_range_matches_original() -> None:
    seg = _make_basic_packed()
    full = seg.slice(0, 3)
    assert torch.equal(full.tokens, seg.tokens)
    assert torch.equal(full.cu_seqlens, seg.cu_seqlens)


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------


def test_select_reorder() -> None:
    seg = _make_basic_packed()
    picked = seg.select(torch.tensor([2, 0]))
    # Sample 2: [6, 7, 8, 9]; Sample 0: [1, 2]
    assert picked.tokens.tolist() == [6, 7, 8, 9, 1, 2]
    # New sizes: [4, 2] → cu [0, 4, 6]
    assert picked.cu_seqlens.tolist() == [0, 4, 6]


def test_select_subsample() -> None:
    seg = _make_basic_packed()
    picked = seg.select(torch.tensor([1]))
    assert picked.tokens.tolist() == [3, 4, 5]
    assert picked.cu_seqlens.tolist() == [0, 3]


# ---------------------------------------------------------------------------
# clone, to_device
# ---------------------------------------------------------------------------


def test_clone_preserves_cu_seqlens() -> None:
    seg = _make_basic_packed()
    cloned = seg.clone()
    assert torch.equal(cloned.tokens, seg.tokens)
    assert torch.equal(cloned.cu_seqlens, seg.cu_seqlens)
    # Independent storage:
    assert cloned.tokens.data_ptr() != seg.tokens.data_ptr()
    assert cloned.cu_seqlens.data_ptr() != seg.cu_seqlens.data_ptr()


def test_to_device_moves_cu_seqlens() -> None:
    seg = _make_basic_packed()
    moved = seg.to_device("cpu")  # no-op device but exercises the path
    assert moved.tokens.device.type == "cpu"
    assert moved.cu_seqlens.device.type == "cpu"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_packed_handles_none_through_ops() -> None:
    a = _PackedOnly.pack(tokens=None)
    b = _PackedOnly.pack(tokens=None)
    merged = _PackedOnly.concat([a, b])
    assert merged.tokens is None
    assert merged.cu_seqlens is None


def test_batch_size_falls_back_to_cu_seqlens() -> None:
    seg = _PackedOnly.pack(
        tokens=[
            torch.tensor([1, 2]),
            torch.tensor([3, 4, 5]),
            torch.tensor([6, 7, 8, 9]),
        ]
    )
    # _PackedOnly has no concat field; batch_size must come from cu_seqlens.
    assert seg.batch_size == 3


def test_lengths_property() -> None:
    seg = _make_basic_packed()
    assert seg.lengths.tolist() == [2, 3, 4]


def test_lengths_property_none_when_unset() -> None:
    seg = _PackedOnly(tokens=None)
    assert seg.lengths is None


def test_packed_field_kind_metadata() -> None:
    """Sanity: the constructor really tags the field with FieldKind.PACKED."""
    from dataclasses import fields as dc_fields

    field_kinds = {f.name: f.metadata.get("kind") for f in dc_fields(_TwoPacked)}
    assert field_kinds["tokens"] is FieldKind.PACKED
    assert field_kinds["log_probs"] is FieldKind.PACKED
