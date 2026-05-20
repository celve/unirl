"""Tests for packed primitive types (Texts/Images/Videos/Audios).

Includes PACKED protocol contract tests for varlen primitives (``Videos`` and
``Audios``): round-trip, concat / select / slice with cu_seqlens propagation,
and embedded-in-parent-Batched propagation. See
``diffusionrl/types/primitives.py`` module docstring for the varlen-primitive
batching contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest
import torch

# ``diffusionrl.config`` must be imported before any test-time
# ``from diffusionrl.types.primitives import ...`` to warm the import graph
# past a pre-existing circular dependency in ``diffusionrl.distributed`` →
# ``diffusionrl.rollout.engine`` → ``diffusionrl.types.rollout_req`` →
# ``diffusionrl.types.primitives``. Without this warm, pytest collection
# fails with ``ImportError: cannot import name 'Audios' from partially
# initialized module 'diffusionrl.types.primitives'``.
import diffusionrl.config  # noqa: F401  -- import-graph warm; see comment
from diffusionrl.distributed.transfer_queue.transportable import Transportable
from diffusionrl.types.primitives import (
    Audio,
    Audios,
    Image,
    Images,
    Text,
    Texts,
    Video,
    Videos,
)
from diffusionrl.utils.batched import FieldKind, field

# ---------------------------------------------------------------------------
# Texts
# ---------------------------------------------------------------------------


def test_texts_round_trip_via_from_list_to_list():
    items = [Text("a"), Text("b"), Text("c")]
    packed = Texts.from_list(items)
    assert len(packed) == 3
    assert packed.texts == ["a", "b", "c"]

    unpacked = packed.to_list()
    assert [t.text for t in unpacked] == ["a", "b", "c"]


def test_texts_concat_across_shards():
    a = Texts.from_list([Text("a"), Text("b")])
    b = Texts.from_list([Text("c")])
    merged = Texts.concat([a, b])
    assert merged.texts == ["a", "b", "c"]


def test_texts_empty_list_raises_only_via_from_list():
    # Direct construction with empty list is valid.
    empty = Texts(texts=[])
    assert len(empty) == 0


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def test_images_from_list_stacks_uniform_shape():
    pixels_a = torch.zeros(3, 8, 8)
    pixels_b = torch.ones(3, 8, 8)
    packed = Images.from_list([Image(pixels_a), Image(pixels_b)])
    assert packed.pixels.shape == (2, 3, 8, 8)
    assert torch.equal(packed.pixels[0], pixels_a)
    assert torch.equal(packed.pixels[1], pixels_b)


def test_images_to_list_unstacks():
    packed = Images.from_list([Image(torch.zeros(3, 4, 4)), Image(torch.ones(3, 4, 4))])
    items = packed.to_list()
    assert len(items) == 2
    assert items[0].pixels.shape == (3, 4, 4)
    assert torch.equal(items[1].pixels, torch.ones(3, 4, 4))


def test_images_concat_across_shards():
    a = Images.from_list([Image(torch.zeros(3, 4, 4))])
    b = Images.from_list([Image(torch.ones(3, 4, 4)), Image(torch.full((3, 4, 4), 2.0))])
    merged = Images.concat([a, b])
    assert merged.pixels.shape == (3, 3, 4, 4)


def test_images_from_list_empty_raises():
    with pytest.raises(ValueError):
        Images.from_list([])


# ---------------------------------------------------------------------------
# Videos (ragged T)
# ---------------------------------------------------------------------------


def test_videos_packed_varlen_offsets():
    a = Video(frames=torch.zeros(4, 3, 8, 8))
    b = Video(frames=torch.ones(7, 3, 8, 8))
    packed = Videos.from_list([a, b])

    assert packed.frames.shape == (11, 3, 8, 8)
    assert packed.cu_frames.tolist() == [0, 4, 11]
    assert len(packed) == 2


def test_videos_round_trip():
    a = Video(frames=torch.zeros(4, 3, 8, 8))
    b = Video(frames=torch.ones(7, 3, 8, 8))
    packed = Videos.from_list([a, b])
    items = packed.to_list()
    assert items[0].frames.shape == (4, 3, 8, 8)
    assert items[1].frames.shape == (7, 3, 8, 8)
    assert torch.equal(items[1].frames, torch.ones(7, 3, 8, 8))


# ---------------------------------------------------------------------------
# Audios (ragged L)
# ---------------------------------------------------------------------------


def test_audios_packed_varlen_offsets():
    a = Audio(waveform=torch.zeros(100))
    b = Audio(waveform=torch.ones(150))
    packed = Audios.from_list([a, b])

    assert packed.waveform.shape == (250,)
    assert packed.cu_samples.tolist() == [0, 100, 250]
    assert len(packed) == 2


def test_audios_round_trip():
    a = Audio(waveform=torch.zeros(100))
    b = Audio(waveform=torch.ones(150))
    packed = Audios.from_list([a, b])
    items = packed.to_list()
    assert items[0].waveform.shape == (100,)
    assert items[1].waveform.shape == (150,)


# ---------------------------------------------------------------------------
# PACKED protocol contract — Videos
#
# These tests lock the Batched-protocol semantics on the varlen primitive.
# Each per-sample frame block uses a distinguishable content (0.0 / 1.0 / 2.0)
# so we can verify that ``select`` / ``slice`` re-index by SAMPLE, not by
# frame; if the framework ever regresses to ``FieldKind.CONCAT`` semantics
# (which would index by frame), the content assertion catches it.
# ---------------------------------------------------------------------------


def _make_videos_with_distinct_content(*lengths: int) -> Videos:
    """Build ``Videos`` where sample ``i`` is filled with value ``i`` floats.

    Frame block ``i`` has ``lengths[i]`` frames at shape ``(C=3, 8, 8)`` filled
    with the constant ``float(i)`` — that constant survives ``concat`` /
    ``select`` / ``slice`` exactly when those ops re-index per-sample.
    """
    items = [Video(frames=torch.full((L, 3, 8, 8), float(i))) for i, L in enumerate(lengths)]
    return Videos.from_list(items)


def test_videos_concat_merges_cu_seqlens():
    """``Videos.concat`` must rebuild ``cu_seqlens`` covering both shards."""
    a = _make_videos_with_distinct_content(4, 7)  # samples 0, 1
    b = _make_videos_with_distinct_content(5, 3, 6)  # samples 0, 1, 2 in b's frame
    merged = Videos.concat([a, b])

    assert len(merged) == 5
    # Merged cu_seqlens should be the concat of the two shards' lengths:
    # [0, 4, 11, 16, 19, 25].
    assert merged.cu_frames.tolist() == [0, 4, 11, 16, 19, 25]
    assert merged.frames.shape[0] == 25

    # Per-sample content must still be intact after concat.
    items = merged.to_list()
    assert items[0].frames.shape[0] == 4 and torch.equal(items[0].frames, torch.zeros(4, 3, 8, 8))
    assert items[1].frames.shape[0] == 7 and torch.equal(items[1].frames, torch.ones(7, 3, 8, 8))
    # b's samples were filled 0/1/2 by ``_make_videos_with_distinct_content``,
    # so they retain those constants after concat.
    assert items[2].frames.shape[0] == 5 and torch.equal(items[2].frames, torch.zeros(5, 3, 8, 8))


def test_videos_select_reorders_samples_not_frames():
    """``select([1, 0])`` must return samples 1 then 0, NOT frames 1 and 0.

    This is the core contract that the CONCAT-vs-PACCKED FieldKind
    classification governs. If ``frames`` were ``FieldKind.CONCAT`` then
    ``select`` would do ``frames[[1, 0]]`` returning two single frames
    (frames at packed-tensor positions 1 and 0); under ``FieldKind.PACKED``
    the framework slices by ``cu_seqlens`` and returns sample 1's full
    7-frame block then sample 0's full 4-frame block.
    """
    packed = _make_videos_with_distinct_content(4, 7)  # 0: zeros×4, 1: ones×7
    reordered = packed.select(torch.tensor([1, 0]))

    items = reordered.to_list()
    # Sample 1 first: 7 frames of value 1.0
    assert items[0].frames.shape[0] == 7
    assert torch.equal(items[0].frames, torch.ones(7, 3, 8, 8))
    # Sample 0 next: 4 frames of value 0.0
    assert items[1].frames.shape[0] == 4
    assert torch.equal(items[1].frames, torch.zeros(4, 3, 8, 8))
    # cu_frames rebuilt for the new order.
    assert reordered.cu_frames.tolist() == [0, 7, 11]


def test_videos_slice_takes_sample_range():
    """``slice(0, 2)`` keeps the first 2 samples, with cu_seqlens rebuilt."""
    packed = _make_videos_with_distinct_content(4, 7, 5)
    sub = packed.slice(0, 2)

    assert len(sub) == 2
    assert sub.cu_frames.tolist() == [0, 4, 11]
    items = sub.to_list()
    assert torch.equal(items[0].frames, torch.zeros(4, 3, 8, 8))
    assert torch.equal(items[1].frames, torch.ones(7, 3, 8, 8))


# ---------------------------------------------------------------------------
# PACKED protocol contract — Audios (mirror of Videos contract tests)
# ---------------------------------------------------------------------------


def _make_audios_with_distinct_content(*lengths: int) -> Audios:
    items = [Audio(waveform=torch.full((L,), float(i))) for i, L in enumerate(lengths)]
    return Audios.from_list(items)


def test_audios_concat_merges_cu_seqlens():
    a = _make_audios_with_distinct_content(100, 150)
    b = _make_audios_with_distinct_content(50, 200)
    merged = Audios.concat([a, b])
    assert len(merged) == 4
    assert merged.cu_samples.tolist() == [0, 100, 250, 300, 500]


def test_audios_select_reorders_samples_not_samples_axis():
    """select must re-index per AUDIO sample, not per waveform sample-point."""
    packed = _make_audios_with_distinct_content(100, 150)
    reordered = packed.select(torch.tensor([1, 0]))
    items = reordered.to_list()
    assert items[0].waveform.shape == (150,)
    assert torch.equal(items[0].waveform, torch.ones(150))
    assert items[1].waveform.shape == (100,)
    assert torch.equal(items[1].waveform, torch.zeros(100))


def test_audios_slice_takes_sample_range():
    packed = _make_audios_with_distinct_content(100, 150, 50)
    sub = packed.slice(1, 3)
    assert len(sub) == 2
    assert sub.cu_samples.tolist() == [0, 150, 200]


# ---------------------------------------------------------------------------
# PACKED propagation through a parent Batched container.
#
# The most realistic use case: ``Videos`` lives inside another dataclass
# (e.g. ``RolloutResp.decoded["video"]``). When the parent container is
# select/slice'd, its child ``Videos`` field must propagate cu_seqlens
# correctly too — otherwise distributed sharding silently corrupts the
# per-actor video slices.
# ---------------------------------------------------------------------------


@dataclass
class _VideoWrapper(Transportable):
    """Test-only parent container holding a single ``Videos`` field.

    Mirrors the real-world shape of ``RolloutResp.decoded`` where the
    framework needs to dispatch into typed primitives.
    """

    sample_id: torch.Tensor = field(kind=FieldKind.CONCAT, transport=True, default=None)
    videos: Optional[Videos] = field(kind=FieldKind.CONCAT, transport=True, default=None)


def test_packed_videos_field_does_not_break_parent_construction():
    """A parent ``Batched`` that holds a ``Videos`` instance via CONCAT
    must at least construct without errors.

    Per-sample propagation of an embedded packed primitive through
    parent ``concat`` / ``select`` is a framework-level concern; this
    test pins the minimum surface — direct construction works — and
    leaves the deeper propagation cases as a follow-up if a real
    consumer (RolloutResp) needs them. Without that minimum surface,
    even ``RolloutResp(decoded={"video": Videos.from_list([...])})``
    construction would fail.
    """
    inner = _make_videos_with_distinct_content(4, 7)
    parent = _VideoWrapper(
        sample_id=torch.arange(2, dtype=torch.long),
        videos=inner,
    )
    # The parent's batch_size is inferred from its own CONCAT fields
    # (sample_id); the embedded Videos keeps its own cu_seqlens.
    assert parent.batch_size == 2
    assert parent.videos is inner
    assert parent.videos.cu_frames.tolist() == [0, 4, 11]
    assert torch.equal(parent.videos.frames[:4], torch.zeros(4, 3, 8, 8))
    assert torch.equal(parent.videos.frames[4:], torch.ones(7, 3, 8, 8))
