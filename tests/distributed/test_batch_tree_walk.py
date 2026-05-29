"""Packed-batch ``cu_seqlens`` must survive the framework's tree-walkers.

``_packed_cu_seqlens`` is hidden instance metadata, not a dataclass field, so a
naive ``type(obj)(**fields)`` rebuild silently drops it — breaking packed
``slice`` / ``select`` / ``concat`` downstream. Every tree-walker that rebuilds
a ``Batch`` now routes through :meth:`Batch.map`, which carries the metadata
over. These tests drive the *real* walkers on a packed batch whose leaves are
plain tensors (so each walker's leaf transform is a no-op and only the
``Batch`` rebuild branch is exercised) and assert the offsets survive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from diffusionrl.distributed.group.handle import Handle
from diffusionrl.distributed.group.worker import Worker
from diffusionrl.distributed.tensor.batch import (
    Batch,
    concat_field,
    packed_field,
    shared_field,
)
from diffusionrl.distributed.utils import infer_and_validate_batch_size


@dataclass
class _Seg(Batch):
    tokens: Optional[torch.Tensor] = packed_field(default=None)
    log_probs: Optional[torch.Tensor] = packed_field(default=None)
    idx: Optional[torch.Tensor] = concat_field(default=None)
    name: str = shared_field(default="x")


def _make_seg() -> _Seg:
    # 3 samples, per-sample token lengths 2/3/5 → total=10, cu=[0,2,5,10].
    return _Seg.pack(
        tokens=[torch.arange(2), torch.arange(3), torch.arange(5)],
        log_probs=[torch.zeros(2), torch.zeros(3), torch.zeros(5)],
        idx=torch.arange(3),
        name="hi",
    )


def _assert_intact(seg: _Seg) -> None:
    assert seg.cu_seqlens is not None, "cu_seqlens was dropped"
    assert seg.cu_seqlens.tolist() == [0, 2, 5, 10]
    assert seg.batch_size == 3
    # field-kind metadata is class-level → packed ops must still work
    sliced = seg.slice(1, 3)
    assert sliced.batch_size == 2
    assert sliced.tokens.numel() == 8  # samples 1+2 → 3+5 tokens


# ── Batch.map (the shared mechanism) ──


def test_map_preserves_cu_seqlens_under_identity():
    _assert_intact(_make_seg().map(lambda v: v))


def test_map_applies_fn_and_still_preserves_cu_seqlens():
    out = _make_seg().map(lambda v: v + 1 if isinstance(v, torch.Tensor) else v)
    assert torch.equal(out.idx, torch.arange(3) + 1)
    assert out.cu_seqlens.tolist() == [0, 2, 5, 10]


# ── Real walkers, nested in containers ──


def test_worker_transform_tree():
    w = Worker.__new__(Worker)
    out = w._transform_tree({"a": [_make_seg()], "b": (_make_seg(),)}, lambda x: x)
    _assert_intact(out["a"][0])
    _assert_intact(out["b"][0])


def test_handle_rebind_tree():
    h = Handle.__new__(Handle)
    out = h._rebind_tree({"seg": _make_seg()}, worker_handle=None)
    _assert_intact(out["seg"])


def test_handle_unwrap():
    h = Handle.__new__(Handle)
    out = h._unwrap([_make_seg()], dst_worker_id="dw0", dst_device_id=0, foreign={})
    _assert_intact(out[0])


def test_handle_substitute():
    h = Handle.__new__(Handle)
    out = h._substitute((_make_seg(),), subs={})
    _assert_intact(out[0])


def test_handle_cat_multi():
    h = Handle.__new__(Handle)
    out = h._cat_multi({"seg": _make_seg()}, dst_worker=None)
    _assert_intact(out["seg"])


# ── _collect_batch_sizes / infer_and_validate_batch_size ──


def test_infer_batch_size_uses_sample_count_not_packed_total():
    # Packed total is 10, but the batch holds 3 samples — the canonical
    # batch size must be the sample count, else split_value broadcasts it.
    assert infer_and_validate_batch_size((_make_seg(),), {}) == 3
