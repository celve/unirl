"""TensorMeta must survive the DP collect→re-dispatch round-trip.

A remote ``@distributed`` call's outputs come back as :class:`TensorMeta`
(``refs`` + ``sizes``; the real tensors stay on the producing worker). Chaining
that output into the next ``DP_ALL`` call re-splits it, so splitting a
TensorMeta must be the *structural inverse* of the merge that built it —
partition the ``refs`` list — because the data is remote and opaque: row-
indexing a TensorMeta is impossible, so ``slice`` / ``__getitem__`` only support
ranges that land on ref boundaries and raise otherwise ("hydrate first").

``concat`` (collect side) was already TensorMeta-aware; these tests cover the
inverse ``slice`` / ``__getitem__`` (re-dispatch side) — the TensorMeta leaf
that the field-kind-aware ``chunk``/``slice`` split path now also has to handle.
"""

from __future__ import annotations

import pytest

from diffusionrl.distributed.tensor.transport import TensorMeta


def _tm(n_refs: int, rows_each: int) -> TensorMeta:
    return TensorMeta(refs=[f"h{i}" for i in range(n_refs)], sizes=[rows_each] * n_refs)


def test_concat_then_slice_round_trip_is_identity():
    """slice on ref boundaries recovers each per-source TensorMeta concat stacked."""
    parts = [TensorMeta(refs=[f"h{i}"], sizes=[2]) for i in range(4)]
    merged = TensorMeta.concat(parts)
    assert merged.refs == ["h0", "h1", "h2", "h3"]
    assert merged.sizes == [2, 2, 2, 2]
    assert merged.batch_size == 8

    for i in range(4):
        shard = merged.slice(2 * i, 2 * i + 2)
        assert shard.refs == [f"h{i}"]
        assert shard.sizes == [2]
        assert shard.batch_size == 2


def test_getitem_matches_slice():
    """The packed path slices via ``value[a:b]`` — __getitem__ mirrors slice."""
    merged = _tm(4, 2)
    assert merged[4:8].refs == ["h2", "h3"]
    assert merged[:4].refs == ["h0", "h1"]
    assert merged[:].refs == ["h0", "h1", "h2", "h3"]


def test_slice_aligns_by_row_not_ref_count():
    """Uneven ref sizes: the range is matched against cumulative row offsets."""
    merged = TensorMeta(refs=["a", "b", "c"], sizes=[3, 1, 2])  # offsets 0,3,4,6
    assert merged.slice(0, 3).refs == ["a"]
    assert merged.slice(3, 4).refs == ["b"]
    assert merged.slice(4, 6).refs == ["c"]
    assert merged.slice(0, 4).refs == ["a", "b"]


def test_non_aligned_slice_raises():
    """Intra-ref ranges have no representation — must fail closed, not corrupt."""
    merged = _tm(4, 2)  # boundaries at 0,2,4,6,8
    with pytest.raises(NotImplementedError):
        merged.slice(1, 3)
    with pytest.raises(NotImplementedError):
        _ = merged[3:5]


def test_sliced_handle_clones():
    """_slice_packed_data does ``value[a:b].clone()`` — the shard must clone."""
    merged = _tm(4, 2)
    cloned = merged[0:2].clone()
    assert cloned.refs == ["h0"]
    assert cloned.sizes == [2]
