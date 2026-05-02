"""Tests for ``diffusionrl.distributed.transfer_queue.transportable.Transportable``.

Covers:
- The four roundtrip methods (to_tensordict / replace_with_meta /
  collect_remote_metas / restore_from_tensordict) on synthetic dataclasses.
- Dotted-path key construction and that get-side lookup uses the wrapper's
  ``_data_key`` (so sender / receiver shapes can differ).
- ``transport=True``-gated recursion (untagged Transportable children are
  invisible to the walker).
- ``dehydrate`` / ``hydrate`` against a stub client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest
import torch

from diffusionrl.distributed.transfer_queue.meta import TqMeta
from diffusionrl.distributed.transfer_queue.transportable import Transportable
from diffusionrl.utils.batched import FieldKind, concat_field, field, shared_field

# =========================================================
# fixtures
# =========================================================


@dataclass
class _Inner(Transportable):
    embeds: torch.Tensor = field(kind=FieldKind.CONCAT, transport=True)
    mask: Optional[torch.Tensor] = concat_field(default=None)


@dataclass
class _Outer(Transportable):
    latents: torch.Tensor = field(kind=FieldKind.CONCAT, transport=True)
    images: Optional[List[Any]] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    advantages: Optional[torch.Tensor] = concat_field(default=None)
    inner: Optional[_Inner] = field(kind=FieldKind.CONCAT, default=None, transport=True)


class _FakeMeta:
    """Mimics the upstream ``BatchMeta.select_fields`` / ``union`` surface."""

    def __init__(self, keys):
        self._keys = list(keys)

    def select_fields(self, ks):
        return _FakeMeta([k for k in self._keys if k in ks])

    def union(self, other):
        merged = list(self._keys)
        for k in other._keys:
            if k not in merged:
                merged.append(k)
        return _FakeMeta(merged)


def _make_outer(bs: int = 4) -> _Outer:
    return _Outer(
        latents=torch.zeros(bs, 3, 8, 8),
        images=[f"img-{i}" for i in range(bs)],
        advantages=torch.arange(bs, dtype=torch.float32),
        inner=_Inner(embeds=torch.arange(bs * 5).reshape(bs, 5).float()),
    )


# =========================================================
# to_tensordict
# =========================================================


def test_to_tensordict_uses_dotted_path_keys() -> None:
    outer = _make_outer()
    td = outer.to_tensordict()
    assert td is not None
    # latents + images at the root; inner.embeds via the recursion.
    assert sorted(td.keys()) == ["images", "inner.embeds", "latents"]
    assert int(td.batch_size[0]) == 4


def test_to_tensordict_returns_none_when_no_populated_transport_fields() -> None:
    @dataclass
    class _OnlyConcat(Transportable):
        rewards: torch.Tensor = concat_field()

    obj = _OnlyConcat(rewards=torch.zeros(3))
    assert obj.to_tensordict() is None


def test_to_tensordict_skips_none_and_wrapper_slots() -> None:
    outer = _make_outer()
    outer.images = None  # populated → None: skip
    td = outer.to_tensordict()
    assert sorted(td.keys()) == ["inner.embeds", "latents"]


def test_dotted_paths_are_unique_for_same_named_leaves_at_different_depths() -> None:
    """Two leaves with the same attribute name at different depths get
    distinct keys via the prefix walk — no collision."""

    @dataclass
    class _LeftBranch(Transportable):
        x: torch.Tensor = field(kind=FieldKind.CONCAT, transport=True)

    @dataclass
    class _RightBranch(Transportable):
        x: torch.Tensor = field(kind=FieldKind.CONCAT, transport=True)

    @dataclass
    class _Root(Transportable):
        left: _LeftBranch = field(kind=FieldKind.CONCAT, transport=True)
        right: _RightBranch = field(kind=FieldKind.CONCAT, transport=True)

    obj = _Root(
        left=_LeftBranch(x=torch.zeros(3, 1)),
        right=_RightBranch(x=torch.ones(3, 1)),
    )
    td = obj.to_tensordict()
    assert sorted(td.keys()) == ["left.x", "right.x"]


# =========================================================
# transport-gated recursion
# =========================================================


def test_untagged_transportable_child_is_invisible_to_the_walker() -> None:
    """A field whose value is a ``Transportable`` but is *not* tagged
    ``transport=True`` is skipped entirely — its leaves never appear."""

    @dataclass
    class _Hidden(Transportable):
        secret: torch.Tensor = field(kind=FieldKind.CONCAT, transport=True)

    @dataclass
    class _Visible(Transportable):
        public: torch.Tensor = field(kind=FieldKind.CONCAT, transport=True)
        # NOT tagged — recursion does not enter this field.
        hidden: Optional[_Hidden] = concat_field(default=None)

    obj = _Visible(
        public=torch.zeros(2, 3),
        hidden=_Hidden(secret=torch.zeros(2, 5)),
    )
    td = obj.to_tensordict()
    # Only ``public`` shows up; ``hidden.secret`` is invisible because the
    # ``hidden`` field is not transport-tagged.
    assert sorted(td.keys()) == ["public"]


# =========================================================
# replace_with_meta + collect + restore (dotted paths, wrapper-keyed lookup)
# =========================================================


def test_replace_with_meta_wraps_only_transport_leaves_with_path_keys() -> None:
    outer = _make_outer()
    keys = list(outer.to_tensordict().keys())
    outer.replace_with_meta(_FakeMeta(keys))

    assert isinstance(outer.latents, TqMeta)
    assert outer.latents._data_key == "latents"
    assert isinstance(outer.images, TqMeta)
    assert outer.images._data_key == "images"
    assert isinstance(outer.inner.embeds, TqMeta)
    assert outer.inner.embeds._data_key == "inner.embeds"
    # Untagged fields stay untouched.
    assert isinstance(outer.advantages, torch.Tensor)
    assert outer.inner.mask is None


def test_collect_remote_metas_finds_all_subtree_wrappers() -> None:
    outer = _make_outer()
    outer.replace_with_meta(_FakeMeta(list(outer.to_tensordict().keys())))
    assert len(outer.collect_remote_metas()) == 3


def test_restore_uses_wrapper_data_key_not_walker_key() -> None:
    """Even if the receiver's walker computes a different path than the
    sender's, lookup goes through ``wrapper._data_key`` so the wire format is
    determined entirely by the sender."""
    outer = _make_outer()
    outer.replace_with_meta(_FakeMeta(list(outer.to_tensordict().keys())))

    # Build a TD whose keys match the wrappers' _data_key strings (dotted).
    restored = {
        "latents": torch.ones(4, 3, 8, 8) * 9,
        "images": [f"restored-{i}" for i in range(4)],
        "inner.embeds": torch.ones(4, 5) * 7,
    }
    outer.restore_from_tensordict(restored)

    assert torch.equal(outer.latents, restored["latents"])
    assert outer.images == restored["images"]
    assert torch.equal(outer.inner.embeds, restored["inner.embeds"])
    # Untagged fields untouched.
    assert torch.equal(outer.advantages, torch.arange(4, dtype=torch.float32))


def test_restore_restacks_mooncake_list_returns() -> None:
    """Mooncake's ``manager_merge_to_tensordict=False`` returns lists; verify
    re-stack on restore."""

    @dataclass
    class _Single(Transportable):
        latents: torch.Tensor = field(kind=FieldKind.CONCAT, transport=True)

    obj = _Single(latents=torch.zeros(2, 4))
    obj.replace_with_meta(_FakeMeta(["latents"]))
    obj.restore_from_tensordict({"latents": [torch.ones(4), torch.ones(4) * 2]})
    assert obj.latents.shape == (2, 4)
    assert torch.equal(obj.latents[0], torch.ones(4))
    assert torch.equal(obj.latents[1], torch.ones(4) * 2)


# =========================================================
# dehydrate / hydrate against a stub client
# =========================================================


class _StubClient:
    """Records calls for assertion; produces a fake meta on put / a fake TD on get."""

    def __init__(self):
        self.put_calls = []
        self.get_calls = []
        self._stored: Dict[str, Any] = {}

    async def async_put(self, *, data, partition_id):
        # Snapshot keys + record. Return a fake meta keyed by what was put.
        self.put_calls.append({"keys": sorted(data.keys()), "partition_id": partition_id})
        for k in data.keys():
            self._stored[k] = data[k]
        return _FakeMeta(list(data.keys()))

    async def async_get_data(self, meta):
        # Return the stored values for whatever keys are in the meta.
        self.get_calls.append({"keys": list(meta._keys)})
        return {k: self._stored[k] for k in meta._keys if k in self._stored}


def test_dehydrate_then_hydrate_roundtrip() -> None:
    """Synthetic put / get roundtrip via a stub client, driven from a sync
    test by ``asyncio.run``. Avoids a pytest-asyncio dep."""
    import asyncio

    outer = _make_outer()
    client = _StubClient()

    async def go():
        await outer.dehydrate(client, partition_id="test")
        await outer.hydrate(client)

    asyncio.run(go())

    assert len(client.put_calls) == 1
    assert sorted(client.put_calls[0]["keys"]) == ["images", "inner.embeds", "latents"]
    assert client.put_calls[0]["partition_id"] == "test"
    assert len(client.get_calls) == 1
    assert sorted(client.get_calls[0]["keys"]) == ["images", "inner.embeds", "latents"]
    # After roundtrip, slots hold tensors again.
    assert isinstance(outer.latents, torch.Tensor)
    assert isinstance(outer.inner.embeds, torch.Tensor)
    assert isinstance(outer.images, list)


# =========================================================
# ForwardContext-shaped nested Transportable: tensor on the wire,
# scalar / dtype ride the in-memory pickle path
# =========================================================


@dataclass
class _FCLike(Transportable):
    """Mirror of the ForwardContext shape: per-row tensor + untagged shared
    scalar + untagged shared dtype. Only the tensor field is transport-tagged
    so only it travels via TQ; the others survive on the dataclass instance
    via Ray's pickle path on the return value.
    """

    embeds: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None, transport=True)
    guidance_scale: float = shared_field(default=7.0)
    autocast_dtype: Optional[torch.dtype] = shared_field(default=None)


@dataclass
class _OuterWithFC(Transportable):
    latents: torch.Tensor = field(kind=FieldKind.CONCAT, transport=True)
    fc: Optional[_FCLike] = field(kind=FieldKind.CONCAT, default=None, transport=True)


def test_forward_context_shaped_nested_transport_only_tensor_on_wire() -> None:
    """The walker enters the nested ``_FCLike`` (Transportable + tagged) but
    only its tensor leaf produces a wire key; untagged scalar / dtype fields
    are invisible to the walker."""
    bs = 4
    obj = _OuterWithFC(
        latents=torch.zeros(bs, 3, 8, 8),
        fc=_FCLike(
            embeds=torch.arange(bs * 5).reshape(bs, 5).float(),
            guidance_scale=7.0,
            autocast_dtype=torch.bfloat16,
        ),
    )
    td = obj.to_tensordict()
    assert td is not None
    # Only the tagged-tensor leaves appear; guidance_scale and autocast_dtype
    # are NOT on the wire (untagged).
    assert sorted(td.keys()) == ["fc.embeds", "latents"]
    assert int(td.batch_size[0]) == bs


def test_forward_context_shaped_replace_with_meta_preserves_untagged_fields() -> None:
    """``replace_with_meta`` only swaps tagged-tensor leaves; the scalar /
    dtype fields stay on the in-memory instance, ready to be pickled by Ray
    on the return-value path."""
    bs = 2
    obj = _OuterWithFC(
        latents=torch.zeros(bs, 3, 8, 8),
        fc=_FCLike(
            embeds=torch.ones(bs, 5),
            guidance_scale=3.5,
            autocast_dtype=torch.bfloat16,
        ),
    )
    obj.replace_with_meta(_FakeMeta(list(obj.to_tensordict().keys())))

    # Tagged leaves are now TqMeta refs.
    assert isinstance(obj.latents, TqMeta)
    assert isinstance(obj.fc.embeds, TqMeta)
    assert obj.fc.embeds._data_key == "fc.embeds"
    # Untagged scalar / dtype fields are untouched — they ride the Ray pickle.
    assert obj.fc.guidance_scale == 3.5
    assert obj.fc.autocast_dtype is torch.bfloat16


def test_forward_context_shaped_dehydrate_hydrate_roundtrip() -> None:
    """Full TQ roundtrip: tensor goes through the stub client; scalar / dtype
    survive on the instance unchanged."""
    import asyncio

    bs = 3
    obj = _OuterWithFC(
        latents=torch.arange(bs * 2 * 2 * 2).reshape(bs, 2, 2, 2).float(),
        fc=_FCLike(
            embeds=torch.arange(bs * 4).reshape(bs, 4).float() + 100,
            guidance_scale=5.5,
            autocast_dtype=torch.float16,
        ),
    )
    expected_latents = obj.latents.clone()
    expected_embeds = obj.fc.embeds.clone()

    client = _StubClient()

    async def go():
        await obj.dehydrate(client, partition_id="fc-test")
        await obj.hydrate(client)

    asyncio.run(go())

    # Tagged tensors round-tripped via the wire.
    assert isinstance(obj.latents, torch.Tensor)
    assert torch.equal(obj.latents, expected_latents)
    assert isinstance(obj.fc, _FCLike)
    assert isinstance(obj.fc.embeds, torch.Tensor)
    assert torch.equal(obj.fc.embeds, expected_embeds)
    # Untagged scalar + dtype survived in-memory unchanged.
    assert obj.fc.guidance_scale == 5.5
    assert obj.fc.autocast_dtype is torch.float16
    # Wire only carried the tagged-tensor leaves.
    assert sorted(client.put_calls[0]["keys"]) == ["fc.embeds", "latents"]


def test_forward_context_shaped_none_tensor_field_skipped() -> None:
    """When a transport-tagged tensor field is ``None`` it is filtered at
    ``_walk_leaves`` (line 117) and never appears on the wire — matching the
    real ForwardContext flow where most fields default to ``None``."""
    bs = 2
    obj = _OuterWithFC(
        latents=torch.zeros(bs, 3),
        fc=_FCLike(embeds=None, guidance_scale=1.0, autocast_dtype=None),
    )
    td = obj.to_tensordict()
    # ``fc.embeds`` is None → skipped. Only the outer ``latents`` remains.
    assert sorted(td.keys()) == ["latents"]


# =========================================================
# Batched.concat continues to work when slots hold wrappers
# (gated on upstream lib availability — TqMeta.concat needs BatchMeta)
# =========================================================


_UPSTREAM_TQ_AVAILABLE = True
try:  # pragma: no cover — environment probe only
    import transfer_queue as _upstream_tq  # noqa: F401
except ImportError:
    _UPSTREAM_TQ_AVAILABLE = False


@pytest.mark.skipif(
    not _UPSTREAM_TQ_AVAILABLE,
    reason="upstream transfer_queue not installed; TqMeta.concat needs BatchMeta",
)
def test_concat_with_wrapper_slots_preserves_batched_semantics() -> None:
    a = _make_outer(bs=2)
    b = _make_outer(bs=3)
    a.replace_with_meta(_FakeMeta(list(a.to_tensordict().keys())))
    b.replace_with_meta(_FakeMeta(list(b.to_tensordict().keys())))
    merged = _Outer.concat([a, b])
    assert isinstance(merged.latents, TqMeta)
    assert isinstance(merged.inner.embeds, TqMeta)
    assert merged.advantages.shape[0] == 5
