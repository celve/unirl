from __future__ import annotations

import pytest

from diffusionrl.distributed.group import placement as placement_mod
from diffusionrl.distributed.group.device_pool import DevicePool
from diffusionrl.distributed.group.handle import Handle
from diffusionrl.distributed.group.placement import current_placement, placement, remote


@pytest.fixture(autouse=True)
def _reset_module_state():
    placement_mod._current = None
    _stub_counter["n"] = 0
    yield
    placement_mod._current = None


def _pool(num_devices: int = 8, devices_per_node: int = 8) -> DevicePool:
    return DevicePool(num_devices=num_devices, devices_per_node=devices_per_node, workers_per_device=2)


def test_no_active_scope_returns_none():
    assert current_placement() is None


def test_shared_mode_returns_same_slot_for_every_role():
    pool = _pool()
    with placement(pool, fraction=1.0) as scope:
        assert scope.devices == tuple(range(8))
        assert scope.assign() == (list(range(8)), 0)
        assert scope.assign() == (list(range(8)), 0)
        assert scope.assign() == (list(range(8)), 0)


def test_isolated_mode_bumps_slot_per_role():
    pool = _pool()
    with placement(pool, fraction=1.0, shared_workers=False) as scope:
        assert scope.assign() == (list(range(8)), 0)
        assert scope.assign() == (list(range(8)), 1)
        assert scope.assign() == (list(range(8)), 2)


def test_sibling_scopes_get_disjoint_slabs():
    pool = _pool()
    with placement(pool, fraction=0.5) as a:
        a_devices = a.devices
    with placement(pool, fraction=0.5) as b:
        b_devices = b.devices
    assert a_devices == (0, 1, 2, 3)
    assert b_devices == (4, 5, 6, 7)
    assert set(a_devices).isdisjoint(b_devices)


def test_three_sibling_scopes_each_quarter():
    pool = _pool()
    with placement(pool, fraction=0.25) as a:
        pass
    with placement(pool, fraction=0.5) as b:
        pass
    with placement(pool, fraction=0.25) as c:
        pass
    assert a.devices == (0, 1)
    assert b.devices == (2, 3, 4, 5)
    assert c.devices == (6, 7)


def test_nested_scope_carves_subslab_of_parent():
    pool = _pool()
    with placement(pool, fraction=1.0) as outer:
        with placement(pool, fraction=0.5) as inner:
            assert inner.devices == (0, 1, 2, 3)
            assert set(inner.devices).issubset(outer.devices)


def test_nested_shared_inherits_parent_base_slot():
    pool = _pool()
    with placement(pool, fraction=1.0) as outer:
        assert outer.assign() == (list(range(8)), 0)
        with placement(pool, fraction=1.0) as inner:
            assert inner.assign() == (list(range(8)), 0)


def test_nested_isolated_lands_above_parent_roles():
    pool = _pool()
    with placement(pool, fraction=1.0) as outer:
        outer.assign()  # parent role on slot 0
        with placement(pool, fraction=1.0, shared_workers=False) as inner:
            assert inner.assign() == (list(range(8)), 1)
            assert inner.assign() == (list(range(8)), 2)


def test_parent_slot_advances_after_isolated_child_exits():
    pool = _pool()
    with placement(pool, fraction=1.0, shared_workers=False) as outer:
        assert outer.assign() == (list(range(8)), 0)
        with placement(pool, fraction=1.0, shared_workers=False) as inner:
            inner.assign()  # slot 1
            inner.assign()  # slot 2
        # outer's next role lands above inner's consumption
        assert outer.assign() == (list(range(8)), 3)


def test_current_placement_tracks_nesting():
    pool = _pool()
    assert current_placement() is None
    with placement(pool, fraction=1.0) as outer:
        assert current_placement() is outer
        with placement(pool, fraction=0.5) as inner:
            assert current_placement() is inner
        assert current_placement() is outer
    assert current_placement() is None


def test_top_level_claim_blocks_sibling_overlap():
    pool = _pool()
    with placement(pool, fraction=0.5) as a:
        pass
    with placement(pool, fraction=0.5) as b:
        pass
    assert pool._claimed == set(range(8))
    assert set(a.devices).isdisjoint(b.devices)


def test_non_integer_fraction_raises():
    pool = _pool()
    with pytest.raises(ValueError, match="not an integer"):
        with placement(pool, fraction=0.3):
            pass


def test_zero_devices_raises():
    pool = _pool()
    with pytest.raises(ValueError, match="0 devices"):
        with placement(pool, fraction=0.0):
            pass


def test_oversubscription_raises():
    pool = _pool()
    with placement(pool, fraction=1.0):
        pass
    with pytest.raises(ValueError, match="only 0 are available"):
        with placement(pool, fraction=0.5):
            pass


def test_scope_cleared_after_exception():
    pool = _pool()
    with pytest.raises(RuntimeError, match="boom"):
        with placement(pool, fraction=1.0):
            raise RuntimeError("boom")
    assert current_placement() is None


_stub_counter = {"n": 0}


class _StubHandle(Handle):
    """Inherits from Handle so isinstance checks (e.g. _to_marker) match,
    but skips the real __init__ which does Ray RPC."""

    def __init__(self, role_cls, pool, *, device_ids, slot_id, role_name=None, init_kwargs=None):
        self.role_cls = role_cls
        self.device_ids = list(device_ids)
        self.slot_id = slot_id
        self.init_kwargs = init_kwargs or {}
        if role_name is None:
            role_name = f"{role_cls.__name__}_{_stub_counter['n']}"
            _stub_counter["n"] += 1
        self.role_name = role_name


def _patch_create_remote(monkeypatch):
    """Bypass Ray actor creation so create_remote can be exercised without a cluster."""
    monkeypatch.setattr(
        "diffusionrl.distributed.group.device_pool.DevicePool._get_or_create_worker",
        lambda self, device_id, slot: object(),
    )
    monkeypatch.setattr(
        "diffusionrl.distributed.group.device_pool.Handle",
        _StubHandle,
    )


def test_create_remote_inside_scope_uses_scope_args(monkeypatch):
    _patch_create_remote(monkeypatch)
    pool = _pool()
    with placement(pool, fraction=1.0):
        h = pool.create_remote(object)
    assert h.device_ids == list(range(8))
    assert h.slot_id == 0


def test_create_remote_inside_isolated_scope_bumps_slot(monkeypatch):
    _patch_create_remote(monkeypatch)
    pool = _pool()
    with placement(pool, fraction=1.0, shared_workers=False):
        h0 = pool.create_remote(object)
        h1 = pool.create_remote(object)
    assert h0.slot_id == 0
    assert h1.slot_id == 1
    assert h0.device_ids == h1.device_ids == list(range(8))


def test_create_remote_legacy_n_gpus_path_still_works(monkeypatch):
    _patch_create_remote(monkeypatch)
    pool = _pool()
    h = pool.create_remote(object, n_gpus=4)
    assert h.device_ids == [0, 1, 2, 3]
    assert h.slot_id == 0


def test_create_remote_legacy_device_ids_path_still_works(monkeypatch):
    _patch_create_remote(monkeypatch)
    pool = _pool()
    h = pool.create_remote(object, device_ids=[2, 3, 4])
    assert h.device_ids == [2, 3, 4]
    assert h.slot_id == 0


def test_create_remote_explicit_slot_id_overrides_scope(monkeypatch):
    _patch_create_remote(monkeypatch)
    pool = _pool()
    with placement(pool, fraction=1.0, shared_workers=False):
        h = pool.create_remote(object, slot_id=7)
    assert h.slot_id == 7


def test_bare_create_remote_uses_all_devices_slot_zero(monkeypatch):
    _patch_create_remote(monkeypatch)
    pool = _pool()
    h = pool.create_remote(object)
    assert h.device_ids == list(range(8))
    assert h.slot_id == 0


def test_two_bare_create_remote_calls_colocate(monkeypatch):
    _patch_create_remote(monkeypatch)
    pool = _pool()
    a = pool.create_remote(object)
    b = pool.create_remote(object)
    assert a.device_ids == b.device_ids == list(range(8))
    assert a.slot_id == b.slot_id == 0


def test_bare_create_remote_does_not_touch_claimed(monkeypatch):
    _patch_create_remote(monkeypatch)
    pool = _pool()
    pool.create_remote(object)
    assert pool._claimed == set()


def test_placement_unblocked_after_bare_create_remote(monkeypatch):
    _patch_create_remote(monkeypatch)
    pool = _pool()
    pool.create_remote(object)
    with placement(pool, fraction=0.5) as scope:
        assert scope.devices == (0, 1, 2, 3)


def test_remote_inside_scope_creates_via_active_pool(monkeypatch):
    _patch_create_remote(monkeypatch)
    pool = _pool()
    with placement(pool, fraction=1.0):
        h = remote(object)
    assert h.device_ids == list(range(8))
    assert h.slot_id == 0


def test_remote_outside_scope_raises():
    with pytest.raises(RuntimeError, match="placement"):
        remote(object)


# ── Handle auto-resolution ─────────────────────────────────────────────────


def test_remote_substitutes_handle_with_handleref(monkeypatch):
    from diffusionrl.distributed.group.handle import HandleRef

    _patch_create_remote(monkeypatch)
    pool = _pool()
    with placement(pool, fraction=1.0):
        train = remote(object)
        rollout = remote(object, train=train)
    assert rollout.init_kwargs == {"train": HandleRef(role_name=train.role_name)}


def test_remote_passes_plain_kwargs_through(monkeypatch):
    _patch_create_remote(monkeypatch)
    pool = _pool()
    with placement(pool, fraction=1.0):
        h = remote(object, scale=3.0, name="foo")
    assert h.init_kwargs == {"scale": 3.0, "name": "foo"}


def test_remote_substitutes_handle_nested_inside_container_unchanged(monkeypatch):
    """Top-level _to_marker is shallow; nested dicts pass through and are
    resolved on the Worker side instead. This test confirms the driver-side
    shallow conversion contract (Worker handles the recursion)."""

    _patch_create_remote(monkeypatch)
    pool = _pool()
    with placement(pool, fraction=1.0):
        train = remote(object)
        rollout = remote(object, deps={"train": train})
    # Driver passes the dict through verbatim; the inner Handle stays a Handle.
    deps = rollout.init_kwargs["deps"]
    assert isinstance(deps, dict)
    assert deps["train"] is train


def test_worker_resolves_handleref_to_local_remote():
    from diffusionrl.distributed.group.handle import HandleRef
    from diffusionrl.distributed.group.remote import RankInfo, Remote
    from diffusionrl.distributed.group.worker import Worker

    class FirstRemote(Remote):
        pass

    captured = {}

    class SecondRemote(Remote):
        def __init__(self, sibling):
            super().__init__()
            captured["sibling"] = sibling

    w = Worker.__new__(Worker)
    w._init_local(device_id=0, slot=0)

    w.add_remote("FirstRemote_0", FirstRemote, RankInfo(), init_kwargs={}, dist_env=None)
    first = w._roles["FirstRemote_0"]

    w.add_remote(
        "SecondRemote_0",
        SecondRemote,
        RankInfo(),
        init_kwargs={"sibling": HandleRef(role_name="FirstRemote_0")},
        dist_env=None,
    )

    assert captured["sibling"] is first


def test_worker_resolves_handleref_inside_nested_container():
    from diffusionrl.distributed.group.handle import HandleRef
    from diffusionrl.distributed.group.remote import RankInfo, Remote
    from diffusionrl.distributed.group.worker import Worker

    class A(Remote):
        pass

    captured = {}

    class B(Remote):
        def __init__(self, deps):
            super().__init__()
            captured["deps"] = deps

    w = Worker.__new__(Worker)
    w._init_local(device_id=0, slot=0)
    w.add_remote("A_0", A, RankInfo(), init_kwargs={}, dist_env=None)

    w.add_remote(
        "B_0",
        B,
        RankInfo(),
        init_kwargs={"deps": {"a": HandleRef(role_name="A_0"), "other": 42}},
        dist_env=None,
    )

    assert captured["deps"]["a"] is w._roles["A_0"]
    assert captured["deps"]["other"] == 42


def test_worker_raises_clearly_when_sibling_missing():
    from diffusionrl.distributed.group.handle import HandleRef
    from diffusionrl.distributed.group.remote import RankInfo, Remote
    from diffusionrl.distributed.group.worker import Worker

    class C(Remote):
        def __init__(self, sibling):
            super().__init__()

    w = Worker.__new__(Worker)
    w._init_local(device_id=0, slot=0)

    with pytest.raises(RuntimeError, match="MissingRole"):
        w.add_remote(
            "C_0",
            C,
            RankInfo(),
            init_kwargs={"sibling": HandleRef(role_name="MissingRole")},
            dist_env=None,
        )


def test_worker_passes_plain_config_through_unchanged():
    from diffusionrl.distributed.group.remote import RankInfo, Remote
    from diffusionrl.distributed.group.worker import Worker

    captured = {}

    class D(Remote):
        def __init__(self, scale, tags):
            super().__init__()
            captured["scale"] = scale
            captured["tags"] = tags

    w = Worker.__new__(Worker)
    w._init_local(device_id=0, slot=0)

    w.add_remote(
        "D_0",
        D,
        RankInfo(),
        init_kwargs={"scale": 3.0, "tags": ["a", "b"]},
        dist_env=None,
    )

    assert captured == {"scale": 3.0, "tags": ["a", "b"]}
