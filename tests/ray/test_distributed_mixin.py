"""Tests for ``diffusionrl.ray.distributed.DistributedMixin``.

Validate the cooperative-super state install, the ``_write_distributed_env``
primitive, and MRO cooperation with ``ConfigActor``. No Ray required — the
mixin is plain Python.
"""

from __future__ import annotations

import os
from typing import Any, Iterator

import pytest
from omegaconf import OmegaConf

from diffusionrl.ray import actor_config
from diffusionrl.ray.actor_config import ConfigActor, current
from diffusionrl.ray.distributed import DistributedMixin

_DIST_ENV_KEYS = ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK")


@pytest.fixture(autouse=True)
def reset_state() -> Iterator[None]:
    """Clear ContextVar + env vars before and after each test."""
    actor_config._current = None
    saved = {k: os.environ.pop(k, None) for k in _DIST_ENV_KEYS}
    yield
    actor_config._current = None
    for k in _DIST_ENV_KEYS:
        os.environ.pop(k, None)
        if saved.get(k) is not None:
            os.environ[k] = saved[k]


def test_init_installs_state_via_super_chain() -> None:
    class Actor(DistributedMixin):
        pass

    actor = Actor(world_size=4, rank=1, master_addr="10.0.0.1", master_port=29500)

    assert actor.world_size == 4
    assert actor.rank == 1
    assert actor.master_addr == "10.0.0.1"
    assert actor.master_port == 29500
    assert actor._is_distributed_initialized is False


def test_init_defaults_are_single_actor() -> None:
    class Actor(DistributedMixin):
        pass

    actor = Actor()
    assert actor.world_size == 1
    assert actor.rank == 0
    assert actor.master_addr is None
    assert actor.master_port is None


def test_write_distributed_env_sets_all_five_vars() -> None:
    class Actor(DistributedMixin):
        pass

    actor = Actor()
    actor._write_distributed_env(
        master_addr="host.example",
        master_port=29500,
        world_size=4,
        rank=2,
        local_rank=0,
    )

    assert os.environ["MASTER_ADDR"] == "host.example"
    assert os.environ["MASTER_PORT"] == "29500"
    assert os.environ["WORLD_SIZE"] == "4"
    assert os.environ["RANK"] == "2"
    assert os.environ["LOCAL_RANK"] == "0"


def test_set_master_info_roundtrips() -> None:
    class Actor(DistributedMixin):
        pass

    actor = Actor(world_size=2, rank=0)
    actor.set_master_info("host.example", 12345)

    assert actor.master_addr == "host.example"
    assert actor.master_port == 12345


def test_general_utilities_reflect_state() -> None:
    class Actor(DistributedMixin):
        pass

    actor = Actor(world_size=8, rank=3)
    assert actor.get_rank() == 3
    assert actor.get_world_size() == 8
    assert actor.health_check() is True


def test_mro_cooperation_with_config_actor() -> None:
    """ConfigActor runs first (installs cfg), then DistributedMixin installs state."""
    observed_cfg: list[Any] = []
    observed_rank: list[int] = []

    class Peeker:
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            observed_cfg.append(current())

    class Actor(ConfigActor, DistributedMixin, Peeker):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            observed_rank.append(self.rank)

    cfg = OmegaConf.create({"marker": "x"})
    actor = Actor(cfg=cfg, world_size=4, rank=2, master_addr="h", master_port=10)

    # ConfigActor ran before Peeker (cfg available when Peeker's __init__ ran)
    assert observed_cfg == [cfg]
    # DistributedMixin state installed by the time Actor.__init__'s body ran
    assert observed_rank == [2]
    assert actor.master_addr == "h"
    assert actor.master_port == 10


def test_actor_without_write_env_call_leaves_env_clean() -> None:
    """Sanity check: merely instantiating the mixin does not touch os.environ."""

    class Actor(DistributedMixin):
        pass

    Actor(world_size=2, rank=0, master_addr="h", master_port=10)

    for key in _DIST_ENV_KEYS:
        assert key not in os.environ
