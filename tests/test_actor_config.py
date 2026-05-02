"""Tests for ``diffusionrl.ray.actor_config``.

Validate the ambient cfg handle (``current()`` pre/post install) and the
cooperative-super ``ConfigActor`` base class. MRO cooperation is checked
against a synthetic class hierarchy that mirrors the real
``TrainActor`` / ``RolloutActor`` shape (multiple bases with and without
``__init__``).
"""

from __future__ import annotations

from typing import Any, Iterator

import pytest
from omegaconf import OmegaConf

from diffusionrl.ray import actor_config
from diffusionrl.ray.actor_config import ConfigActor, current


@pytest.fixture(autouse=True)
def reset_current() -> Iterator[None]:
    """Clear the module global before and after each test to avoid pollution."""
    actor_config._current = None
    yield
    actor_config._current = None


def test_current_raises_before_install() -> None:
    with pytest.raises(RuntimeError, match="before a ConfigActor installed"):
        current()


def test_config_actor_installs_cfg_into_module_and_self() -> None:
    cfg = OmegaConf.create({"foo": 1, "bar": {"baz": "qux"}})

    class Actor(ConfigActor):
        pass

    actor = Actor(cfg=cfg)

    assert current() is cfg
    assert actor._cfg is cfg
    assert current().bar.baz == "qux"


def test_config_actor_with_cfg_none_does_not_touch_module() -> None:
    class Actor(ConfigActor):
        pass

    Actor(cfg=None)

    with pytest.raises(RuntimeError):
        current()


def test_config_actor_forwards_kwargs_through_super_chain() -> None:
    """ConfigActor drains its own kwarg and forwards the rest up the chain."""

    class Base:
        def __init__(self, *, label: str, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.label = label

    class Actor(ConfigActor, Base):
        pass

    cfg = OmegaConf.create({"x": 42})
    actor = Actor(cfg=cfg, label="hello")

    assert current().x == 42
    assert actor.label == "hello"


def test_config_actor_runs_before_other_parents() -> None:
    """cfg must be installed before any other parent's __init__ runs.

    Mirrors the real invariant: ``DistributedMixin.__init__`` (and anything
    else down the chain) may read ``current()`` during its own setup, so
    ``ConfigActor`` must populate the module handle first.
    """
    seen: list[str] = []

    class Peeker:
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            seen.append(current().marker)

    class Actor(ConfigActor, Peeker):
        pass

    cfg = OmegaConf.create({"marker": "installed"})
    Actor(cfg=cfg)

    assert seen == ["installed"]


def test_config_actor_terminates_cleanly_at_object() -> None:
    """The cooperative chain must drain all kwargs before reaching object."""

    class LeafBase:
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.initialized = True

    class Actor(ConfigActor, LeafBase):
        pass

    cfg = OmegaConf.create({"a": 1})
    actor = Actor(cfg=cfg)

    assert actor.initialized is True


def test_helper_can_read_cfg_after_install() -> None:
    """A module-level helper simulates a deep consumer calling ``current()``."""

    def helper_reads_cfg() -> Any:
        return current().deep.knob

    class Actor(ConfigActor):
        pass

    cfg = OmegaConf.create({"deep": {"knob": "visible"}})
    Actor(cfg=cfg)

    assert helper_reads_cfg() == "visible"
