"""Tests for ``diffusionrl.config.instantiate.freeze``.

The walker applies ``OmegaConf.set_readonly`` per node based on the schema
class registered for that node. ``mutable=True`` schemas stay writable;
everything else is sealed. These tests cover:

- Direct decorator behavior with isolated test schemas (no Hydra compose).
- Real ``conf/train.yaml`` composition: ``ResumeConfig`` writable, others sealed.
- Hydra CLI override semantics still apply (``freeze`` runs *after* compose).
- Nested registered children honor their own flag, independent of the parent's.
- Lists of registered DictConfigs are recursed into per-item.
"""

# Note: deliberately omitting ``from __future__ import annotations`` so the
# function-local test schemas below have real (eager) annotations. OmegaConf
# resolves nested-dataclass refs via ``get_type_hints`` against the dataclass's
# ``__module__`` globals, which doesn't see function-local classes; eager
# annotations side-step that.

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from omegaconf.errors import ReadonlyConfigError

# Trigger @register_config side effects on the real schemas.
import diffusionrl  # noqa: F401
from diffusionrl.algorithms import grpo as _grpo  # noqa: F401
from diffusionrl.algorithms import nft as _nft  # noqa: F401
from diffusionrl.config import debug_config as _debug_config  # noqa: F401
from diffusionrl.config import evaluation_config as _evaluation_config  # noqa: F401
from diffusionrl.config import logging_config as _logging_config  # noqa: F401
from diffusionrl.config import resume_config as _resume_config  # noqa: F401
from diffusionrl.config import run_config as _run_config  # noqa: F401
from diffusionrl.config.instantiate import freeze
from diffusionrl.config.registration import register_config
from diffusionrl.distributed import weight_sync as _weight_sync  # noqa: F401
from diffusionrl.models import flux as _flux  # noqa: F401
from diffusionrl.ray import placement as _placement  # noqa: F401
from diffusionrl.ray.train_actor import TrainingExecutionConfig  # noqa: F401
from diffusionrl.reward import config as _reward_config  # noqa: F401
from diffusionrl.samplers.fsdp import engine as _fsdp_engine  # noqa: F401
from diffusionrl.samplers.sglang import engine as _sglang_engine  # noqa: F401
from diffusionrl.training.backends import fsdp as _fsdp  # noqa: F401
from diffusionrl.training.backends import veomni as _veomni  # noqa: F401
from diffusionrl.types import sampling as _sampling  # noqa: F401

_CONF_DIR = str(Path(__file__).resolve().parent.parent / "conf")


# ---------------------------------------------------------------------------
# Isolated schema fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def unique_prefix() -> Iterator[str]:
    prefix = f"freeze_test_{uuid.uuid4().hex[:8]}"
    yield prefix
    repo = ConfigStore.instance().repo
    for key in [k for k in repo if isinstance(k, str) and k.startswith(prefix)]:
        del repo[key]


# ---------------------------------------------------------------------------
# Direct walker behavior on isolated schemas
# ---------------------------------------------------------------------------


def test_default_seals_registered_node(unique_prefix):
    @register_config(group=unique_prefix, name="schema")
    @dataclass
    class _Immut:
        x: int = 1

    cfg = OmegaConf.structured(_Immut)
    freeze(cfg)

    assert OmegaConf.is_readonly(cfg) is True
    with pytest.raises(ReadonlyConfigError):
        cfg.x = 2


def test_mutable_keeps_node_writable(unique_prefix):
    @register_config(group=unique_prefix, name="schema", mutable=True)
    @dataclass
    class _Mut:
        x: int = 1

    cfg = OmegaConf.structured(_Mut)
    freeze(cfg)

    assert OmegaConf.is_readonly(cfg) is False
    cfg.x = 7
    assert cfg.x == 7


def test_mutable_parent_with_immutable_child_seals_only_the_child(unique_prefix):
    @register_config(group=unique_prefix, name="child")
    @dataclass
    class _Child:
        z: int = 0

    @register_config(group=unique_prefix, name="parent", mutable=True)
    @dataclass
    class _Parent:
        a: int = 5
        child: _Child = field(default_factory=_Child)

    cfg = OmegaConf.structured(_Parent)
    freeze(cfg)

    # Parent's own field is writable.
    cfg.a = 9
    assert cfg.a == 9

    # Child schema seals despite mutable parent.
    assert OmegaConf.is_readonly(cfg.child) is True
    with pytest.raises(ReadonlyConfigError):
        cfg.child.z = 99


def test_immutable_parent_with_mutable_child_seals_only_the_parent(unique_prefix):
    @register_config(group=unique_prefix, name="child", mutable=True)
    @dataclass
    class _Child:
        z: int = 0

    @register_config(group=unique_prefix, name="parent")
    @dataclass
    class _Parent:
        a: int = 5
        child: _Child = field(default_factory=_Child)

    cfg = OmegaConf.structured(_Parent)
    freeze(cfg)

    # Mutable child stays writable.
    cfg.child.z = 42
    assert cfg.child.z == 42

    # Parent's primitive field is sealed.
    with pytest.raises(ReadonlyConfigError):
        cfg.a = 9


def test_list_of_registered_items_each_get_their_own_flag(unique_prefix):
    @register_config(group=unique_prefix, name="item")
    @dataclass
    class _Item:
        v: int = 0

    @register_config(group=unique_prefix, name="parent")
    @dataclass
    class _Parent:
        entries: List[_Item] = field(default_factory=lambda: [_Item(v=1), _Item(v=2)])

    cfg = OmegaConf.structured(_Parent)
    freeze(cfg)

    for entry in cfg.entries:
        assert OmegaConf.is_readonly(entry) is True
        with pytest.raises(ReadonlyConfigError):
            entry.v = 999


def test_optional_none_value_is_skipped(unique_prefix):
    @register_config(group=unique_prefix, name="child")
    @dataclass
    class _Child:
        z: int = 0

    @register_config(group=unique_prefix, name="parent")
    @dataclass
    class _Parent:
        child: Optional[_Child] = None

    cfg = OmegaConf.structured(_Parent)
    freeze(cfg)

    # No crash; parent itself is sealed.
    assert OmegaConf.is_readonly(cfg) is True


def test_unregistered_inline_subsection_seals_by_default():
    cfg = OmegaConf.create({"foo": {"bar": 1}, "baz": [1, 2, 3]})
    freeze(cfg)

    assert OmegaConf.is_readonly(cfg) is True
    assert OmegaConf.is_readonly(cfg.foo) is True
    with pytest.raises(ReadonlyConfigError):
        cfg.foo.bar = 99


def test_mutable_registry_records_class():
    """Sanity: live mutable schemas in production carry the class attribute."""
    from diffusionrl.config.resume_config import ResumeConfig
    from diffusionrl.models.config import ModelBundleConfig

    assert getattr(ResumeConfig, "_hydra_mutable_", False) is True
    assert getattr(ModelBundleConfig, "_hydra_mutable_", False) is True


# ---------------------------------------------------------------------------
# Integration: real conf/train.yaml composition
# ---------------------------------------------------------------------------


@pytest.fixture
def hydra_context():
    with initialize_config_dir(config_dir=_CONF_DIR, version_base=None):
        yield


def test_real_cfg_resume_is_mutable_after_freeze(hydra_context):
    cfg = compose(config_name="train")
    OmegaConf.resolve(cfg)
    freeze(cfg)

    cfg.resume.start_rollout_id = 7
    assert cfg.resume.start_rollout_id == 7


def test_real_cfg_model_is_mutable_after_freeze(hydra_context):
    """``ModelBundleConfig`` is mutable so ``TrainActor`` can inject runtime fields."""
    cfg = compose(config_name="train")
    OmegaConf.resolve(cfg)
    freeze(cfg)

    cfg.model.device = "cuda:0"
    cfg.model.training_only = True
    assert cfg.model.device == "cuda:0"
    assert cfg.model.training_only is True


def test_real_cfg_other_sections_are_sealed_after_freeze(hydra_context):
    cfg = compose(config_name="train")
    OmegaConf.resolve(cfg)
    freeze(cfg)

    # Pick a few representative sections; they should all seal. ``model`` and
    # ``resume`` are mutable by design; pick sections that aren't.
    with pytest.raises(ReadonlyConfigError):
        cfg.training.plan.global_batch_size = 999
    with pytest.raises(ReadonlyConfigError):
        cfg.run.weight_sync_interval = 0


def test_root_cfg_is_sealed_after_freeze(hydra_context):
    cfg = compose(config_name="train")
    OmegaConf.resolve(cfg)
    freeze(cfg)

    # Reassigning a top-level section must fail (root is unregistered, defaults sealed).
    with pytest.raises(ReadonlyConfigError):
        cfg.resume = None


def test_hydra_cli_overrides_survive_freeze(hydra_context):
    """Compose with an override, then freeze; the override must stick."""
    cfg = compose(
        config_name="train",
        overrides=["model.lora_rank=64"],
    )
    OmegaConf.resolve(cfg)
    freeze(cfg)

    assert cfg.model.lora_rank == 64
