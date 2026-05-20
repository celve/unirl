"""Unit tests for :func:`diffusionrl.training_new.policy.compose_policy`.

Exercises the stack-builder logic in isolation: each test uses
hand-rolled Policy classes (and a fake source) so the assertions are
about composition order and error handling, not real LoRA / FSDP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import pytest
from torch import nn

from diffusionrl.config.registration import register_config
from diffusionrl.training_new.policy import (
    PolicyBase,
    compose_policy,
    walk_source_chain,
)

# ---------------------------------------------------------------------------
# Test scaffolding: registered fake Policy classes for compose_policy
# ---------------------------------------------------------------------------


# Use ``__name__`` so the ``_target_`` string resolves back to *this* module
# regardless of whether pytest imports it as ``test_compose_policy`` or
# ``tests.test_compose_policy``. Otherwise Hydra's get_method() re-imports
# under the dotted name and produces a *different* class object than the one
# the test file's local namespace holds, breaking ``isinstance`` checks.
@register_config(
    group="training/policy",
    name="_compose_test_a",
    target=f"{__name__}._FakePolicyA",
)
@dataclass
class _FakePolicyAConfig:
    name: ClassVar[str] = "_compose_test_a"
    label: str = "A"


@register_config(
    group="training/policy",
    name="_compose_test_b",
    target=f"{__name__}._FakePolicyB",
)
@dataclass
class _FakePolicyBConfig:
    name: ClassVar[str] = "_compose_test_b"
    label: str = "B"


@dataclass
class _UnregisteredConfig:
    """Plain dataclass without the @register_config decorator — has no
    ``_target_`` attribute, so compose_policy should reject it."""

    label: str = "unregistered"


class _FakePolicyA(PolicyBase):
    def __init__(self, config: _FakePolicyAConfig, source: Any) -> None:
        self.config = config
        self.source = source
        self.model = source.trainable_module() if hasattr(source, "trainable_module") else source


class _FakePolicyB(PolicyBase):
    def __init__(self, config: _FakePolicyBConfig, source: Any) -> None:
        self.config = config
        self.source = source
        self.model = source.trainable_module() if hasattr(source, "trainable_module") else source


class _FakeStage(nn.Module):
    """Minimal Stage-shaped object: an nn.Module with ``trainable_module()``."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def trainable_module(self) -> nn.Module:
        return self


# ---------------------------------------------------------------------------
# compose_policy
# ---------------------------------------------------------------------------


def test_compose_policy_empty_configs_returns_source_unchanged():
    stage = _FakeStage()
    out = compose_policy(stage, [])
    assert out is stage


def test_compose_policy_single_config_wraps_once():
    stage = _FakeStage()
    cfg_a = _FakePolicyAConfig()
    out = compose_policy(stage, [cfg_a])
    assert isinstance(out, _FakePolicyA)
    assert out.source is stage
    assert out.model is stage  # source.trainable_module() == stage itself


def test_compose_policy_stacks_inside_out():
    """First config is innermost, last is outermost.

    Stack: ``compose_policy(stage, [A_cfg, B_cfg])`` →
    ``B(A(stage))``. Walking source chain from the outer handle yields
    ``B → A → stage`` in that order.
    """

    stage = _FakeStage()
    cfg_a = _FakePolicyAConfig()
    cfg_b = _FakePolicyBConfig()

    out = compose_policy(stage, [cfg_a, cfg_b])

    assert isinstance(out, _FakePolicyB)
    assert isinstance(out.source, _FakePolicyA)
    assert out.source.source is stage

    chain = list(walk_source_chain(out))
    assert isinstance(chain[0], _FakePolicyB)
    assert isinstance(chain[1], _FakePolicyA)
    assert chain[2] is stage


def test_compose_policy_three_configs_order():
    stage = _FakeStage()
    cfgs = [_FakePolicyAConfig(label="inner"), _FakePolicyBConfig(label="middle"), _FakePolicyAConfig(label="outer")]
    out = compose_policy(stage, cfgs)

    chain = list(walk_source_chain(out))
    # outermost first, innermost last
    assert chain[0].config.label == "outer"
    assert chain[1].config.label == "middle"
    assert chain[2].config.label == "inner"
    assert chain[3] is stage


def test_compose_policy_missing_target_raises_value_error():
    stage = _FakeStage()
    bad_cfg = _UnregisteredConfig()
    with pytest.raises(ValueError, match=r"_target_"):
        compose_policy(stage, [bad_cfg])


def test_compose_policy_passes_config_object_through_to_constructor():
    stage = _FakeStage()
    cfg_a = _FakePolicyAConfig(label="custom-label")
    out = compose_policy(stage, [cfg_a])
    assert out.config is cfg_a
    assert out.config.label == "custom-label"
