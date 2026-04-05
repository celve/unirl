from __future__ import annotations

import pytest

from diffusionrl.sde.kernels import CPSSDEStrategy, DPM2Strategy, FlowSDEStrategy
from diffusionrl.sde.registry import resolve_sde_strategy_class


def test_resolve_sde_strategy_class_by_registered_name() -> None:
    assert resolve_sde_strategy_class("flow") is FlowSDEStrategy
    assert resolve_sde_strategy_class("cps") is CPSSDEStrategy
    assert resolve_sde_strategy_class("dpm2") is DPM2Strategy


def test_resolve_sde_strategy_class_by_dotpath() -> None:
    resolved = resolve_sde_strategy_class("diffusionrl.sde.kernels.FlowSDEStrategy")
    assert resolved is FlowSDEStrategy


def test_resolve_sde_strategy_class_invalid_name_lists_available() -> None:
    with pytest.raises(ValueError, match="Available registered names"):
        resolve_sde_strategy_class("missing_strategy")
