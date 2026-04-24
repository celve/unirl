"""Args-agnostic framework helper constants and rollout mode utilities."""

from __future__ import annotations

import logging
from typing import Dict

from diffusionrl.samplers import derive_rollout_engine_class
from diffusionrl.types.engine import ROLLOUT_ENGINE_TYPES

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "diffusionrl.models.hunyuan.HunyuanModelBundle"
DEFAULT_SAMPLER_PATH = "diffusionrl.samplers.fsdp.hunyuan_sampler.FSDPHunyuanSampler"
DIRECT_ROLLOUT_MODE = "direct_sampling"
SEPARATE_ROLLOUT_MODE = "separate"
COLOCATE_ROLLOUT_MODE = "colocate"

ROLLOUT_MODES = {DIRECT_ROLLOUT_MODE, SEPARATE_ROLLOUT_MODE, COLOCATE_ROLLOUT_MODE}


def rollout_mode_is_colocated(mode: str) -> bool:
    return mode == COLOCATE_ROLLOUT_MODE


def load_engine_capabilities(engine_type: str) -> Dict[str, bool]:
    """Derive engine capabilities from engine class declaration."""
    engine_cls = derive_rollout_engine_class(engine_type)
    declared = getattr(engine_cls, "declared_capabilities", None)
    if not callable(declared):
        raise ValueError(f"Engine class {engine_cls} must define classmethod declared_capabilities().")
    return dict(declared())


__all__ = [
    "COLOCATE_ROLLOUT_MODE",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_SAMPLER_PATH",
    "DIRECT_ROLLOUT_MODE",
    "ROLLOUT_ENGINE_TYPES",
    "ROLLOUT_MODES",
    "SEPARATE_ROLLOUT_MODE",
    "load_engine_capabilities",
    "rollout_mode_is_colocated",
]
