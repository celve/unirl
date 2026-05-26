"""
Cross-module data types for diffusionrl.

This package provides shared dataclasses and validation helpers used by:
- rollout control-plane
- ray actors
- samplers and losses
"""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    "RewardRequest": ("diffusionrl.types.reward", "RewardRequest"),
    "RewardResponse": ("diffusionrl.types.reward", "RewardResponse"),
    "RewardType": ("diffusionrl.types.reward", "RewardType"),
    "EngineConfig": ("diffusionrl.types.engine", "EngineConfig"),
    "MediaPreview": ("diffusionrl.types.media_preview", "MediaPreview"),
    "RolloutReq": ("diffusionrl.types.rollout_req", "RolloutReq"),
    "RolloutResp": ("diffusionrl.types.rollout_resp", "RolloutResp"),
    "RolloutTrack": ("diffusionrl.types.rollout_resp", "RolloutTrack"),
    "ARSamplingParams": ("diffusionrl.types.sampling", "ARSamplingParams"),
    "BaseSamplingParams": ("diffusionrl.types.sampling", "BaseSamplingParams"),
    "ComposedSamplingParams": ("diffusionrl.types.sampling", "ComposedSamplingParams"),
    "DiffusionSamplingParams": ("diffusionrl.types.sampling", "DiffusionSamplingParams"),
    "SamplingRequirements": ("diffusionrl.types.sampling", "SamplingRequirements"),
    "get_diffusion_params": ("diffusionrl.types.sampling", "get_diffusion_params"),
    "TrajectoryStore": ("diffusionrl.types.trajectory_store", "TrajectoryStore"),
}

__all__ = list(_LAZY_ATTRS.keys())


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))
