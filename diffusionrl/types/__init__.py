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
    "BufferedTrainingPayload": ("diffusionrl.types.buffer_contracts", "BufferedTrainingPayload"),
    "RewardRequest": ("diffusionrl.types.reward", "RewardRequest"),
    "RewardResponse": ("diffusionrl.types.reward", "RewardResponse"),
    "RewardType": ("diffusionrl.types.reward", "RewardType"),
    "EngineConfig": ("diffusionrl.types.engine", "EngineConfig"),
    "ForwardContext": ("diffusionrl.types.forward_context", "ForwardContext"),
    "FluxForwardContext": ("diffusionrl.types.forward_context", "FluxForwardContext"),
    "SD3ForwardContext": ("diffusionrl.types.forward_context", "SD3ForwardContext"),
    "HunyuanVideoForwardContext": ("diffusionrl.types.forward_context", "HunyuanVideoForwardContext"),
    "MochiForwardContext": ("diffusionrl.types.forward_context", "MochiForwardContext"),
    "WAN21ForwardContext": ("diffusionrl.types.forward_context", "WAN21ForwardContext"),
    "DefaultForwardContext": ("diffusionrl.types.forward_context", "DefaultForwardContext"),
    "get_forward_context_cls": ("diffusionrl.types.forward_context", "get_forward_context_cls"),
    "register_forward_context": ("diffusionrl.types.forward_context", "register_forward_context"),
    "RolloutPayload": ("diffusionrl.types.buffer_contracts", "RolloutPayload"),
    "RolloutRequest": ("diffusionrl.types.request", "RolloutRequest"),
    "RolloutResponse": ("diffusionrl.types.response", "RolloutResponse"),
    "RolloutResponseMeta": ("diffusionrl.types.response", "RolloutResponseMeta"),
    "RolloutSamples": ("diffusionrl.types.sample", "RolloutSamples"),
    "LogProbData": ("diffusionrl.types.sample", "LogProbData"),
    "SamplingParams": ("diffusionrl.types.sampling", "SamplingParams"),
    "SamplingRequirements": ("diffusionrl.types.sampling", "SamplingRequirements"),
    "SDEConfig": ("diffusionrl.types.sampling", "SDEConfig"),
    "GenerateGroup": ("diffusionrl.types.protocols", "GenerateGroup"),
    "TrajectoryStore": ("diffusionrl.types.trajectory_store", "TrajectoryStore"),
    "TimestepData": ("diffusionrl.types.training_batch", "TimestepData"),
    "TrainingBatch": ("diffusionrl.types.training_batch", "TrainingBatch"),
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
