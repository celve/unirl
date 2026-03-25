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
    "EngineCapabilities": ("diffusionrl.types.engine", "EngineCapabilities"),
    "EngineConfig": ("diffusionrl.types.engine", "EngineConfig"),
    "RolloutSamples": ("diffusionrl.types.sampling", "RolloutSamples"),
    "RolloutRequest": ("diffusionrl.types.sampling", "RolloutRequest"),
    "LogProbData": ("diffusionrl.types.sampling", "LogProbData"),
    "PromptEmbeddings": ("diffusionrl.types.sampling", "PromptEmbeddings"),
    "RolloutPayload": ("diffusionrl.types.buffer_contracts", "RolloutPayload"),
    "SamplingRequirements": ("diffusionrl.types.sampling", "SamplingRequirements"),
    "ResolvedSamplingSpec": ("diffusionrl.types.sampling", "ResolvedSamplingSpec"),
    "SDEConfig": ("diffusionrl.types.sde", "SDEConfig"),
    "SDEScheduleConfig": ("diffusionrl.types.sde", "SDEScheduleConfig"),
    "BackwardTrainingBatch": (
        "diffusionrl.types.training_batch",
        "BackwardTrainingBatch",
    ),
    "ForwardTrainingBatch": (
        "diffusionrl.types.training_batch",
        "ForwardTrainingBatch",
    ),
    "TimestepData": ("diffusionrl.types.training_batch", "TimestepData"),
    "TrainingBatch": ("diffusionrl.types.training_batch", "TrainingBatch"),
    "is_backward_batch": (
        "diffusionrl.types.training_batch",
        "is_backward_batch",
    ),
    "is_forward_batch": ("diffusionrl.types.training_batch", "is_forward_batch"),
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
