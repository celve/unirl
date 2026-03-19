"""Canonical Python package for this repository (`diffusionrl`)."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

__version__ = "0.1.0"

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    # shared types
    "RolloutOutputType": ("diffusionrl.types", "RolloutOutput"),
    "RolloutRequest": ("diffusionrl.types", "RolloutRequest"),
    "RewardRequestType": ("diffusionrl.types", "RewardRequest"),
    "RewardResponseType": ("diffusionrl.types", "RewardResponse"),
    "TrainingBatch": ("diffusionrl.types", "TrainingBatch"),
    # samplers
    "BaseSampler": ("diffusionrl.samplers", "BaseSampler"),
    "RolloutOutput": ("diffusionrl.samplers", "RolloutOutput"),
    "compute_sde_log_prob": ("diffusionrl.sde", "compute_sde_log_prob"),
    "get_sigma_schedule": ("diffusionrl.sde", "get_sigma_schedule"),
    "sde_step_with_log_prob": ("diffusionrl.sde", "sde_step_with_log_prob"),
    "SDEConfig": ("diffusionrl.types", "SDEConfig"),
    "SDEScheduleConfig": ("diffusionrl.types", "SDEScheduleConfig"),
    # reward scorers
    "BaseRewardScorer": ("diffusionrl.reward", "BaseRewardScorer"),
    "RewardRequest": ("diffusionrl.types", "RewardRequest"),
    "RewardResponse": ("diffusionrl.types", "RewardResponse"),
    "RewardType": ("diffusionrl.types", "RewardType"),
    # algorithms / advantages
    "BaseAlgorithm": ("diffusionrl.algorithms", "BaseAlgorithm"),
    "SamplingRequirements": ("diffusionrl.types", "SamplingRequirements"),
    "GRPOAlgorithm": ("diffusionrl.algorithms", "GRPOAlgorithm"),
    "NFTAlgorithm": ("diffusionrl.algorithms", "NFTAlgorithm"),
    # config / utils / models
    "TrainingArguments": ("diffusionrl.config", "TrainingArguments"),
    "parse_args": ("diffusionrl.config", "parse_args"),
    "get_default_args": ("diffusionrl.config", "get_default_args"),
    "load_function": ("diffusionrl.utils", "load_function"),
    "set_seed": ("diffusionrl.utils", "set_seed"),
    "configure_logger": ("diffusionrl.utils", "configure_logger"),
    "ModelBundle": ("diffusionrl.models", "ModelBundle"),
}

__all__ = ["__version__", *_LAZY_ATTRS.keys()]


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
