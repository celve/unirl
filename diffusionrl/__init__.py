"""Canonical Python package for this repository (`diffusionrl`)."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

__version__ = "0.1.0"

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    # shared types
    "SampleStatus": ("diffusionrl.types", "SampleStatus"),
    "RolloutOutputType": ("diffusionrl.types", "RolloutOutput"),
    "RolloutRequest": ("diffusionrl.types", "RolloutRequest"),
    "RewardRequestType": ("diffusionrl.types", "RewardRequest"),
    "RewardResponseType": ("diffusionrl.types", "RewardResponse"),
    "TrainingBatch": ("diffusionrl.types", "TrainingBatch"),
    # samplers
    "BaseSampler": ("diffusionrl.samplers", "BaseSampler"),
    "RolloutOutput": ("diffusionrl.samplers", "RolloutOutput"),
    "TrajectoryReplaySampler": ("diffusionrl.samplers", "TrajectoryReplaySampler"),
    "FastVideoSampler": ("diffusionrl.samplers", "FastVideoSampler"),
    "FastVideoSamplerV2": ("diffusionrl.samplers", "FastVideoSamplerV2"),
    "compute_sde_log_prob": ("diffusionrl.samplers", "compute_sde_log_prob"),
    "get_sigma_schedule": ("diffusionrl.samplers", "get_sigma_schedule"),
    "sde_step_with_log_prob": ("diffusionrl.samplers", "sde_step_with_log_prob"),
    # reward workers
    "BaseRewardWorker": ("diffusionrl.reward", "BaseRewardWorker"),
    "RewardRequest": ("diffusionrl.types", "RewardRequest"),
    "RewardResponse": ("diffusionrl.types", "RewardResponse"),
    "RewardType": ("diffusionrl.types", "RewardType"),
    "LocalRewardWorker": ("diffusionrl.reward", "LocalRewardWorker"),
    "HTTPRewardWorker": ("diffusionrl.reward", "HTTPRewardWorker"),
    "RewardService": ("diffusionrl.reward", "RewardService"),
    # losses / algorithms / advantages
    "GRPOLoss": ("diffusionrl.losses", "GRPOLoss"),
    "BaseAlgorithm": ("diffusionrl.algorithms", "BaseAlgorithm"),
    "SamplingRequirements": ("diffusionrl.algorithms", "SamplingRequirements"),
    "GRPOAlgorithm": ("diffusionrl.algorithms", "GRPOAlgorithm"),
    "compute_advantage": ("diffusionrl.advantages", "compute_advantage"),
    # config / utils / models
    "TrainingArguments": ("diffusionrl.config", "TrainingArguments"),
    "parse_args": ("diffusionrl.config", "parse_args"),
    "get_default_args": ("diffusionrl.config", "get_default_args"),
    "load_function": ("diffusionrl.utils", "load_function"),
    "load_class": ("diffusionrl.utils", "load_class"),
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
