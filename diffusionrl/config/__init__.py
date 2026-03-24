"""Public config package exports.

Keep this package entry lazy so low-level helper modules such as
``diffusionrl.config.argument_parsing`` can be imported without pulling in the
full config stack and creating package-level import cycles.
"""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    # argument surface
    "TrainingArguments": ("diffusionrl.config.arguments", "TrainingArguments"),
    "parse_args": ("diffusionrl.config.arguments", "parse_args"),
    "get_default_args": ("diffusionrl.config.arguments", "get_default_args"),
    "validate_args": ("diffusionrl.config.arguments", "validate_args"),
    "build_resolved_config_view": ("diffusionrl.config.arguments", "build_resolved_config_view"),
    "ResolvedRuntimeConfig": ("diffusionrl.config.runtime_bootstrap", "ResolvedRuntimeConfig"),
    "resolve_runtime_config": ("diffusionrl.config.runtime_bootstrap", "resolve_runtime_config"),
    "AlgorithmConfig": ("diffusionrl.config.arguments", "AlgorithmConfig"),
    "DebugConfig": ("diffusionrl.config.arguments", "DebugConfig"),
    "WindowSchedulerConfig": ("diffusionrl.config.arguments", "WindowSchedulerConfig"),
    "ModelConfig": ("diffusionrl.config.arguments", "ModelConfig"),
    "TrainingConfig": ("diffusionrl.config.arguments", "TrainingConfig"),
    "RolloutConfig": ("diffusionrl.config.arguments", "RolloutConfig"),
    "SamplingConfig": ("diffusionrl.config.arguments", "SamplingConfig"),
    "RewardConfig": ("diffusionrl.config.arguments", "RewardConfig"),
    "RayConfig": ("diffusionrl.config.arguments", "RayConfig"),
    # domain builders
    "build_model_config": ("diffusionrl.config.build_domain_args", "build_model_config"),
    "build_rollout_actor_init_config": (
        "diffusionrl.config.build_domain_args",
        "build_rollout_actor_init_config",
    ),
    "build_training_sampling_config": (
        "diffusionrl.config.build_domain_args",
        "build_training_sampling_config",
    ),
    "build_rollout_engine_config": (
        "diffusionrl.config.build_domain_args",
        "build_rollout_engine_config",
    ),
    "build_training_actor_init_config": (
        "diffusionrl.config.build_domain_args",
        "build_training_actor_init_config",
    ),
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
