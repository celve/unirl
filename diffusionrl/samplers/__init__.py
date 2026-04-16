"""Sampler subsystem entrypoint with lazy exports."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

from diffusionrl.samplers.registry import ensure_builtin_rollout_engine_registration

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    "BaseSampler": ("diffusionrl.samplers.base", "BaseSampler"),
    "RolloutSamples": ("diffusionrl.samplers.base", "RolloutSamples"),
    "BaseRolloutEngine": ("diffusionrl.samplers.engine", "BaseRolloutEngine"),
    "build_rollout_engine_init_payload_from_args": (
        "diffusionrl.cmdline.rollout_engine",
        "build_rollout_engine_init_payload_from_args",
    ),
    "derive_rollout_engine_class": (
        "diffusionrl.samplers.registry",
        "derive_rollout_engine_class",
    ),
    "create_rollout_engine_from_init_payload": (
        "diffusionrl.samplers.construction",
        "create_rollout_engine_from_init_payload",
    ),
    "FluxSampler": ("diffusionrl.samplers.fsdp.flux_sampler", "FluxSampler"),
    "SD3Sampler": ("diffusionrl.samplers.fsdp.sd3_sampler", "SD3Sampler"),
    "FSDPHunyuanSampler": ("diffusionrl.samplers.fsdp.hunyuan_sampler", "FSDPHunyuanSampler"),
    "SGLangRolloutEngine": ("diffusionrl.samplers.sglang.engine", "SGLangRolloutEngine"),
}

ensure_builtin_rollout_engine_registration()

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
