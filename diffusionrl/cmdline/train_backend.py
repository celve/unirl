"""Built-in train-backend cmdline adaptation helpers."""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from typing import Any, Dict, Optional

from diffusionrl.cmdline.construction import build_component_init_payload_from_args
from diffusionrl.cmdline.registry import register_cmdline_config_parser
from diffusionrl.construction import ComponentInitPayload
from diffusionrl.training.backends.fsdp import FSDPTrainBackendConfig
from diffusionrl.training.backends.megatron import MegatronTrainBackendConfig
from diffusionrl.training.backends.registry import (
    TRAIN_BACKEND_COMPONENT_FAMILY,
    supported_train_backends,
)
from diffusionrl.training.backends.veomni_native import VeOmniTrainBackendConfig


@register_cmdline_config_parser(FSDPTrainBackendConfig)
def build_fsdp_train_backend_config_from_args(args: Any) -> FSDPTrainBackendConfig:
    extra = dict(getattr(args.training, "train_backend_kwargs", {}) or {})
    validate_train_backend_kwargs(
        config_class=FSDPTrainBackendConfig,
        train_backend_kwargs=extra,
    )
    extra = _coerce_simple_dataclass_fields(FSDPTrainBackendConfig, extra)
    # Inject fsdp_mode from dedicated CLI arg (--training.fsdp-mode)
    fsdp_mode = str(getattr(args.training, "fsdp_mode", "full") or "full").strip().lower()
    reshard_after_forward = bool(getattr(args.training, "reshard_after_forward", True))
    return FSDPTrainBackendConfig(
        backend_dotpath=_resolve_backend_dotpath_from_args(args),
        cpu_offload=bool(args.training.fsdp_cpu_offload),
        param_dtype=str(args.precision.fsdp_precision),
        fsdp_mode=fsdp_mode,
        reshard_after_forward=reshard_after_forward,
        **extra,
    )


@register_cmdline_config_parser(MegatronTrainBackendConfig)
def build_megatron_train_backend_config_from_args(
    args: Any,
) -> MegatronTrainBackendConfig:
    extra = dict(getattr(args.training, "train_backend_kwargs", {}) or {})
    validate_train_backend_kwargs(
        config_class=MegatronTrainBackendConfig,
        train_backend_kwargs=extra,
    )
    extra = _coerce_simple_dataclass_fields(MegatronTrainBackendConfig, extra)
    runtime_env = extra.get("runtime_env")
    actor_kwargs = extra.get("actor_kwargs")
    if runtime_env is not None and not isinstance(runtime_env, dict):
        raise TypeError(
            "training.train_backend_kwargs.runtime_env must be a dict when provided."
        )
    if actor_kwargs is not None and not isinstance(actor_kwargs, dict):
        raise TypeError(
            "training.train_backend_kwargs.actor_kwargs must be a dict when provided."
        )
    if runtime_env is not None:
        extra["runtime_env"] = dict(runtime_env)
    if actor_kwargs is not None:
        extra["actor_kwargs"] = dict(actor_kwargs)
    return MegatronTrainBackendConfig(
        backend_dotpath=_resolve_backend_dotpath_from_args(args),
        **extra,
    )


@register_cmdline_config_parser(VeOmniTrainBackendConfig)
def build_veomni_train_backend_config_from_args(
    args: Any,
) -> VeOmniTrainBackendConfig:
    extra = dict(getattr(args.training, "train_backend_kwargs", {}) or {})
    validate_train_backend_kwargs(
        config_class=VeOmniTrainBackendConfig,
        train_backend_kwargs=extra,
    )
    extra = _coerce_simple_dataclass_fields(VeOmniTrainBackendConfig, extra)
    parallelize_kwargs = extra.get("parallelize_kwargs")
    if parallelize_kwargs is not None and not isinstance(parallelize_kwargs, dict):
        raise TypeError(
            "training.train_backend_kwargs.parallelize_kwargs must be a dict when provided."
        )
    if parallelize_kwargs is not None:
        extra["parallelize_kwargs"] = dict(parallelize_kwargs)
    return VeOmniTrainBackendConfig(
        backend_dotpath=_resolve_backend_dotpath_from_args(args),
        **extra,
    )


def build_train_backend_init_payload_from_args(args: Any) -> ComponentInitPayload:
    backend_identifier = _resolve_train_backend_identifier_from_args(args)
    return build_component_init_payload_from_args(
        component_family=TRAIN_BACKEND_COMPONENT_FAMILY,
        identifier=backend_identifier,
        args=args,
    )


def validate_train_backend_kwargs(
    *,
    config_class: type,
    train_backend_kwargs: Dict[str, Any],
) -> None:
    reserved_fields = {"name", "backend_dotpath"}
    config_field_names = {field.name for field in dataclass_fields(config_class)}
    allowed_extension_fields = config_field_names - reserved_fields
    unknown = sorted(
        key
        for key in train_backend_kwargs.keys()
        if key not in allowed_extension_fields
    )
    if unknown:
        raise ValueError(
            "training.train_backend_kwargs contains unsupported keys for "
            f"{config_class.__name__}: {unknown}. "
            f"Allowed backend-specific keys: {sorted(allowed_extension_fields)}."
        )


## -------- Helper functions --------
def _resolve_backend_name_from_args(args: Any) -> str:
    return str(args.training.train_backend or "fsdp").strip().lower()


def _resolve_backend_dotpath_from_args(args: Any) -> Optional[str]:
    backend_dotpath = str(args.training.train_backend_dotpath or "").strip()
    return backend_dotpath or None


def _resolve_train_backend_identifier_from_args(args: Any) -> str:
    backend_dotpath = _resolve_backend_dotpath_from_args(args)
    backend_name = _resolve_backend_name_from_args(args)
    if backend_dotpath is None and backend_name not in supported_train_backends():
        raise ValueError(
            f"Unsupported train_backend={backend_name!r}. "
            f"Expected one of {list(supported_train_backends())} or provide training.train_backend_dotpath."
        )
    return backend_dotpath or backend_name


def _coerce_simple_dataclass_fields(
    config_class: type, raw: Dict[str, Any]
) -> Dict[str, Any]:
    import typing

    simple_types = {int, float, bool, str}
    hints = typing.get_type_hints(config_class)
    field_name_to_type = {
        name: hint for name, hint in hints.items()
        if hint in simple_types
    }
    coerced: Dict[str, Any] = {}
    for key, value in raw.items():
        field_type = field_name_to_type.get(key)
        if field_type is None:
            coerced[key] = value
            continue
        coerced[key] = field_type(value)
    return coerced


__all__ = [
    "build_train_backend_init_payload_from_args",
    "validate_train_backend_kwargs",
]
