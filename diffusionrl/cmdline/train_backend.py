"""Built-in train-backend cmdline adaptation helpers."""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from typing import Any, Callable, Dict

from diffusionrl.cmdline.registry import register_cmdline_config_parser
from diffusionrl.construction import ComponentInitPayload
from diffusionrl.training.backends.fsdp import FSDPBackend, FSDPBackendConfig
from diffusionrl.training.backends.veomni import VeOmniBackend, VeOmniBackendConfig
from diffusionrl.utils.dtypes import parse_torch_dtype


_SUPPORTED_BACKENDS = ("fsdp", "veomni")


@register_cmdline_config_parser(FSDPBackendConfig)
def build_fsdp_train_backend_config_from_args(args: Any) -> FSDPBackendConfig:
    extra = dict(getattr(args.training, "train_backend_kwargs", {}) or {})
    validate_train_backend_kwargs(
        config_class=FSDPBackendConfig,
        train_backend_kwargs=extra,
    )
    extra = _coerce_simple_dataclass_fields(FSDPBackendConfig, extra)
    # param_dtype may arrive as str in train_backend_kwargs; normalize to torch.dtype.
    extra_param_dtype = extra.pop("param_dtype", None)
    resolved_param_dtype = parse_torch_dtype(
        extra_param_dtype
        if extra_param_dtype is not None
        else str(args.precision.fsdp_precision),
        field_name="training.train_backend_kwargs.param_dtype",
    )
    fsdp_mode = str(getattr(args.training, "fsdp_mode", "full") or "full").strip().lower()
    reshard_after_forward = bool(getattr(args.training, "reshard_after_forward", True))
    return FSDPBackendConfig(
        cpu_offload=bool(args.training.fsdp_cpu_offload),
        param_dtype=resolved_param_dtype,
        fsdp_mode=fsdp_mode,
        reshard_after_forward=reshard_after_forward,
        **extra,
    )


@register_cmdline_config_parser(VeOmniBackendConfig)
def build_veomni_train_backend_config_from_args(
    args: Any,
) -> VeOmniBackendConfig:
    extra = dict(getattr(args.training, "train_backend_kwargs", {}) or {})
    validate_train_backend_kwargs(
        config_class=VeOmniBackendConfig,
        train_backend_kwargs=extra,
    )
    extra = _coerce_simple_dataclass_fields(VeOmniBackendConfig, extra)
    parallelize_kwargs = extra.get("parallelize_kwargs")
    if parallelize_kwargs is not None and not isinstance(parallelize_kwargs, dict):
        raise TypeError(
            "training.train_backend_kwargs.parallelize_kwargs must be a dict when provided."
        )
    if parallelize_kwargs is not None:
        extra["parallelize_kwargs"] = dict(parallelize_kwargs)
    return VeOmniBackendConfig(**extra)


_PARSER_DISPATCH: Dict[str, Callable[[Any], Any]] = {
    "fsdp": build_fsdp_train_backend_config_from_args,
    "veomni": build_veomni_train_backend_config_from_args,
}


_CLASS_DOTPATH_DISPATCH: Dict[str, str] = {
    "fsdp": f"{FSDPBackend.__module__}.{FSDPBackend.__qualname__}",
    "veomni": f"{VeOmniBackend.__module__}.{VeOmniBackend.__qualname__}",
}


def build_train_backend_init_payload_from_args(args: Any) -> ComponentInitPayload:
    identifier = resolve_train_backend_identifier(args)
    parser_fn = _PARSER_DISPATCH.get(identifier)
    if parser_fn is None:
        raise ValueError(
            f"Unsupported train_backend={identifier!r}. "
            f"Supported built-in backends: {list(_SUPPORTED_BACKENDS)}."
        )
    component_config = parser_fn(args)
    return ComponentInitPayload(
        component_dotpath=_CLASS_DOTPATH_DISPATCH[identifier],
        component_config=component_config,
    )


def validate_train_backend_kwargs(
    *,
    config_class: type,
    train_backend_kwargs: Dict[str, Any],
) -> None:
    config_field_names = {field.name for field in dataclass_fields(config_class)}
    unknown = sorted(
        key for key in train_backend_kwargs.keys() if key not in config_field_names
    )
    if unknown:
        raise ValueError(
            "training.train_backend_kwargs contains unsupported keys for "
            f"{config_class.__name__}: {unknown}. "
            f"Allowed backend-specific keys: {sorted(config_field_names)}."
        )


def _resolve_backend_name_from_args(args: Any) -> str:
    return str(args.training.train_backend or "fsdp").strip().lower()


def resolve_train_backend_identifier(args: Any) -> str:
    """Return the backend identifier from args (one of the supported names)."""
    backend_name = _resolve_backend_name_from_args(args)
    if backend_name == "megatron":
        raise ValueError(
            "Megatron backend was removed in the backends migration "
            "(the old implementation was a scaffold that never had a runtime "
            "path). If you need Megatron, plug a custom backend via a future "
            "extension mechanism."
        )
    if backend_name not in _SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported train_backend={backend_name!r}. "
            f"Supported: {list(_SUPPORTED_BACKENDS)}."
        )
    return backend_name


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
    "resolve_train_backend_identifier",
    "validate_train_backend_kwargs",
]
