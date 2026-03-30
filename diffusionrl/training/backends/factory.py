"""Centralized train-backend factory with canonical config resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from diffusionrl.utils import load_function

from .base import TrainBackend, TrainBackendCapabilities, TrainBackendLaunchSpec
from .fsdp import FSDPTrainBackend
from .megatron import MegatronTrainBackend
from .veomni import VeOmniTrainBackend

_BUILTIN_BACKENDS: dict[str, type[TrainBackend]] = {
    "fsdp": FSDPTrainBackend,
    "veomni": VeOmniTrainBackend,
    "megatron": MegatronTrainBackend,
}

_FSDP_ALLOWED_KEYS = {
    "mixed_precision",
    "use_fsdp",
}

_VEOMNI_ALLOWED_KEYS = {
    "data_parallel_mode",
    "dp_size",
    "dp_replicate_size",
    "dp_shard_size",
    "tp_size",
    "pp_size",
    "sp_size",
    "ep_size",
    "cp_size",
    "enable_full_shard",
    "enable_reshard_after_forward",
    "enable_mixed_precision",
    "enable_gradient_checkpointing",
    "enable_reentrant",
    "enable_forward_prefetch",
    "enable_fsdp_offload",
    "init_device",
    "broadcast_model_weights_from_rank0",
    "basic_modules",
    "weights_path",
    "weights_path_mode",
    "optimizer_type",
    "fused_optimizer",
    "no_decay_modules",
    "no_decay_params",
    "lr_decay_ratio",
    "lr_min",
    "lr_start",
    "veomni_repo_path",
    "parallelize_kwargs",
}

_MEGATRON_ALLOWED_KEYS = {
    "actor_class_path",
    "dp_size",
    "tp_size",
    "pp_size",
    "sp_size",
    "ep_size",
    "num_gpus_per_actor",
    "runtime_env",
    "actor_kwargs",
}


@dataclass(frozen=True)
class TrainBackendConfig:
    name: str
    backend_dotpath: Optional[str] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": str(self.name),
            "backend_dotpath": self.backend_dotpath,
            "kwargs": dict(self.kwargs),
        }


def supported_train_backends() -> tuple[str, ...]:
    """Return built-in train backend names recognized by configuration."""
    return tuple(_BUILTIN_BACKENDS.keys())


def _resolve_backend_cls(
    name: str,
    *,
    backend_dotpath: Optional[str] = None,
) -> type[TrainBackend]:
    """Resolve a backend class from a built-in name or custom dotpath."""
    if backend_dotpath:
        backend_cls = load_function(backend_dotpath)
        if not isinstance(backend_cls, type) or not issubclass(backend_cls, TrainBackend):
            raise TypeError(
                f"train_backend_dotpath must resolve to a TrainBackend subclass, got: {backend_cls}"
            )
        return backend_cls

    backend_name = str(name or "fsdp").strip().lower()
    backend_cls = _BUILTIN_BACKENDS.get(backend_name)
    if backend_cls is None:
        raise ValueError(
            f"Unsupported train_backend={name!r}. "
            f"Expected one of {list(_BUILTIN_BACKENDS)} or provide train_backend_dotpath."
        )
    return backend_cls


def _resolve_backend_name_from_args(args: Any) -> str:
    return str(args.training.train_backend or "fsdp").strip().lower()


def _resolve_backend_dotpath_from_args(args: Any) -> Optional[str]:
    backend_dotpath = str(args.training.train_backend_dotpath or "").strip()
    return backend_dotpath or None


def _parse_backend_kwargs_from_args(args: Any) -> Dict[str, Any]:
    raw = args.training.train_backend_kwargs
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            "training.train_backend_kwargs must already be a dict after config parsing. "
            f"Got: {type(raw).__name__}."
        )
    return dict(raw)


def _require_positive_int_if_present(*, field_name: str, value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must be an integer >= 1 when provided. Got: {value!r}."
        ) from exc
    if resolved < 1:
        raise ValueError(f"{field_name} must be >= 1. Got: {resolved}.")
    return resolved


def _require_positive_number_if_present(*, field_name: str, value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must be a positive number when provided. Got: {value!r}."
        ) from exc
    if resolved <= 0:
        raise ValueError(f"{field_name} must be > 0. Got: {resolved}.")
    return resolved


def _raise_unknown_backend_kwargs(
    *,
    backend_name: str,
    unknown: list[str],
) -> None:
    if not unknown:
        return
    raise ValueError(
        f"training.train_backend_kwargs contains unsupported keys for train_backend={backend_name!r}: "
        f"{', '.join(sorted(unknown))}."
    )


def _raise_backend_duplicate_entry(*, backend_name: str, key: str, canonical_path: str) -> None:
    raise ValueError(
        f"training.train_backend_kwargs.{key} is no longer supported for train_backend={backend_name!r}. "
        f"Use {canonical_path} instead."
    )


def _resolve_fsdp_backend_kwargs(
    args: Any,
    raw_kwargs: Mapping[str, Any],
    *,
    strict_unknowns: bool,
) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(raw_kwargs)

    if "fsdp_config" in merged:
        raise ValueError(
            "training.train_backend_kwargs.fsdp_config has been removed. "
            "Pass supported built-in FSDP backend keys directly under training.train_backend_kwargs."
        )
    if "cpu_offload" in merged:
        _raise_backend_duplicate_entry(
            backend_name="fsdp",
            key="cpu_offload",
            canonical_path="training.fsdp_cpu_offload",
        )
    if "param_dtype" in merged:
        _raise_backend_duplicate_entry(
            backend_name="fsdp",
            key="param_dtype",
            canonical_path="precision.fsdp_precision",
        )

    if strict_unknowns:
        _raise_unknown_backend_kwargs(
            backend_name="fsdp",
            unknown=[key for key in merged.keys() if key not in _FSDP_ALLOWED_KEYS],
        )

    merged["cpu_offload"] = bool(args.training.fsdp_cpu_offload)
    merged["param_dtype"] = str(args.precision.fsdp_precision)
    if "use_fsdp" in merged:
        merged["use_fsdp"] = bool(merged["use_fsdp"])
    if "mixed_precision" in merged:
        merged["mixed_precision"] = bool(merged["mixed_precision"])
    return merged


def _resolve_veomni_backend_kwargs(
    args: Any,
    raw_kwargs: Mapping[str, Any],
    *,
    actor_count: int,
    strict_unknowns: bool,
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        "data_parallel_mode": "fsdp2",
        "dp_size": int(actor_count),
        "dp_replicate_size": 1,
        "dp_shard_size": int(actor_count),
        "tp_size": 1,
        "pp_size": 1,
        "sp_size": 1,
        "ep_size": 1,
        "enable_full_shard": True,
        "enable_reshard_after_forward": True,
        "enable_mixed_precision": bool(actor_count > 1),
        "enable_gradient_checkpointing": bool(args.training.use_gradient_checkpointing),
        "init_device": "meta",
        "broadcast_model_weights_from_rank0": True,
        "weights_path_mode": "transformer_subdir",
    }
    merged.update(dict(raw_kwargs))

    if strict_unknowns:
        _raise_unknown_backend_kwargs(
            backend_name="veomni",
            unknown=[key for key in merged.keys() if key not in _VEOMNI_ALLOWED_KEYS],
        )

    mode = str(merged.get("data_parallel_mode", "fsdp2") or "fsdp2").strip().lower()
    if mode != "fsdp2":
        raise ValueError(
            "train_backend=veomni in diffusionRL now targets FSDP2 only. "
            "Set training.train_backend_kwargs.data_parallel_mode='fsdp2' or omit this field."
        )
    merged["data_parallel_mode"] = mode

    for key in (
        "dp_size",
        "dp_replicate_size",
        "dp_shard_size",
        "tp_size",
        "pp_size",
        "sp_size",
        "ep_size",
        "cp_size",
    ):
        if key in merged:
            merged[key] = _require_positive_int_if_present(
                field_name=f"training.train_backend_kwargs.{key}",
                value=merged[key],
            )

    if merged.get("parallelize_kwargs") is not None and not isinstance(
        merged.get("parallelize_kwargs"), dict
    ):
        raise TypeError(
            "training.train_backend_kwargs.parallelize_kwargs must be a dict when provided."
        )
    return merged


def _resolve_megatron_backend_kwargs(
    raw_kwargs: Mapping[str, Any],
    *,
    strict_unknowns: bool,
) -> Dict[str, Any]:
    merged = dict(raw_kwargs)
    if "num_actors" in merged:
        raise ValueError(
            "training.train_backend_kwargs.num_actors is not supported. "
            "Training actor count is owned by ray.training_num_nodes × "
            "ray.training_num_gpus_per_node."
        )

    if strict_unknowns:
        _raise_unknown_backend_kwargs(
            backend_name="megatron",
            unknown=[key for key in merged.keys() if key not in _MEGATRON_ALLOWED_KEYS],
        )

    for key in ("dp_size", "tp_size", "pp_size", "sp_size", "ep_size"):
        if key in merged:
            merged[key] = _require_positive_int_if_present(
                field_name=f"training.train_backend_kwargs.{key}",
                value=merged[key],
            )

    if "num_gpus_per_actor" in merged:
        merged["num_gpus_per_actor"] = _require_positive_number_if_present(
            field_name="training.train_backend_kwargs.num_gpus_per_actor",
            value=merged["num_gpus_per_actor"],
        )
    if merged.get("runtime_env") is not None and not isinstance(merged.get("runtime_env"), dict):
        raise TypeError("training.train_backend_kwargs.runtime_env must be a dict when provided.")
    if merged.get("actor_kwargs") is not None and not isinstance(merged.get("actor_kwargs"), dict):
        raise TypeError("training.train_backend_kwargs.actor_kwargs must be a dict when provided.")
    return merged


def resolve_train_backend_config_from_args(args: Any) -> TrainBackendConfig:
    """Canonicalize backend selection and kwargs exactly once from args."""
    backend_name = _resolve_backend_name_from_args(args)
    backend_dotpath = _resolve_backend_dotpath_from_args(args)
    raw_kwargs = _parse_backend_kwargs_from_args(args)
    strict_unknowns = backend_dotpath is None

    if backend_dotpath is None and backend_name not in _BUILTIN_BACKENDS:
        raise ValueError(
            f"Unsupported train_backend={backend_name!r}. "
            f"Expected one of {sorted(_BUILTIN_BACKENDS)} or provide training.train_backend_dotpath."
        )

    actor_count = max(
        1,
        int(args.ray.training_num_nodes)
        * int(args.ray.training_num_gpus_per_node),
    )

    if backend_name == "fsdp":
        resolved_kwargs = _resolve_fsdp_backend_kwargs(
            args,
            raw_kwargs,
            strict_unknowns=strict_unknowns,
        )
    elif backend_name == "veomni":
        resolved_kwargs = _resolve_veomni_backend_kwargs(
            args,
            raw_kwargs,
            actor_count=actor_count,
            strict_unknowns=strict_unknowns,
        )
    elif backend_name == "megatron":
        resolved_kwargs = _resolve_megatron_backend_kwargs(
            raw_kwargs,
            strict_unknowns=strict_unknowns,
        )
    else:
        resolved_kwargs = dict(raw_kwargs)

    return TrainBackendConfig(
        name=backend_name,
        backend_dotpath=backend_dotpath,
        kwargs=resolved_kwargs,
    )


def create_train_backend(
    name: str,
    *,
    backend_dotpath: Optional[str] = None,
    backend_kwargs: Optional[Mapping[str, Any]] = None,
) -> TrainBackend:
    """Create backend instance from an explicit built-in branch or a custom dotpath."""
    backend_cls = _resolve_backend_cls(
        name,
        backend_dotpath=backend_dotpath,
    )
    return backend_cls(backend_kwargs=dict(backend_kwargs or {}))


def create_train_backend_from_config(
    config: TrainBackendConfig,
) -> TrainBackend:
    return create_train_backend(
        config.name,
        backend_dotpath=config.backend_dotpath,
        backend_kwargs=config.kwargs,
    )


def resolve_train_backend_capabilities(
    name: str,
    *,
    backend_dotpath: Optional[str] = None,
) -> TrainBackendCapabilities:
    """Resolve backend capabilities without instantiating runtime objects."""
    backend_cls = _resolve_backend_cls(
        name,
        backend_dotpath=backend_dotpath,
    )
    return backend_cls.declared_capabilities()


def resolve_train_backend_capabilities_from_config(
    config: TrainBackendConfig,
) -> TrainBackendCapabilities:
    return resolve_train_backend_capabilities(
        config.name,
        backend_dotpath=config.backend_dotpath,
    )


def resolve_train_backend_launch_spec(
    config: TrainBackendConfig,
    *,
    args: Any,
    topology: Any,
) -> TrainBackendLaunchSpec:
    backend_cls = _resolve_backend_cls(
        config.name,
        backend_dotpath=config.backend_dotpath,
    )
    if backend_cls.launch_spec is TrainBackend.launch_spec:
        return backend_cls.declared_launch_spec(
            args=args,
            topology=topology,
            backend_kwargs=config.kwargs,
        )
    backend = create_train_backend_from_config(config)
    return backend.launch_spec(args=args, topology=topology)


def resolve_train_backend_capabilities_from_args(args: Any) -> Dict[str, Any]:
    """Resolve backend capabilities directly from TrainingArguments-like args."""
    config = resolve_train_backend_config_from_args(args)
    capabilities = resolve_train_backend_capabilities_from_config(config)
    return capabilities.as_dict()


__all__ = [
    "TrainBackendConfig",
    "create_train_backend",
    "create_train_backend_from_config",
    "resolve_train_backend_capabilities",
    "resolve_train_backend_capabilities_from_args",
    "resolve_train_backend_capabilities_from_config",
    "resolve_train_backend_config_from_args",
    "resolve_train_backend_launch_spec",
    "supported_train_backends",
]
