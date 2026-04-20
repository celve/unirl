"""Shared training types used by the config/launch layers.

These types were previously defined in ``diffusionrl/training/backends/base.py``
as part of the legacy abc-based backend system. They are now preserved here
as standalone dataclasses because downstream config/assembly/validation code
depends on their shape, even though the new protocol-based backend system
(``diffusionrl.training.backends``) does not use them directly.

For each supported built-in backend (``fsdp``, ``veomni``) a canonical
capability record is declared below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from diffusionrl.training.backends.base import TrainBackendConfig


# ``BaseTrainBackendConfig`` is retained as an alias for the new marker so that
# legacy type hints in the config layer continue to resolve to a real type.
BaseTrainBackendConfig = TrainBackendConfig


@dataclass(frozen=True)
class TrainBackendCapabilities:
    """Capability declaration for a training backend implementation.

    Historically constructed by ``TrainBackend.declared_capabilities``.
    Preserved here so that validation and launch-assembly code can continue
    to gate behavior on backend traits without the old registry.
    """

    name: str
    distributed_backend: str = "nccl"
    supports_training_actor_sampling: bool = False
    supports_state_dict_export: bool = True
    supports_custom_actor_class: bool = False
    requires_custom_actor_class: bool = False
    supports_custom_optimizer: bool = False
    supports_custom_scheduler: bool = False
    supports_custom_train_step: bool = False
    supports_backend_managed_offload: bool = False
    preferred_weight_export_format: Optional[str] = None
    preferred_weight_export_format_by_rollout_engine: Mapping[str, str] = field(
        default_factory=dict
    )
    supported_weight_export_formats: tuple[str, ...] = ("state_dict",)
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "distributed_backend": self.distributed_backend,
            "supports_training_actor_sampling": self.supports_training_actor_sampling,
            "supports_state_dict_export": self.supports_state_dict_export,
            "supports_custom_actor_class": self.supports_custom_actor_class,
            "requires_custom_actor_class": self.requires_custom_actor_class,
            "supports_custom_optimizer": self.supports_custom_optimizer,
            "supports_custom_scheduler": self.supports_custom_scheduler,
            "supports_custom_train_step": self.supports_custom_train_step,
            "supports_backend_managed_offload": self.supports_backend_managed_offload,
            "preferred_weight_export_format": self.preferred_weight_export_format,
            "preferred_weight_export_format_by_rollout_engine": dict(
                self.preferred_weight_export_format_by_rollout_engine
            ),
            "supported_weight_export_formats": list(self.supported_weight_export_formats),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class TrainTopology:
    """Unified training topology used by both config resolution and launch.

    ``actor_count`` is the size of the launched training actor group (set
    during config resolution; may be ``None`` before that).
    ``world_size`` is the distributed training rank count.
    ``dp_size`` is the data-parallel consumer count used for training batch
    geometry.

    These values often coincide in the current FSDP mainline but should not
    be treated as interchangeable.
    """

    world_size: int
    dp_size: int
    dp_replicate_size: int = 1
    dp_shard_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    sp_size: int = 1
    ep_size: int = 1
    actor_count: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "world_size": int(self.world_size),
            "dp_size": int(self.dp_size),
            "dp_replicate_size": int(self.dp_replicate_size),
            "dp_shard_size": int(self.dp_shard_size),
            "tp_size": int(self.tp_size),
            "pp_size": int(self.pp_size),
            "sp_size": int(self.sp_size),
            "ep_size": int(self.ep_size),
        }
        if self.actor_count is not None:
            d["actor_count"] = int(self.actor_count)
        return d


@dataclass(frozen=True)
class TrainBackendLaunchSpec:
    """Backend-declared actor/group launch hints used by group factory."""

    actor_class_path: Optional[str] = None
    actor_kwargs: Optional[Mapping[str, Any]] = None
    num_gpus_per_actor: Optional[float] = None
    runtime_env: Optional[Mapping[str, Any]] = None
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "actor_class_path": self.actor_class_path,
            "actor_kwargs": dict(self.actor_kwargs or {}),
            "num_gpus_per_actor": self.num_gpus_per_actor,
            "runtime_env": dict(self.runtime_env or {}),
            "notes": self.notes,
        }


_FSDP_CAPABILITIES = TrainBackendCapabilities(
    name="fsdp",
    distributed_backend="nccl",
    supports_training_actor_sampling=True,
    supports_state_dict_export=True,
    supports_custom_optimizer=False,
    supports_custom_scheduler=False,
    supports_backend_managed_offload=True,
    preferred_weight_export_format="state_dict",
    supported_weight_export_formats=("state_dict",),
    notes="Built-in FSDP2 training backend (new protocol-based).",
)


_VEOMNI_CAPABILITIES = TrainBackendCapabilities(
    name="veomni",
    distributed_backend="nccl",
    supports_training_actor_sampling=True,
    supports_state_dict_export=True,
    supports_custom_optimizer=True,
    supports_custom_scheduler=True,
    supports_backend_managed_offload=False,
    preferred_weight_export_format="state_dict",
    supported_weight_export_formats=("state_dict",),
    notes=(
        "Built-in VeOmni backend (new protocol-based). Uses VeOmni native APIs "
        "for model parallelization, optimizer/lr scheduler construction, and "
        "EP-aware grad clipping."
    ),
)


_BUILTIN_CAPABILITIES: Dict[str, TrainBackendCapabilities] = {
    "fsdp": _FSDP_CAPABILITIES,
    "veomni": _VEOMNI_CAPABILITIES,
}


def resolve_train_backend_capabilities(identifier: str) -> TrainBackendCapabilities:
    """Return the canonical capabilities record for a backend identifier.

    ``identifier`` is the short name used on the ``--train-backend`` CLI flag
    (``fsdp`` or ``veomni``) or a full dotpath. Dotpaths are not supported in
    the new system beyond the built-in set; out-of-tree backends should wire
    their own capability handling if they need to differ from FSDP's defaults.
    """
    key = str(identifier or "").strip().lower()
    record = _BUILTIN_CAPABILITIES.get(key)
    if record is None:
        raise ValueError(
            f"Unknown train backend identifier: {identifier!r}. "
            f"Supported built-in backends: {sorted(_BUILTIN_CAPABILITIES)}."
        )
    return record


def resolve_train_backend_launch_spec(
    config: Any,
    *,
    args: Any,
    topology: Any,
) -> TrainBackendLaunchSpec:
    """Return the launch-spec hints for a built-in backend.

    Neither built-in backend currently exposes non-default launch hints;
    out-of-tree backends can subclass or provide their own configuration.
    """
    del config, args, topology
    return TrainBackendLaunchSpec()


# Legacy names kept for import compatibility with the retired
# ``diffusionrl/training/backends/`` package.
derive_train_backend_capabilities = resolve_train_backend_capabilities
derive_train_backend_launch_spec = resolve_train_backend_launch_spec


def supported_train_backends() -> tuple[str, ...]:
    """Return the list of supported built-in train backend identifiers."""
    return tuple(sorted(_BUILTIN_CAPABILITIES))


__all__ = [
    "BaseTrainBackendConfig",
    "TrainBackendCapabilities",
    "TrainBackendLaunchSpec",
    "TrainTopology",
    "resolve_train_backend_capabilities",
    "resolve_train_backend_launch_spec",
    "derive_train_backend_capabilities",
    "derive_train_backend_launch_spec",
    "supported_train_backends",
]
