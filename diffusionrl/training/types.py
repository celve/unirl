"""Back-compat shim — dataclasses moved to ``diffusionrl.training.backends.base``.

The dataclass definitions (``TrainBackendCapabilities``, ``TrainTopology``,
``TrainBackendLaunchSpec``) now live next to the ``TrainBackend`` protocol
they describe. This module keeps the per-backend capability constants +
resolver helpers used by the (still-present) CLI resolution path; they will
move alongside their concrete backend files as the CLI path is deleted.
"""

from __future__ import annotations

from typing import Any, Dict

from diffusionrl.training.backends.base import (
    TrainBackendCapabilities,
    TrainBackendConfig,
    TrainBackendLaunchSpec,
    TrainTopology,
)

# Retained alias for legacy type hints in the config layer.
BaseTrainBackendConfig = TrainBackendConfig


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
    """Return the canonical capabilities record for a backend identifier."""
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
    """Return the launch-spec hints for a built-in backend."""
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
