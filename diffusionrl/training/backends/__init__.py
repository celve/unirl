"""Training backend contracts and registry-backed construction."""

from __future__ import annotations

from .base import (
    ActorTrainBackendContext,
    BaseTrainBackendConfig,
    TrainBackend,
    TrainBackendCapabilities,
    TrainBackendLaunchSpec,
    TrainTopology,
    derive_train_backend_capabilities,
    derive_train_backend_launch_spec,
)
from .construction import create_train_backend_from_init_payload
from .fsdp import FSDPTrainBackend, FSDPTrainBackendConfig
from .megatron import MegatronTrainBackend, MegatronTrainBackendConfig
from .registry import (
    ensure_builtin_train_backend_registration,
    register_train_backend,
    resolve_train_backend_class,
    supported_train_backends,
)
from .veomni import VeOmniTrainBackend
from .veomni_native import VeOmniNativeTrainBackend, VeOmniTrainBackendConfig

ensure_builtin_train_backend_registration()

__all__ = [
    "TrainBackend",
    "ActorTrainBackendContext",
    "BaseTrainBackendConfig",
    "TrainBackendCapabilities",
    "TrainBackendLaunchSpec",
    "TrainTopology",
    "create_train_backend_from_init_payload",
    "derive_train_backend_capabilities",
    "derive_train_backend_launch_spec",
    "register_train_backend",
    "resolve_train_backend_class",
    "supported_train_backends",
    "FSDPTrainBackendConfig",
    "FSDPTrainBackend",
    "MegatronTrainBackendConfig",
    "MegatronTrainBackend",
    "VeOmniTrainBackendConfig",
    "VeOmniNativeTrainBackend",
    "VeOmniTrainBackend",
]
