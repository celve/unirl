"""Training backend contracts and explicit factory."""

from __future__ import annotations

from .base import (
    ActorTrainBackendContext,
    TrainBackend,
    TrainBackendCapabilities,
    TrainBackendLaunchSpec,
    TrainTopology,
)
from .factory import (
    TrainBackendConfig,
    create_train_backend,
    create_train_backend_from_config,
    resolve_train_backend_capabilities,
    resolve_train_backend_capabilities_from_args,
    resolve_train_backend_capabilities_from_config,
    resolve_train_backend_config_from_args,
    resolve_train_backend_launch_spec,
    supported_train_backends,
)
from .fsdp import FSDPTrainBackend
from .megatron import MegatronTrainBackend
from .veomni_native import VeOmniNativeTrainBackend
from .veomni import VeOmniTrainBackend

__all__ = [
    "TrainBackend",
    "ActorTrainBackendContext",
    "TrainBackendCapabilities",
    "TrainBackendLaunchSpec",
    "TrainTopology",
    "TrainBackendConfig",
    "create_train_backend",
    "create_train_backend_from_config",
    "resolve_train_backend_capabilities",
    "resolve_train_backend_capabilities_from_args",
    "resolve_train_backend_capabilities_from_config",
    "resolve_train_backend_config_from_args",
    "resolve_train_backend_launch_spec",
    "supported_train_backends",
    "FSDPTrainBackend",
    "MegatronTrainBackend",
    "VeOmniNativeTrainBackend",
    "VeOmniTrainBackend",
]
