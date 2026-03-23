"""Training backend contracts and explicit factory."""

from __future__ import annotations

from .base import (
    ActorTrainBackendContext,
    TrainBackend,
    TrainBackendCapabilities,
    TrainBackendLaunchSpec,
    TrainTopology,
)
from .factory import create_train_backend, resolve_train_backend_capabilities, supported_train_backends
from .factory import resolve_train_backend_capabilities_from_args
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
    "create_train_backend",
    "resolve_train_backend_capabilities",
    "resolve_train_backend_capabilities_from_args",
    "supported_train_backends",
    "FSDPTrainBackend",
    "MegatronTrainBackend",
    "VeOmniNativeTrainBackend",
    "VeOmniTrainBackend",
]
