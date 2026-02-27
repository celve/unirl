"""Training backend contracts and explicit factory."""

from __future__ import annotations

from .base import (
    TrainBackend,
    TrainBackendCapabilities,
    TrainBackendLaunchSpec,
    TrainTopology,
)
from .factory import create_train_backend, supported_train_backends
from .fsdp import FSDPTrainBackend
from .megatron import MegatronTrainBackend
from .veomni_native import VeOmniNativeTrainBackend
from .veomni import VeOmniTrainBackend

__all__ = [
    "TrainBackend",
    "TrainBackendCapabilities",
    "TrainBackendLaunchSpec",
    "TrainTopology",
    "create_train_backend",
    "supported_train_backends",
    "FSDPTrainBackend",
    "MegatronTrainBackend",
    "VeOmniNativeTrainBackend",
    "VeOmniTrainBackend",
]
