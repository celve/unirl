"""Training backend contracts and explicit factory."""

from __future__ import annotations

from .base import TrainBackend, TrainBackendCapabilities
from .factory import create_train_backend, supported_train_backends
from .fsdp import FSDPTrainBackend

__all__ = [
    "TrainBackend",
    "TrainBackendCapabilities",
    "create_train_backend",
    "supported_train_backends",
    "FSDPTrainBackend",
]
