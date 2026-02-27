"""Centralized train-backend factory with explicit backend branching."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from diffusionrl.utils import load_function

from .base import TrainBackend
from .fsdp import FSDPTrainBackend
from .megatron import MegatronTrainBackend
from .veomni import VeOmniTrainBackend

_SUPPORTED_BUILTIN_BACKENDS = ("fsdp", "megatron", "veomni")


def supported_train_backends() -> tuple[str, ...]:
    """Return built-in train backend names recognized by configuration."""
    return _SUPPORTED_BUILTIN_BACKENDS


def create_train_backend(
    name: str,
    *,
    backend_path: Optional[str] = None,
    backend_kwargs: Optional[Mapping[str, Any]] = None,
) -> TrainBackend:
    """Create backend instance from an explicit built-in branch or a custom dotpath."""
    if backend_path:
        backend_cls = load_function(backend_path)
        if not isinstance(backend_cls, type) or not issubclass(backend_cls, TrainBackend):
            raise TypeError(
                f"train_backend_path must resolve to a TrainBackend subclass, got: {backend_cls}"
            )
        return backend_cls(backend_kwargs=backend_kwargs)

    backend_name = str(name or "fsdp").strip().lower()

    if backend_name == "fsdp":
        return FSDPTrainBackend(backend_kwargs=dict(backend_kwargs or {}))

    if backend_name == "veomni":
        return VeOmniTrainBackend(backend_kwargs=dict(backend_kwargs or {}))

    if backend_name == "megatron":
        return MegatronTrainBackend(backend_kwargs=dict(backend_kwargs or {}))

    raise ValueError(
        f"Unsupported train_backend={name!r}. "
        f"Expected one of {list(_SUPPORTED_BUILTIN_BACKENDS)} or provide train_backend_path."
    )


__all__ = [
    "create_train_backend",
    "supported_train_backends",
]
