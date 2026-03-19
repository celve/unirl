"""Centralized train-backend factory with explicit backend branching."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from diffusionrl.utils import load_function

from .base import TrainBackend, TrainBackendCapabilities
from .fsdp import FSDPTrainBackend
from .megatron import MegatronTrainBackend
from .veomni import VeOmniTrainBackend

_BUILTIN_BACKENDS: dict[str, type[TrainBackend]] = {
    "fsdp": FSDPTrainBackend,
    "veomni": VeOmniTrainBackend,
    "megatron": MegatronTrainBackend,
}


def supported_train_backends() -> tuple[str, ...]:
    """Return built-in train backend names recognized by configuration."""
    return tuple(_BUILTIN_BACKENDS.keys())


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
    backend_cls = _BUILTIN_BACKENDS.get(backend_name)
    if backend_cls is None:
        raise ValueError(
            f"Unsupported train_backend={name!r}. "
            f"Expected one of {list(_BUILTIN_BACKENDS)} or provide train_backend_path."
    )
    return backend_cls(backend_kwargs=dict(backend_kwargs or {}))


def resolve_train_backend_capabilities(
    name: str,
    *,
    backend_path: Optional[str] = None,
) -> TrainBackendCapabilities:
    """Resolve backend capabilities without instantiating runtime objects."""
    if backend_path:
        backend_cls = load_function(backend_path)
        if not isinstance(backend_cls, type) or not issubclass(backend_cls, TrainBackend):
            raise TypeError(
                f"train_backend_path must resolve to a TrainBackend subclass, got: {backend_cls}"
            )
        return backend_cls.declared_capabilities()

    backend_name = str(name or "fsdp").strip().lower()
    backend_cls = _BUILTIN_BACKENDS.get(backend_name)
    if backend_cls is None:
        raise ValueError(
            f"Unsupported train_backend={name!r}. "
            f"Expected one of {list(_BUILTIN_BACKENDS)} or provide train_backend_path."
        )
    return backend_cls.declared_capabilities()


__all__ = [
    "create_train_backend",
    "resolve_train_backend_capabilities",
    "supported_train_backends",
]
