"""Train backend contracts."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class TrainBackendCapabilities:
    """Capability declaration for a training backend implementation."""

    name: str
    distributed_backend: str = "nccl"
    supports_training_actor_sampling: bool = False
    buffer_partition_mode: str = "data_parallel"
    supports_state_dict_export: bool = True
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "distributed_backend": self.distributed_backend,
            "supports_training_actor_sampling": self.supports_training_actor_sampling,
            "buffer_partition_mode": self.buffer_partition_mode,
            "supports_state_dict_export": self.supports_state_dict_export,
            "notes": self.notes,
        }


class TrainBackend(abc.ABC):
    """Backend abstraction used by TrainingActor."""

    BACKEND_NAME = "unknown"

    def __init__(self, *, backend_kwargs: Optional[Mapping[str, Any]] = None) -> None:
        self.backend_kwargs: Dict[str, Any] = dict(backend_kwargs or {})

    @property
    def name(self) -> str:
        return str(self.BACKEND_NAME)

    @classmethod
    def declared_capabilities(cls) -> TrainBackendCapabilities:
        return TrainBackendCapabilities(name=cls.BACKEND_NAME)

    @property
    def capabilities(self) -> TrainBackendCapabilities:
        return self.declared_capabilities()

    def before_model_load(self, actor: Any) -> None:
        """Hook called after device setup but before model construction."""
        del actor

    def wrap_model(self, actor: Any) -> None:
        """Hook called after actor.model is created."""
        del actor

    @abc.abstractmethod
    def uses_sharded_model(self) -> bool:
        """Whether grad clipping should use model-native sharded routine."""

    @abc.abstractmethod
    def get_state_dict(
        self,
        actor: Any,
        *,
        lora_only: bool = False,
        rank0_only: bool = True,
    ) -> Dict[str, Any]:
        """Collect model state dict for sync/checkpoint."""

    @abc.abstractmethod
    def load_state_dict(self, actor: Any, state_dict: Dict[str, Any]) -> None:
        """Load model state dict into actor.model."""

    def broadcast_parameters(self, actor: Any) -> None:
        """Optional broadcast step after updates."""
        del actor

    def data_parallel_size(self, actor: Any) -> int:
        """Return effective data-parallel size consumed by train batches."""
        return int(getattr(actor, "world_size", 1))

    def buffer_consumer_spec(self, actor: Any) -> Dict[str, Any]:
        """Declare how rollout buffer should prepare train payloads."""
        dp_size = self.data_parallel_size(actor)
        return {
            "dp_size": dp_size,
            "partition_train_data": True,
            "partition_mode": self.capabilities.buffer_partition_mode,
        }


__all__ = [
    "TrainBackendCapabilities",
    "TrainBackend",
]
