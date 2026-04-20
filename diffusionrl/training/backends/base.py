from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from torch import nn

from diffusionrl.config.training_sections import LrSchedulerConfig, OptimizerConfig
from diffusionrl.models.base import ModelBundle
from diffusionrl.training.backends.protocols import (
    LRSchedulerProtocol,
    OptimizerProtocol,
)


@dataclass(frozen=True)
class TrainBackendConfig:
    """Marker base for new-style training backend configs.

    Concrete configs (e.g. ``FSDPBackendConfig``) subclass this and declare
    their own backend-specific settings. Two intentional constraints:

    1. **Identifier is a ClassVar, not a field.** Each concrete config
       declares ``name: ClassVar[str]`` as its stable identifier (for
       example ``"fsdp"``, ``"veomni"``). Consumers that need the
       identifier read ``config.name``
       (``cmdline.resolution``, ``cmdline.schema``). Using a ClassVar
       keeps the identifier out of the frozen-dataclass init signature
       and out of Ray-side serialization payloads, while still giving
       validation / schema code a single attribute to look up. Backend
       class resolution (the dotpath) stays out of the config entirely —
       it is dispatched from args by
       ``diffusionrl.cmdline.train_backend.resolve_train_backend_identifier``.

    2. **Frozen dataclass.** Configs are immutable so that actor-side
       replication, hashing, and Ray serialization are well-defined, and so
       that nothing silently mutates a backend's settings after launch.

    No fields are declared on the base by design. Any field added here must
    be meaningful to *every* training backend implementation, and in
    practice superficially shared names like ``cpu_offload`` have
    backend-specific semantics that belong in the concrete subclass.
    """


@runtime_checkable
class TrainBackend(Protocol):
    model: nn.Module
    model_bundle: ModelBundle

    def get_state_dict(self, *, lora_only: bool = False) -> Dict[str, Any]: ...
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None: ...
    def clip_grad_norm(self, max_grad_norm: float) -> None: ...
    def onload(self) -> None: ...
    def offload(self) -> None: ...

    def build_optimizer(
        self, config: OptimizerConfig
    ) -> Optional[OptimizerProtocol]: ...
    def build_scheduler(
        self,
        config: LrSchedulerConfig,
        optimizer: OptimizerProtocol,
    ) -> Optional[LRSchedulerProtocol]: ...


__all__ = [
    "TrainBackendConfig",
    "TrainBackend",
]
