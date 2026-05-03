from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol, runtime_checkable

from torch import nn

from diffusionrl.config.registration import register_config
from diffusionrl.config.require import require
from diffusionrl.models.base import ModelBundle
from diffusionrl.training.backends.protocols import (
    LRSchedulerProtocol,
    OptimizerProtocol,
)


@register_config(group="training/optimizer", name="default")
@dataclass
class OptimizerConfig:
    """AdamW-style optimizer hyperparameters consumed by the training actor."""

    learning_rate: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    weight_decay: float


@register_config(group="training/lr_scheduler", name="default")
@dataclass
class LrSchedulerConfig:
    """Learning-rate scheduler hyperparameters.

    Distinguished from ``diffusionrl.config.arguments.SchedulerConfig``, which
    is the timestep-index scheduler used by ``diffusionrl.algorithms.grpo``.
    """

    type: str
    warmup_steps: int
    total_steps: int


@register_config(group="training/backend", name="base")
@dataclass
class TrainBackendConfig:
    """Marker base for new-style training backend configs.

    Concrete configs (e.g. ``FSDPBackendConfig``) subclass this and declare
    their own backend-specific settings. Two intentional constraints:

    1. **Identifier is a ClassVar, not a field.** Each concrete config
       declares ``name: ClassVar[str]`` as its stable identifier (for
       example ``"fsdp"``). Consumers that need the
       identifier read ``config.name``. Using a ClassVar keeps the
       identifier out of the frozen-dataclass init signature and out of
       Ray-side serialization payloads, while still giving validation /
       schema code a single attribute to look up. Backend class
       resolution (the dotpath) stays out of the config entirely — it is
       dispatched from cfg via Hydra ``_target_`` instantiation.

    2. **Frozen dataclass.** Configs are immutable so that actor-side
       replication, hashing, and Ray serialization are well-defined, and so
       that nothing silently mutates a backend's settings after launch.

    No fields are declared on the base by design. Any field added here must
    be meaningful to *every* training backend implementation, and in
    practice superficially shared names like ``cpu_offload`` have
    backend-specific semantics that belong in the concrete subclass.
    """


@dataclass
class TrainBackendCapabilities:
    """Capability declaration for a training backend implementation.

    Historically constructed by ``TrainBackend.declared_capabilities``.
    Preserved here so validation and launch-assembly code can continue to
    gate behavior on backend traits without the old registry. Colocated with
    the ``TrainBackend`` protocol that it describes.
    """

    name: str
    distributed_backend: str = "nccl"
    supports_training_actor_sampling: bool = False
    supports_state_dict_export: bool = True
    supports_custom_actor_class: bool = False
    requires_custom_actor_class: bool = False
    supports_custom_optimizer: bool = False
    supports_custom_scheduler: bool = False
    supports_custom_train_step: bool = False
    supports_backend_managed_offload: bool = False
    preferred_weight_export_format: Optional[str] = None
    preferred_weight_export_format_by_rollout_engine: Mapping[str, str] = field(default_factory=dict)
    supported_weight_export_formats: tuple[str, ...] = ("state_dict",)
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "distributed_backend": self.distributed_backend,
            "supports_training_actor_sampling": self.supports_training_actor_sampling,
            "supports_state_dict_export": self.supports_state_dict_export,
            "supports_custom_actor_class": self.supports_custom_actor_class,
            "requires_custom_actor_class": self.requires_custom_actor_class,
            "supports_custom_optimizer": self.supports_custom_optimizer,
            "supports_custom_scheduler": self.supports_custom_scheduler,
            "supports_custom_train_step": self.supports_custom_train_step,
            "supports_backend_managed_offload": self.supports_backend_managed_offload,
            "preferred_weight_export_format": self.preferred_weight_export_format,
            "preferred_weight_export_format_by_rollout_engine": dict(
                self.preferred_weight_export_format_by_rollout_engine
            ),
            "supported_weight_export_formats": list(self.supported_weight_export_formats),
            "notes": self.notes,
        }


@register_config(group="training/topology", name="default")
@dataclass
class TrainTopology:
    """Unified training topology — injected into the concrete backend at build time.

    ``dp_size`` / ``dp_shard_size`` are ``Optional[int]``: ``None`` means
    "derive from ``torch.distributed.get_world_size()`` at runtime"; an
    explicit int means "use this value". ``dp_replicate_size`` defaults to 1
    (no replication). ``actor_count`` cross-checks the Ray placement's
    train-actor count at bootstrap time.

    Parallel dims (``tp_size`` / ``pp_size`` / ``sp_size`` / ``ep_size``) are
    consumed only by backends that honor them; FSDP ignores topology and
    derives its own sizes from ``dist.get_world_size()``.
    """

    dp_size: Optional[int] = None
    dp_replicate_size: int = 1
    dp_shard_size: Optional[int] = None
    tp_size: int = 1
    pp_size: int = 1
    sp_size: int = 1
    ep_size: int = 1
    cp_size: int = 1
    actor_count: Optional[int] = 1

    def __post_init__(self) -> None:
        require(
            self.dp_size is None or self.dp_size >= 1,
            f"TrainTopology.dp_size must be >= 1 when set; got {self.dp_size!r}",
        )
        require(
            self.dp_shard_size is None or self.dp_shard_size >= 1,
            f"TrainTopology.dp_shard_size must be >= 1 when set; got {self.dp_shard_size!r}",
        )
        for name in ("dp_replicate_size", "tp_size", "pp_size", "sp_size", "ep_size", "cp_size"):
            value = getattr(self, name)
            require(value >= 1, f"TrainTopology.{name} must be >= 1; got {value!r}")
        require(
            self.actor_count is None or self.actor_count >= 1,
            f"TrainTopology.actor_count must be >= 1 when set; got {self.actor_count!r}",
        )

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "dp_replicate_size": int(self.dp_replicate_size),
            "tp_size": int(self.tp_size),
            "pp_size": int(self.pp_size),
            "sp_size": int(self.sp_size),
            "ep_size": int(self.ep_size),
            "cp_size": int(self.cp_size),
        }
        if self.dp_size is not None:
            d["dp_size"] = int(self.dp_size)
        if self.dp_shard_size is not None:
            d["dp_shard_size"] = int(self.dp_shard_size)
        if self.actor_count is not None:
            d["actor_count"] = int(self.actor_count)
        return d


@dataclass
class TrainBackendLaunchSpec:
    """Backend-declared actor/group launch hints used by group factory."""

    actor_class_path: Optional[str] = None
    actor_kwargs: Optional[Mapping[str, Any]] = None
    num_gpus_per_actor: Optional[float] = None
    runtime_env: Optional[Mapping[str, Any]] = None
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "actor_class_path": self.actor_class_path,
            "actor_kwargs": dict(self.actor_kwargs or {}),
            "num_gpus_per_actor": self.num_gpus_per_actor,
            "runtime_env": dict(self.runtime_env or {}),
            "notes": self.notes,
        }


@runtime_checkable
class TrainBackend(Protocol):
    model: nn.Module
    model_bundle: ModelBundle

    def get_state_dict(self, *, lora_only: bool = False) -> Dict[str, Any]: ...
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None: ...
    def clip_grad_norm(self, max_grad_norm: float) -> None: ...
    def onload(self) -> None: ...
    def offload(self) -> None: ...

    def build_optimizer(self, config: OptimizerConfig) -> Optional[OptimizerProtocol]: ...
    def build_scheduler(
        self,
        config: LrSchedulerConfig,
        optimizer: OptimizerProtocol,
    ) -> Optional[LRSchedulerProtocol]: ...


__all__ = [
    "LrSchedulerConfig",
    "OptimizerConfig",
    "TrainBackend",
    "TrainBackendCapabilities",
    "TrainBackendConfig",
    "TrainBackendLaunchSpec",
    "TrainTopology",
]
