"""Train backend contracts."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


class ActorTrainBackendContext:
    """Minimal mutable actor view exposed to training backends.

    This context intentionally exposes only the actor state that backend hooks
    are allowed to read or mutate directly. It gives backend code a stable
    contract without forcing the full TrainingActor surface to become public.
    """

    _READ_MAP = {
        "rank": "rank",
        "world_size": "world_size",
        "model": "model",
        "model_bundle": "model_bundle",
        "optimizer": "optimizer",
        "lr_scheduler": "lr_scheduler",
        "fsdp_cpu_offload": "_fsdp_cpu_offload",
        "_fsdp_cpu_offload": "_fsdp_cpu_offload",
    }
    _WRITE_MAP = {
        "model": "model",
        "model_bundle": "model_bundle",
        "optimizer": "optimizer",
        "lr_scheduler": "lr_scheduler",
        "fsdp_cpu_offload": "_fsdp_cpu_offload",
        "_fsdp_cpu_offload": "_fsdp_cpu_offload",
    }

    def __init__(self, actor: Any) -> None:
        object.__setattr__(self, "_actor", actor)

    def __getattr__(self, name: str) -> Any:
        target_name = self._READ_MAP.get(name)
        if target_name is None:
            raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")
        return getattr(self._actor, target_name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_actor":
            object.__setattr__(self, name, value)
            return
        target_name = self._WRITE_MAP.get(name)
        if target_name is None:
            raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")
        setattr(self._actor, target_name, value)


@dataclass(frozen=True)
class BaseTrainBackendConfig:
    """Base typed config shared by train backend implementations."""

    name: str
    backend_dotpath: Optional[str] = None


@dataclass(frozen=True)
class TrainBackendCapabilities:
    """Capability declaration for a training backend implementation."""

    name: str
    distributed_backend: str = "nccl"
    supports_training_actor_sampling: bool = False
    buffer_partition_mode: str = "data_parallel"
    supports_state_dict_export: bool = True
    supports_custom_actor_class: bool = False
    requires_custom_actor_class: bool = False
    supports_custom_optimizer: bool = False
    supports_custom_scheduler: bool = False
    supports_custom_train_step: bool = False
    supports_backend_managed_offload: bool = False
    preferred_weight_export_format: Optional[str] = None
    preferred_weight_export_format_by_rollout_engine: Mapping[str, str] = ()
    supported_weight_export_formats: tuple[str, ...] = ("state_dict",)
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "distributed_backend": self.distributed_backend,
            "supports_training_actor_sampling": self.supports_training_actor_sampling,
            "buffer_partition_mode": self.buffer_partition_mode,
            "supports_state_dict_export": self.supports_state_dict_export,
            "supports_custom_actor_class": self.supports_custom_actor_class,
            "requires_custom_actor_class": self.requires_custom_actor_class,
            "supports_custom_optimizer": self.supports_custom_optimizer,
            "supports_custom_scheduler": self.supports_custom_scheduler,
            "supports_custom_train_step": self.supports_custom_train_step,
            "supports_backend_managed_offload": self.supports_backend_managed_offload,
            "preferred_weight_export_format": self.preferred_weight_export_format,
            "preferred_weight_export_format_by_rollout_engine": dict(
                self.preferred_weight_export_format_by_rollout_engine or {}
            ),
            "supported_weight_export_formats": list(self.supported_weight_export_formats),
            "notes": self.notes,
        }

@dataclass(frozen=True)
class TrainTopology:
    """Unified training topology used by both config resolution and backend runtime.

    actor_count is the size of the launched training actor group (set during
    config resolution; backends may leave it as ``None``).
    world_size is the distributed training rank count.
    dp_size is the data-parallel consumer count used for training batch geometry.

    These values often coincide in the current FSDP mainline, but they should
    not be treated as interchangeable concepts.
    """

    world_size: int
    dp_size: int
    dp_replicate_size: int = 1
    dp_shard_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    sp_size: int = 1
    ep_size: int = 1
    data_partition_axis: str = "dp"
    actor_count: Optional[int] = None
    partition_mode: str = "data_parallel"

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "world_size": int(self.world_size),
            "dp_size": int(self.dp_size),
            "dp_replicate_size": int(self.dp_replicate_size),
            "dp_shard_size": int(self.dp_shard_size),
            "tp_size": int(self.tp_size),
            "pp_size": int(self.pp_size),
            "sp_size": int(self.sp_size),
            "ep_size": int(self.ep_size),
            "data_partition_axis": str(self.data_partition_axis),
            "partition_mode": str(self.partition_mode),
        }
        if self.actor_count is not None:
            d["actor_count"] = int(self.actor_count)
        return d


@dataclass(frozen=True)
class TrainBackendLaunchSpec:
    """Backend-declared actor/group launch hints used by group factory."""

    actor_class_path: Optional[str] = None
    actor_kwargs: Mapping[str, Any] = None
    num_gpus_per_actor: Optional[float] = None
    runtime_env: Mapping[str, Any] = None
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "actor_class_path": self.actor_class_path,
            "actor_kwargs": dict(self.actor_kwargs or {}),
            "num_gpus_per_actor": self.num_gpus_per_actor,
            "runtime_env": dict(self.runtime_env or {}),
            "notes": self.notes,
        }


class TrainBackend(abc.ABC):
    """Backend abstraction used by TrainingActor.

    Methods are organized into three tiers:

    **Core contract** (abstract, all backends must implement):
        ``uses_sharded_model``, ``get_state_dict``, ``load_state_dict``

    **Lifecycle hooks** (most backends override):
        ``before_model_load``, ``wrap_model``, ``topology``,
        ``clip_grad_norm``, ``broadcast_parameters``, ``data_parallel_size``

    **Optional hooks** (override only when needed, safe defaults):
        ``build_optimizer``, ``build_scheduler``, ``run_train_step``,
        ``launch_spec``, ``export_weights_to_path``, ``offload``, ``onload``
    """

    BACKEND_NAME = "unknown"

    def __init__(self, config: BaseTrainBackendConfig) -> None:
        if not isinstance(config, BaseTrainBackendConfig):
            raise TypeError(
                f"{type(self).__name__} expected {BaseTrainBackendConfig.__name__}, got: {config!r}"
            )
        self.config = config

    @property
    def name(self) -> str:
        return str(self.BACKEND_NAME)

    @classmethod
    def declared_capabilities(cls) -> TrainBackendCapabilities:
        return TrainBackendCapabilities(name=cls.BACKEND_NAME)

    @property
    def capabilities(self) -> TrainBackendCapabilities:
        return self.declared_capabilities()

    # ---- Core contract (abstract) ----

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

    # ---- Lifecycle hooks (most backends override) ----

    def before_model_load(self, actor: Any) -> None:
        """Hook called after device setup but before model construction."""
        del actor

    def wrap_model(self, actor: Any) -> None:
        """Hook called after actor.model is created."""
        del actor

    def clip_grad_norm(
        self,
        actor: Any,
        *,
        model: Any,
        max_grad_norm: float,
    ) -> Any:
        """Grad-norm clip; override for sharded-model-aware clipping."""
        import torch

        del actor
        return torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_grad_norm,
        )

    def topology(self, actor: Any) -> TrainTopology:
        """Report backend runtime topology."""
        dp_size = self.data_parallel_size(actor)
        return TrainTopology(
            world_size=int(getattr(actor, "world_size", dp_size)),
            dp_size=int(dp_size),
            dp_replicate_size=1,
            dp_shard_size=int(dp_size),
        )

    def broadcast_parameters(self, actor: Any) -> None:
        """Broadcast step after weight updates."""
        del actor

    def data_parallel_size(self, actor: Any) -> int:
        """Return effective data-parallel size consumed by train batches."""
        return int(getattr(actor, "world_size", 1))

    # ---- Optional hooks (override only when needed) ----

    @classmethod
    def declared_launch_spec(
        cls,
        *,
        args: Any,
        topology: Any,
        config: BaseTrainBackendConfig,
    ) -> TrainBackendLaunchSpec:
        """Launch-time actor/group hints consumed by group factory."""
        del cls, args, topology, config
        return TrainBackendLaunchSpec()

    def launch_spec(self, *, args: Any, topology: Any) -> TrainBackendLaunchSpec:
        return type(self).declared_launch_spec(
            args=args,
            topology=topology,
            config=self.config,
        )

    def build_optimizer(self, actor: Any, optimizer_config: Mapping[str, Any]) -> Any:
        """Optimizer construction hook; return None to use actor default."""
        del actor, optimizer_config
        return None

    def build_scheduler(self, actor: Any, scheduler_config: Mapping[str, Any]) -> Any:
        """Scheduler construction hook; return None to use actor default."""
        del actor, scheduler_config
        return None

    def run_train_step(
        self,
        actor: Any,
        *,
        rollout_id: int,
        batch: Any,
        executor: Any,
    ) -> Optional[Dict[str, Any]]:
        """Train-step override; return metrics dict to bypass default executor."""
        del actor, rollout_id, batch, executor
        return None

    def export_weights_to_path(
        self,
        actor: Any,
        checkpoint_path: str,
        *,
        export_format: str,
    ) -> Optional[str]:
        """Export hook for non-state-dict weight artifacts."""
        del actor, checkpoint_path, export_format
        return None

    def offload(self, actor: Any) -> bool:
        """Backend-managed offload; return True when fully handled."""
        del actor
        return False

    def onload(self, actor: Any) -> bool:
        """Backend-managed onload; return True when fully handled."""
        del actor
        return False

    def backend_info(self, actor: Any) -> Dict[str, Any]:
        """Return backend metadata bundle used by orchestration/logging."""
        caps = self.capabilities
        return {
            "name": self.name,
            "capabilities": caps.as_dict(),
            "topology": self.topology(actor).as_dict(),
            "weight_sync": {
                "preferred_export_format": caps.preferred_weight_export_format,
                "preferred_export_format_by_rollout_engine": dict(
                    caps.preferred_weight_export_format_by_rollout_engine or {}
                ),
                "supported_export_formats": list(caps.supported_weight_export_formats),
            },
        }


def resolve_train_backend_capabilities(
    identifier: str,
) -> TrainBackendCapabilities:
    from .registry import resolve_train_backend_class

    backend_cls = resolve_train_backend_class(identifier)
    return backend_cls.declared_capabilities()


def resolve_train_backend_capabilities_from_config(
    config: BaseTrainBackendConfig,
) -> TrainBackendCapabilities:
    identifier = (
        str(config.backend_dotpath).strip()
        if config.backend_dotpath
        else str(config.name).strip().lower()
    )
    return resolve_train_backend_capabilities(identifier)


def resolve_train_backend_launch_spec(
    config: BaseTrainBackendConfig,
    *,
    args: Any,
    topology: Any,
) -> TrainBackendLaunchSpec:
    from .registry import resolve_train_backend_class

    identifier = (
        str(config.backend_dotpath).strip()
        if config.backend_dotpath
        else str(config.name).strip().lower()
    )
    backend_cls = resolve_train_backend_class(identifier)
    if backend_cls.launch_spec is TrainBackend.launch_spec:
        return backend_cls.declared_launch_spec(
            args=args,
            topology=topology,
            config=config,
        )
    backend = backend_cls(config=config)
    return backend.launch_spec(args=args, topology=topology)


__all__ = [
    "ActorTrainBackendContext",
    "BaseTrainBackendConfig",
    "TrainBackendCapabilities",
    "TrainTopology",
    "TrainBackendLaunchSpec",
    "TrainBackend",
    "resolve_train_backend_capabilities",
    "resolve_train_backend_capabilities_from_config",
    "resolve_train_backend_launch_spec",
]
