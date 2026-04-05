"""Megatron training backend scaffold.

This module intentionally provides interface + launch-structure only.
Training lifecycle integration is staged and not fully implemented yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .base import (
    BaseTrainBackendConfig,
    TrainBackend,
    TrainBackendCapabilities,
    TrainBackendLaunchSpec,
    TrainTopology,
)
from .registry import register_train_backend


def _as_optional_int(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    try:
        value = int(raw)
    except Exception:
        return None
    if value < 1:
        return None
    return value


@dataclass(frozen=True)
class MegatronTrainBackendConfig(BaseTrainBackendConfig):
    name: str = "megatron"
    actor_class_path: Optional[str] = None
    dp_size: Optional[int] = None
    tp_size: int = 1
    pp_size: int = 1
    sp_size: int = 1
    ep_size: int = 1
    num_gpus_per_actor: Optional[float] = None
    runtime_env: Dict[str, Any] = field(default_factory=dict)
    actor_kwargs: Dict[str, Any] = field(default_factory=dict)


@register_train_backend(
    component_name="megatron",
    component_cfg=MegatronTrainBackendConfig,
)
class MegatronTrainBackend(TrainBackend):
    """Megatron backend interface scaffold (launcher + topology contract)."""

    BACKEND_NAME = "megatron"

    def __init__(self, config: MegatronTrainBackendConfig) -> None:
        super().__init__(config)
        self._actor_class_path = config.actor_class_path
        self._dp_size_hint = _as_optional_int(config.dp_size)
        self._tp_size = _as_optional_int(config.tp_size) or 1
        self._pp_size = _as_optional_int(config.pp_size) or 1
        self._sp_size = _as_optional_int(config.sp_size) or 1
        self._ep_size = _as_optional_int(config.ep_size) or 1
        self._launch_num_gpus_per_actor = config.num_gpus_per_actor
        self._launch_runtime_env = dict(config.runtime_env or {})
        self._launch_actor_kwargs = dict(config.actor_kwargs or {})

    @classmethod
    def declared_capabilities(cls) -> TrainBackendCapabilities:
        return TrainBackendCapabilities(
            name=cls.BACKEND_NAME,
            distributed_backend="nccl",
            supports_training_actor_sampling=False,
            buffer_partition_mode="data_parallel",
            supports_state_dict_export=False,
            supports_custom_actor_class=True,
            requires_custom_actor_class=True,
            supports_custom_optimizer=True,
            supports_custom_scheduler=True,
            supports_custom_train_step=True,
            supports_backend_managed_offload=True,
            preferred_weight_export_format="state_dict",
            supported_weight_export_formats=("state_dict",),
            notes=(
                "Megatron backend scaffold: launch/topology hooks are wired, "
                "runtime training path is intentionally not implemented yet. "
                "A Megatron-dedicated actor class must be provided via config."
            ),
        )

    @classmethod
    def declared_launch_spec(
        cls,
        *,
        args: Any,
        topology: Any,
        config: MegatronTrainBackendConfig,
    ) -> TrainBackendLaunchSpec:
        del cls, args, topology
        return TrainBackendLaunchSpec(
            actor_class_path=config.actor_class_path,
            actor_kwargs=dict(config.actor_kwargs or {}),
            num_gpus_per_actor=config.num_gpus_per_actor,
            runtime_env=dict(config.runtime_env or {}),
            notes=(
                "Use config.actor_class_path to switch to a Megatron-dedicated Ray actor "
                "when runtime implementation is ready."
            ),
        )

    def data_parallel_size(self, actor: Any) -> int:
        if self._dp_size_hint is not None:
            return int(self._dp_size_hint)
        world_size = int(getattr(actor, "world_size", 1))
        denom = max(1, self._tp_size * self._pp_size * self._sp_size)
        return max(1, world_size // denom)

    def topology(self, actor: Any) -> TrainTopology:
        world_size = int(getattr(actor, "world_size", 1))
        dp_size = self.data_parallel_size(actor)
        return TrainTopology(
            world_size=world_size,
            dp_size=dp_size,
            dp_replicate_size=dp_size,
            dp_shard_size=1,
            tp_size=int(self._tp_size),
            pp_size=int(self._pp_size),
            sp_size=int(self._sp_size),
            ep_size=int(self._ep_size),
            data_partition_axis="dp",
        )

    def uses_sharded_model(self) -> bool:
        return True

    def _raise_not_implemented(self, op: str) -> None:
        raise NotImplementedError(
            "Megatron backend runtime is not implemented yet. "
            f"Missing operation: {op}. "
            "Current scope only ships launch/topology interfaces for future integration."
        )

    def before_model_load(self, actor: Any) -> None:
        del actor

    def wrap_model(self, actor: Any) -> None:
        del actor
        self._raise_not_implemented("wrap_model")

    def get_state_dict(
        self,
        actor: Any,
        *,
        lora_only: bool = False,
        rank0_only: bool = True,
    ) -> Dict[str, Any]:
        del actor, lora_only, rank0_only
        self._raise_not_implemented("get_state_dict")

    def load_state_dict(self, actor: Any, state_dict: Dict[str, Any]) -> None:
        del actor, state_dict
        self._raise_not_implemented("load_state_dict")


__all__ = [
    "MegatronTrainBackendConfig",
    "MegatronTrainBackend",
]
