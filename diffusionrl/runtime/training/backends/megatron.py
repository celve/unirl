"""Megatron training backend scaffold.

This module intentionally provides interface + launch-structure only.
Training lifecycle integration is staged and not fully implemented yet.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import (
    TrainBackend,
    TrainBackendCapabilities,
    TrainBackendLaunchSpec,
    TrainTopology,
)


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


class MegatronTrainBackend(TrainBackend):
    """Megatron backend interface scaffold (launcher + topology contract)."""

    BACKEND_NAME = "megatron"

    def __init__(self, *, backend_kwargs: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(backend_kwargs=backend_kwargs)
        kwargs = dict(self.backend_kwargs)

        self._actor_class_path = kwargs.pop("actor_class_path", None)
        self._dp_size_hint = _as_optional_int(kwargs.pop("dp_size", None))
        self._tp_size = _as_optional_int(kwargs.pop("tp_size", None)) or 1
        self._pp_size = _as_optional_int(kwargs.pop("pp_size", None)) or 1
        self._sp_size = _as_optional_int(kwargs.pop("sp_size", None)) or 1
        self._ep_size = _as_optional_int(kwargs.pop("ep_size", None)) or 1

        unsupported_num_actors = kwargs.pop("num_actors", None)
        if unsupported_num_actors is not None:
            raise ValueError(
                "train_backend_kwargs.num_actors is no longer supported. "
                "Training actor count is owned by ray.training_num_nodes × "
                "ray.training_num_gpus_per_node."
            )
        self._launch_num_gpus_per_actor = kwargs.pop("num_gpus_per_actor", None)
        runtime_env = kwargs.pop("runtime_env", None)
        actor_kwargs = kwargs.pop("actor_kwargs", None)
        self._launch_runtime_env = runtime_env if isinstance(runtime_env, dict) else {}
        self._launch_actor_kwargs = actor_kwargs if isinstance(actor_kwargs, dict) else {}
        if kwargs:
            import logging

            logging.getLogger(__name__).warning(
                "MegatronTrainBackend received unknown train_backend_kwargs keys: %s. "
                "These keys are currently ignored by the megatron scaffold backend.",
                sorted(kwargs.keys()),
            )

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
            preferred_weight_transport="checkpoint_path",
            preferred_weight_export_format="state_dict",
            supported_weight_export_formats=("state_dict",),
            notes=(
                "Megatron backend scaffold: launch/topology hooks are wired, "
                "runtime training path is intentionally not implemented yet. "
                "A Megatron-dedicated actor class must be provided via backend kwargs."
            ),
        )

    def launch_spec(self, *, args: Any, topology: Any) -> TrainBackendLaunchSpec:
        del args, topology
        return TrainBackendLaunchSpec(
            actor_class_path=self._actor_class_path,
            actor_kwargs=dict(self._launch_actor_kwargs),
            num_gpus_per_actor=self._launch_num_gpus_per_actor,
            runtime_env=dict(self._launch_runtime_env),
            notes=(
                "Use backend_kwargs.actor_class_path to switch to a Megatron-dedicated Ray actor "
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


__all__ = ["MegatronTrainBackend"]
