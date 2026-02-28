"""Weight synchronization strategies for rollout updates."""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Type

import ray
import logging

from diffusionrl.utils.misc import load_function
from diffusionrl.utils.weight_sync_checkpoint import cleanup_published_checkpoint

logger = logging.getLogger(__name__)


def _resolve_target_modules(args: Any) -> list[str]:
    engine_kwargs = getattr(args, "engine_kwargs", {}) or {}
    if not isinstance(engine_kwargs, dict):
        return ["transformer"]
    raw = engine_kwargs.get("target_modules")
    if isinstance(raw, (list, tuple)) and raw:
        return [str(name) for name in raw]
    return ["transformer"]


def _resolve_bucket_size_mb(args: Any) -> int:
    explicit = getattr(args, "weight_sync_bucket_mb", None)
    if explicit is not None:
        return max(1, int(explicit))
    engine_kwargs = getattr(args, "engine_kwargs", {}) or {}
    if isinstance(engine_kwargs, dict) and engine_kwargs.get("weight_sync_bucket_mb") is not None:
        return max(1, int(engine_kwargs["weight_sync_bucket_mb"]))
    return 256


def _resolve_flush_cache(args: Any) -> bool:
    explicit = getattr(args, "weight_sync_flush_cache", None)
    if explicit is not None:
        return bool(explicit)
    engine_kwargs = getattr(args, "engine_kwargs", {}) or {}
    if isinstance(engine_kwargs, dict) and engine_kwargs.get("weight_sync_flush_cache") is not None:
        return bool(engine_kwargs["weight_sync_flush_cache"])
    return True


def _validate_tensor_sync_topology(args: Any) -> None:
    """Guard invalid topology values for tensor/distributed sync paths."""
    engine_kwargs = getattr(args, "engine_kwargs", {}) or {}
    tp_size = None
    if isinstance(engine_kwargs, dict):
        tp_size = engine_kwargs.get("tp_size")
    if tp_size is None:
        tp_size = getattr(args, "tp_size", 1)
    try:
        tp_size_int = int(tp_size)
    except Exception:
        tp_size_int = 1
    if tp_size_int < 1:
        raise ValueError(f"Invalid tp_size={tp_size_int}. Expected tp_size >= 1.")


def _resolve_rollout_tp_payload_count(rollout_manager: Any) -> int:
    """Resolve payload fan-out count for IPC tensor updates.

    SGLang workers index `serialized_named_tensors` by local TP rank.
    For tp_size>1, provide one slot per TP rank (same payload can be reused).
    """
    topology = ray.get(rollout_manager.get_weight_sync_topology.remote())
    if not isinstance(topology, dict):
        return 1
    try:
        return max(1, int(topology.get("num_gpus_per_actor", 1)))
    except Exception:
        return 1


class WeightSyncStrategy(ABC):
    """Strategy interface for syncing training weights to rollout actors."""

    def __init__(self, args: Any) -> None:
        self.args = args

    @abstractmethod
    def sync(
        self,
        *,
        rollout_id: int,
        training_group: Any,
        rollout_manager: Any,
    ) -> None:
        """Synchronize latest policy weights to rollout side."""

class CheckpointPathWeightSync(WeightSyncStrategy):
    """Sync strategy using shared checkpoint paths."""

    def _select_export_format(self) -> str:
        engine_type = str(getattr(self.args, "sampler_engine_type", "") or "").lower()
        if engine_type == "sglang":
            return "sglang_transformer_safetensors"
        return "state_dict"

    def _build_weight_checkpoint_path(self, rollout_id: int, *, export_format: str) -> str:
        os.makedirs(self.args.weight_sync_dir, exist_ok=True)
        if export_format == "sglang_transformer_safetensors":
            return os.path.join(
                self.args.weight_sync_dir,
                f"weights_rollout_{rollout_id}_{int(time.time_ns())}",
            )
        return os.path.join(
            self.args.weight_sync_dir,
            f"weights_rollout_{rollout_id}_{int(time.time_ns())}.pt",
        )

    def sync(
        self,
        *,
        rollout_id: int,
        training_group: Any,
        rollout_manager: Any,
    ) -> None:
        export_format = self._select_export_format()
        checkpoint_path = self._build_weight_checkpoint_path(
            rollout_id,
            export_format=export_format,
        )
        training_group.export_weights_to_path(
            checkpoint_path,
            export_format=export_format,
        )

        ray.get(rollout_manager.wake_up.remote())
        ray.get(rollout_manager.update_weights_from_path.remote(checkpoint_path))
        cleanup_published_checkpoint(checkpoint_path)


class ObjectRefWeightSync(WeightSyncStrategy):
    """Sync strategy using Ray object references."""

    def sync(
        self,
        *,
        rollout_id: int,
        training_group: Any,
        rollout_manager: Any,
    ) -> None:
        del rollout_id

        weights_ref = training_group.get_weights()
        ray.wait([weights_ref], num_returns=1)

        ray.get(rollout_manager.wake_up.remote())
        ray.get(rollout_manager.update_weights.remote(weights_ref))


class IPCWeightSync(WeightSyncStrategy):
    """Sync strategy using local tensor IPC payload transfer."""

    def sync(
        self,
        *,
        rollout_id: int,
        training_group: Any,
        rollout_manager: Any,
    ) -> None:
        del rollout_id

        _validate_tensor_sync_topology(self.args)
        ray.get(rollout_manager.wake_up.remote())
        tp_payload_count = _resolve_rollout_tp_payload_count(rollout_manager)
        stats = training_group.sync_weights_to_rollout_ipc(
            rollout_manager=rollout_manager,
            target_modules=_resolve_target_modules(self.args),
            bucket_size_mb=_resolve_bucket_size_mb(self.args),
            flush_cache=_resolve_flush_cache(self.args),
            tp_payload_count=tp_payload_count,
        )
        logger.info(
            "IPC weight sync stats: %s (tp_payload_count=%s)",
            stats,
            tp_payload_count,
        )


class NCCLWeightSync(WeightSyncStrategy):
    """Sync strategy using custom NCCL group broadcast."""

    def sync(
        self,
        *,
        rollout_id: int,
        training_group: Any,
        rollout_manager: Any,
    ) -> None:
        _validate_tensor_sync_topology(self.args)
        ray.get(rollout_manager.wake_up.remote())
        topology = ray.get(rollout_manager.get_weight_sync_topology.remote())
        if not isinstance(topology, dict):
            raise RuntimeError(f"Invalid rollout weight sync topology payload: {topology!r}")
        total_rollout_gpus = int(topology.get("total_gpus", 0))
        if total_rollout_gpus <= 0:
            logger.info("Skipping NCCL weight sync: no rollout GPUs configured.")
            return

        endpoint = training_group.get_rank0_ip_and_free_port()
        master_address = str(endpoint["master_address"])
        master_port = int(endpoint["master_port"])
        world_size = int(total_rollout_gpus + 1)
        group_name = f"diffusionrl_wsync_{int(rollout_id)}_{int(time.time_ns())}"

        rollout_init_ref = rollout_manager.init_weights_update_group.remote(
            master_address=master_address,
            master_port=master_port,
            world_size=world_size,
            group_name=group_name,
            backend="nccl",
        )
        train_init_ref = training_group.async_init_weights_update_group(
            master_address=master_address,
            master_port=master_port,
            world_size=world_size,
            group_name=group_name,
            backend="nccl",
        )
        try:
            ray.get([rollout_init_ref, train_init_ref])
        except Exception:
            logger.exception("Failed to initialize NCCL weight-sync group %s", group_name)
            try:
                training_group.destroy_weights_update_group(group_name=group_name)
            except Exception:
                logger.exception("Failed to clean training-side weight-sync group %s", group_name)
            try:
                ray.get(
                    rollout_manager.destroy_weights_update_group.remote(
                        group_name=group_name,
                    )
                )
            except Exception:
                logger.exception("Failed to clean rollout-side weight-sync group %s", group_name)
            raise

        try:
            stats = training_group.sync_weights_to_rollout_nccl(
                rollout_manager=rollout_manager,
                group_name=group_name,
                target_modules=_resolve_target_modules(self.args),
                bucket_size_mb=_resolve_bucket_size_mb(self.args),
                flush_cache=_resolve_flush_cache(self.args),
            )
            logger.info("NCCL weight sync stats: %s", stats)
        finally:
            try:
                training_group.destroy_weights_update_group(group_name=group_name)
            except Exception:
                logger.exception("Failed to destroy training-side weight-sync group %s", group_name)
            finally:
                try:
                    ray.get(
                        rollout_manager.destroy_weights_update_group.remote(
                            group_name=group_name,
                        )
                    )
                except Exception:
                    logger.exception("Failed to destroy rollout-side weight-sync group %s", group_name)


_BUILTIN_WEIGHT_SYNC_STRATEGIES: Dict[str, Type[WeightSyncStrategy]] = {
    "checkpoint_path": CheckpointPathWeightSync,
    "object_ref": ObjectRefWeightSync,
    "ipc": IPCWeightSync,
    "nccl": NCCLWeightSync,
}


def create_weight_sync_strategy(args: Any) -> WeightSyncStrategy:
    """
    Create weight sync strategy from runtime args.

    Extension point:
    - If args.weight_sync_strategy_path exists, dynamically load custom strategy.
    - Otherwise resolve built-in strategies from args.weight_sync_mode.
    """
    strategy_path = getattr(args, "weight_sync_strategy_path", None)
    if strategy_path:
        strategy_cls = load_function(strategy_path)
        return strategy_cls(args)

    mode = getattr(args, "weight_sync_mode", "object_ref")
    strategy_cls = _BUILTIN_WEIGHT_SYNC_STRATEGIES.get(mode)
    if strategy_cls is None:
        raise ValueError(
            f"Unsupported weight_sync_mode={mode}. "
            f"Expected one of: {sorted(_BUILTIN_WEIGHT_SYNC_STRATEGIES.keys())}"
        )
    return strategy_cls(args)


__all__ = [
    "WeightSyncStrategy",
    "CheckpointPathWeightSync",
    "ObjectRefWeightSync",
    "IPCWeightSync",
    "NCCLWeightSync",
    "create_weight_sync_strategy",
]
