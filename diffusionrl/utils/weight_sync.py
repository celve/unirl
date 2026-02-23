"""Weight synchronization strategies for rollout updates."""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Type

import ray

from diffusionrl.utils.misc import load_function
from diffusionrl.utils.weight_sync_checkpoint import cleanup_published_checkpoint


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

    @staticmethod
    def _run_post_update_hooks(rollout_manager: Any) -> None:
        ray.get(rollout_manager.after_weight_update.remote())
        ray.get(rollout_manager.reload_runtime_cache.remote())


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

        ray.get(rollout_manager.load_weights.remote())
        ray.get(rollout_manager.update_weights_from_path.remote(checkpoint_path))
        self._run_post_update_hooks(rollout_manager)
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

        ray.get(rollout_manager.load_weights.remote())
        ray.get(rollout_manager.update_weights.remote(weights_ref))
        self._run_post_update_hooks(rollout_manager)


_BUILTIN_WEIGHT_SYNC_STRATEGIES: Dict[str, Type[WeightSyncStrategy]] = {
    "checkpoint_path": CheckpointPathWeightSync,
    "object_ref": ObjectRefWeightSync,
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
    "create_weight_sync_strategy",
]
