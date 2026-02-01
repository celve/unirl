"""Sampling mode adapters for train loop orchestration."""

from __future__ import annotations

from typing import Any, Dict


class SamplingModeAdapter:
    """
    Thin adapter interface for sampling backend runtime differences.

    Keep train loop skeleton stable while isolating mode-specific transitions.
    """

    def __init__(self, args) -> None:
        self.args = args
        self._rollout_on_gpu = True
        self._current_weight_version = 0

    @property
    def current_weight_version(self) -> int:
        return int(self._current_weight_version)

    def rollout_pg_result(self, pgs: Dict[str, Any]):
        raise NotImplementedError

    def create_training_group(self, pgs: Dict[str, Any], rollout_manager):
        raise NotImplementedError

    def before_rollout(self, rollout_manager) -> None:
        return

    def after_rollout(self, rollout_manager) -> None:
        return

    def before_train(self, training_group) -> None:
        return

    def after_train(self, training_group) -> None:
        return

    def maybe_sync_weights(
        self,
        *,
        rollout_id: int,
        training_group,
        rollout_manager,
        sync_weights_fn,
    ) -> None:
        return

    def before_eval(self, rollout_manager) -> None:
        return
