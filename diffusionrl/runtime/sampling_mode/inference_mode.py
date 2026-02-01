"""Inference-actor sampling mode adapter."""

from __future__ import annotations

import logging

import ray

from diffusionrl.ray.actor_group import create_training_actor_group

from .base import SamplingModeAdapter

logger = logging.getLogger(__name__)


class InferenceSamplingMode(SamplingModeAdapter):
    """Default sampling backend where dedicated inference actors serve rollout."""

    def rollout_pg_result(self, pgs):
        return pgs.get("inference")

    def create_training_group(self, pgs, rollout_manager):
        if self.args.colocate_inference_training and self.args.offload_rollout:
            logger.info("Colocate mode: offloading inference actors before training actor creation")
            ray.get(rollout_manager.offload.remote())
            self._rollout_on_gpu = False

        training_group = create_training_actor_group(self.args, pgs["training"])
        logger.info("Training actor group created")

        training_group.update_weights()
        logger.info("Initial weights synchronized")

        if self.args.offload_rollout and self.args.offload_train:
            training_group.offload()
            ray.get(rollout_manager.onload.remote())
            self._rollout_on_gpu = True
        elif self.args.offload_rollout:
            ray.get(rollout_manager.onload.remote())
            self._rollout_on_gpu = True

        return training_group

    def _ensure_rollout_on_gpu(self, rollout_manager) -> None:
        if self.args.offload_rollout and not self._rollout_on_gpu:
            ray.get(rollout_manager.onload.remote())
            self._rollout_on_gpu = True

    def before_rollout(self, rollout_manager) -> None:
        self._ensure_rollout_on_gpu(rollout_manager)

    def after_rollout(self, rollout_manager) -> None:
        if self.args.offload_rollout:
            ray.get(rollout_manager.offload.remote())
            self._rollout_on_gpu = False

    def before_train(self, training_group) -> None:
        if self.args.offload_train:
            training_group.onload()

    def after_train(self, training_group) -> None:
        if self.args.offload_train:
            training_group.offload()
        else:
            training_group.clear_memory()

    def maybe_sync_weights(
        self,
        *,
        rollout_id: int,
        training_group,
        rollout_manager,
        sync_weights_fn,
    ) -> None:
        if (rollout_id + 1) % self.args.update_weights_interval != 0:
            return

        self._ensure_rollout_on_gpu(rollout_manager)
        self._current_weight_version = int(
            sync_weights_fn(
                self.args,
                rollout_id,
                training_group,
                rollout_manager,
                target_weight_version=self._current_weight_version + 1,
            )
        )

    def before_eval(self, rollout_manager) -> None:
        self._ensure_rollout_on_gpu(rollout_manager)
