"""Training-actor sampling mode plugin."""

from __future__ import annotations

import logging

import ray

from diffusionrl.ray.actor_group import create_training_actor_group

from .base import SamplingModePlugin

logger = logging.getLogger(__name__)


class TrainingSamplingMode(SamplingModePlugin):
    """Sampling backend where training actors also serve sampling requests."""

    def rollout_pg_result(self, pgs):
        return None

    def create_training_group(self, pgs, rollout_manager):
        training_group = create_training_actor_group(self.args, pgs["training"])
        logger.info("Training actor group created (sampling backend: training)")
        ray.get(rollout_manager.attach_sampling_actors.remote(training_group))
        return training_group
