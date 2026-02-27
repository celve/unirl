"""Minimal reward worker plugin template."""

from __future__ import annotations

import time

from diffusionrl.reward.base import BaseRewardWorker
from diffusionrl.types.reward import RewardRequest, RewardResponse


class MinimalRewardWorker(BaseRewardWorker):
    """Template reward worker that returns constant zero rewards."""

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        start = time.time()
        batch_size = request.batch_size
        rewards = [0.0] * batch_size
        return RewardResponse(
            rewards=rewards,
            reward_components={"constant": rewards},
            successes=[True] * batch_size,
            errors=[None] * batch_size,
            compute_time=time.time() - start,
        )

    def is_available(self) -> bool:
        return True

