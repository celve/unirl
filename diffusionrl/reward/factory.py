"""Reward executor factories.

This module keeps deployment construction explicit:

- driver-side reward execution -> ``create_driver_reward_executor``
- actor-local precompute -> see ``diffusionrl.reward.actor_local``
"""

from __future__ import annotations

from typing import Any, Optional

from diffusionrl.reward.schema import RewardSchema

from .service import RewardService


def create_driver_reward_executor(
    reward_schema: RewardSchema,
    *,
    reward_pg_result: Optional[Any] = None,
) -> Optional[RewardService]:
    """Create the driver-side reward executor when the driver owns scoring."""
    if reward_schema.to_execution_plan().uses_sampling_actor_execution:
        return None
    return RewardService(
        reward_schema=reward_schema,
        reward_pg_result=reward_pg_result,
    )


__all__ = [
    "create_driver_reward_executor",
]
