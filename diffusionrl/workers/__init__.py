"""
Workers for GRPO training.

This module contains worker implementations for:
- Inference (sampling)
- Reward computation
"""

from .reward import (
    BaseRewardWorker,
    LocalRewardWorker,
    HTTPRewardWorker,
    RewardService,
    RewardRequest,
    RewardResponse,
    RewardType,
)

__all__ = [
    "BaseRewardWorker",
    "LocalRewardWorker",
    "HTTPRewardWorker",
    "RewardService",
    "RewardRequest",
    "RewardResponse",
    "RewardType",
]
