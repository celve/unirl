"""
Reward workers for GRPO training.

Supports:
- Local reward computation (LocalRewardWorker)
- HTTP-based remote reward services (HTTPRewardWorker)
- Ray-based GPU-isolated reward computation (RayRewardWorker)
- Unified RewardService with multiple worker support

Architecture:

    RewardService (unified entry point)
        │
        ├── LocalRewardWorker (CPU/same-process GPU)
        │
        ├── HTTPRewardWorker (HTTP API)
        │
        └── RayRewardWorker (GPU-isolated via Ray actors)

Reward Modes:
    1. HTTP: use_http_reward=True -> HTTPRewardWorker
    2. Independent GPU: reward_dedicated_num_gpus > 0 -> RayRewardWorker
    3. CPU: default -> LocalRewardWorker

Example usage:
    # Simple usage via RewardService (recommended)
    from diffusionrl.reward import RewardService
    service = RewardService(args, reward_pg_result=pgs.get("reward"))
    response = service.compute_rewards(request)

    # Direct worker usage (for custom integrations)
    from diffusionrl.reward import LocalRewardWorker, RewardRequest
    worker = LocalRewardWorker(model_name="hpsv2", weight=1.0)
    response = worker.compute_rewards(RewardRequest(images=imgs, prompts=prompts))
"""

# Base types and interfaces
from .base import (
    BaseRewardWorker,
    RewardRequest,
    RewardResponse,
    RewardType,
)

# Concrete workers
from .local import LocalRewardWorker, VideoRewardWorker
from .http import HTTPRewardWorker, AsyncHTTPRewardWorker
from .ray_worker import RayRewardWorker

# Unified service
from .service import RewardService

__all__ = [
    # Base types
    "BaseRewardWorker",
    "RewardRequest",
    "RewardResponse",
    "RewardType",
    # Workers
    "LocalRewardWorker",
    "VideoRewardWorker",
    "HTTPRewardWorker",
    "AsyncHTTPRewardWorker",
    "RayRewardWorker",
    # Service
    "RewardService",
]
