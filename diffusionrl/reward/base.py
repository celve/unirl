"""
Base reward worker interface.

All reward workers must inherit from BaseRewardWorker.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import time

from diffusionrl.types.reward import RewardRequest, RewardResponse, RewardType


class BaseRewardWorker(ABC):
    """
    Abstract base class for reward workers.

    Reward workers compute rewards for generated images/videos given prompts.
    They can run locally or connect to remote services.

    This is the unified interface that merges the old Worker and Backend concepts.
    All reward workers should inherit from this class.

    Example usage:
        worker = LocalRewardWorker(
            model_name="pickscore",
            device="cuda",
            weight=1.0,
        )

        response = worker.compute_rewards(
            RewardRequest(
                images=[img1, img2],
                prompts=["a cat", "a dog"],
            )
        )
        print(response.rewards)  # [0.8, 0.7]
    """

    def __init__(
        self,
        model_name: str = "",
        weight: float = 1.0,
        reward_types: Optional[List[RewardType]] = None,
        batch_size: int = 8,
        timeout: float = 60.0,
        **kwargs,
    ):
        """
        Initialize reward worker.

        Args:
            model_name: Name of the reward model
            weight: Weight for this worker in multi-reward aggregation
            reward_types: Types of rewards this worker can compute
            batch_size: Maximum batch size for processing
            timeout: Timeout for reward computation (seconds)
        """
        self.model_name = model_name
        self.weight = weight
        self.reward_types = reward_types or [RewardType.IMAGE_TEXT_ALIGNMENT]
        self.batch_size = batch_size
        self.timeout = timeout

    @abstractmethod
    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        """
        Compute rewards for the given request.

        Args:
            request: RewardRequest with images/videos and prompts

        Returns:
            RewardResponse with computed rewards
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the reward worker is available."""
        pass

    def get_weight(self) -> float:
        """Get the weight for multi-reward aggregation."""
        return self.weight

    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model_name

    async def compute_rewards_async(self, request: RewardRequest) -> RewardResponse:
        """
        Async version of compute_rewards.

        Default implementation falls back to sync version.
        Subclasses may override for true async support.

        Args:
            request: RewardRequest with images/videos and prompts

        Returns:
            RewardResponse with computed rewards
        """
        return self.compute_rewards(request)

    def offload(self) -> None:
        """
        Offload model to CPU to free GPU memory.

        Default implementation does nothing.
        Subclasses with GPU models should override.
        """
        pass

    def onload(self) -> None:
        """
        Load model back to GPU.

        Default implementation does nothing.
        Subclasses with GPU models should override.
        """
        pass

    def dispose(self) -> None:
        """
        Clean up resources.

        Default implementation does nothing.
        Subclasses should override to release resources.
        """
        pass

    def _timed_compute(
        self,
        func,
        *args,
        **kwargs,
    ) -> tuple:
        """Helper to time computation."""
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed

