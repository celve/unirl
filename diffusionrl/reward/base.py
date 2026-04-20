"""Base abstractions for reward scorers and executors."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple

from diffusionrl.types.reward import RewardRequest, RewardResponse, RewardType


class _BaseRewardNode(ABC):
    """Shared runtime metadata for reward scorers and executors."""

    input_kind = "image"

    def __init__(
        self,
        model_name: str = "",
        batch_size: int = 8,
        timeout: float = 60.0,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.timeout = timeout

    def get_model_name(self) -> str:
        """Get the model or component name for this runtime node."""
        return self.model_name

    @property
    def preferred_input_kind(self) -> str:
        """Return the decoded media kind consumed by this runtime node."""
        return str(getattr(self, "input_kind", "image") or "image").strip().lower()

    async def compute_rewards_async(self, request: RewardRequest) -> RewardResponse:
        """Async fallback that delegates to the sync implementation."""
        return self.compute_rewards(request)

    def offload(self) -> None:
        """Optional lifecycle hook for releasing device memory."""
        pass

    def onload(self) -> None:
        """Optional lifecycle hook for reacquiring device memory."""
        pass

    def dispose(self) -> None:
        """Optional lifecycle hook for terminal cleanup."""
        pass

    def _timed_compute(
        self,
        func: Callable[..., Any],
        *args,
        **kwargs,
    ) -> Tuple[Any, float]:
        """Helper to time computation."""
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed


class BaseRewardScorer(_BaseRewardNode):
    """Leaf scorer interface: turn a RewardRequest into scores."""

    def __init__(
        self,
        model_name: str = "",
        reward_types: Optional[List[RewardType]] = None,
        batch_size: int = 8,
        timeout: float = 60.0,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name=model_name,
            batch_size=batch_size,
            timeout=timeout,
            **kwargs,
        )
        self.reward_types = reward_types or [RewardType.IMAGE_TEXT_ALIGNMENT]

    @abstractmethod
    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        """Compute rewards for the given request."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the scorer is available."""


class BaseRewardExecutor(_BaseRewardNode):
    """Execution interface: run one reward component on some backend."""

    def __init__(
        self,
        model_name: str = "",
        weight: float = 1.0,
        batch_size: int = 8,
        timeout: float = 60.0,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name=model_name,
            batch_size=batch_size,
            timeout=timeout,
            **kwargs,
        )
        self.weight = float(weight)

    def get_weight(self) -> float:
        """Get the semantic aggregation weight for this executor."""
        return self.weight

    @abstractmethod
    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        """Execute one reward component against the given request."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the executor backend is available."""


__all__ = [
    "BaseRewardScorer",
    "BaseRewardExecutor",
]
