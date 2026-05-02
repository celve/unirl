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


class BaseRewardComponentSpec(ABC):
    """Marker base for every reward component spec.

    Each scorer registers a concrete ``<Name>Spec`` dataclass via
    ``@register_config(group="reward/component", name="<short>", target=...)``
    and inherits from this base. Kept as a plain ``ABC`` (not a ``@dataclass``)
    so that ``register_config``'s auto-promotion processes each subclass's own
    field annotations without ``is_dataclass(parent)`` short-circuiting.

    The Spec is pure data. The ``target`` on each subclass points at the
    runtime constructor (a scorer ``__init__`` for in-process scorers);
    ``RewardService.from_configs`` invokes ``config.build(...)`` per component
    to produce the runtime node.

    The polymorphic-list field on ``RewardConfig`` is typed as
    ``Tuple[Any, ...]`` at runtime rather than
    ``Tuple[BaseRewardComponentSpec, ...]``: OmegaConf's structured-config
    validation rejects raw dict assignment to a typed-base list at YAML
    compose time. The polymorphism is carried by ``polymorphic_field``
    metadata, not by the field's declared element type.
    """

    weight: float


class InProcessRewardExecutor(BaseRewardExecutor):
    """Thin executor wrapper around one in-process reward scorer."""

    def __init__(
        self,
        scorer: BaseRewardScorer,
        *,
        weight: float,
    ) -> None:
        super().__init__(
            model_name=scorer.get_model_name(),
            weight=weight,
            batch_size=scorer.batch_size,
            timeout=scorer.timeout,
        )
        self.scorer = scorer

    @property
    def preferred_input_kind(self) -> str:
        return self.scorer.preferred_input_kind

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        return self.scorer.compute_rewards(request)

    def is_available(self) -> bool:
        return self.scorer.is_available()

    def offload(self) -> None:
        self.scorer.offload()

    def onload(self) -> None:
        self.scorer.onload()

    def dispose(self) -> None:
        self.scorer.dispose()


__all__ = [
    "BaseRewardComponentSpec",
    "BaseRewardScorer",
    "BaseRewardExecutor",
    "InProcessRewardExecutor",
]
