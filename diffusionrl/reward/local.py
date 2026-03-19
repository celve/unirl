"""Compatibility facade for built-in local reward scorers."""

from __future__ import annotations

import time
from typing import Callable, List, Optional

import torch

from diffusionrl.reward.base import BaseRewardScorer, RewardRequest, RewardResponse, RewardType

from .scorers.registry import (
    available_builtin_reward_models,
    resolve_builtin_reward_scorer_class,
)
from .scorers.video import VideoRewardScorer


class LocalRewardScorer(BaseRewardScorer):
    """
    Backward-compatible local reward scorer facade.

    Existing configs keep using ``diffusionrl.reward.local.LocalRewardScorer``.
    Internally this class now delegates built-in model implementations to
    dedicated scorer modules under ``diffusionrl.reward.scorers``.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        weight: float = 1.0,
        reward_fn: Optional[Callable] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        batch_size: int = 8,
        timeout: float = 60.0,
        **model_kwargs,
    ) -> None:
        super().__init__(
            model_name=model_name or "",
            batch_size=batch_size,
            timeout=timeout,
        )

        self.reward_fn = reward_fn
        self.device = device
        self.dtype = dtype
        self.model_kwargs = dict(model_kwargs)
        self._delegate: Optional[BaseRewardScorer] = None
        self._is_loaded = False

        if reward_fn is not None:
            self.reward_types = [RewardType.CUSTOM]
            self._is_loaded = True
        elif model_name is not None:
            scorer_cls = resolve_builtin_reward_scorer_class(model_name)
            self._delegate = scorer_cls(
                model_name=model_name,
                weight=weight,
                device=device,
                dtype=dtype,
                batch_size=batch_size,
                timeout=timeout,
                **model_kwargs,
            )
            self.reward_types = list(
                getattr(
                    self._delegate,
                    "reward_types",
                    [RewardType.IMAGE_TEXT_ALIGNMENT],
                )
            )
            self._is_loaded = self._delegate.is_available()

    @property
    def preferred_input_kind(self) -> str:
        if self._delegate is not None:
            return self._delegate.preferred_input_kind
        return super().preferred_input_kind

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        start = time.time()

        if not self._is_loaded:
            return RewardResponse(
                rewards=[0.0] * request.batch_size,
                successes=[False] * request.batch_size,
                errors=["Model not loaded"] * request.batch_size,
                compute_time=0.0,
            )

        try:
            if self.reward_fn is not None:
                rewards = self._compute_custom(request)
                return RewardResponse(
                    rewards=rewards,
                    successes=[True] * len(rewards),
                    errors=[None] * len(rewards),
                    compute_time=time.time() - start,
                )
            if self._delegate is None:
                raise ValueError(
                    "Unknown model_name: "
                    f"{self.model_name}. Available: {list(available_builtin_reward_models())}"
                )
            response = self._delegate.compute_rewards(request)
            response.compute_time = time.time() - start
            return response
        except Exception as e:
            return RewardResponse(
                rewards=[0.0] * request.batch_size,
                successes=[False] * request.batch_size,
                errors=[str(e)] * request.batch_size,
                compute_time=time.time() - start,
            )

    def _compute_custom(self, request: RewardRequest) -> List[float]:
        if request.is_video:
            return self.reward_fn(request.videos, request.prompts)
        return self.reward_fn(request.images, request.prompts)

    def is_available(self) -> bool:
        if self.reward_fn is not None:
            return self._is_loaded
        if self._delegate is None:
            return False
        return self._delegate.is_available()

    def offload(self) -> None:
        if self._delegate is not None:
            self._delegate.offload()

    def onload(self) -> None:
        if self._delegate is not None:
            self._delegate.onload()

    def dispose(self) -> None:
        if self._delegate is not None:
            self._delegate.dispose()
            self._delegate = None
        self._is_loaded = self.reward_fn is not None


__all__ = [
    "LocalRewardScorer",
    "VideoRewardScorer",
]
