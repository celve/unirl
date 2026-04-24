"""Reward service: dispatch + lifecycle over a fixed set of executors."""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from diffusionrl.reward.config import RewardSpec

from .aggregation import aggregate
from .base import BaseRewardExecutor, RewardRequest, RewardResponse
from .factory import build_executors

logger = logging.getLogger(__name__)


class RewardService:
    """Owns per-component executors for one runtime host; dispatches and aggregates."""

    def __init__(
        self,
        executors: Optional[List[BaseRewardExecutor]] = None,
        aggregation_method: str = "weighted_sum",
        *,
        reward_config: Optional[RewardSpec] = None,
    ) -> None:
        if reward_config is not None:
            if executors is not None:
                raise ValueError("RewardService: pass either reward_config= or executors=, not both.")
            if not isinstance(reward_config, RewardSpec):
                raise TypeError(f"RewardService requires RewardSpec, got: {type(reward_config).__name__}")
            executors = build_executors(reward_config)
            aggregation_method = reward_config.to_definition().reward_aggregation_method

        self.executors: List[BaseRewardExecutor] = list(executors or [])
        self.reward_aggregation_method = str(aggregation_method)

        logger.info(
            "RewardService initialized with %d executor(s), aggregation=%s",
            len(self.executors),
            self.reward_aggregation_method,
        )

    @classmethod
    def from_spec(cls, spec: RewardSpec) -> "RewardService":
        """Build a RewardService directly from a RewardSpec."""
        return cls(reward_config=spec)

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        """Compute rewards using configured executors."""
        if not self.executors:
            return RewardResponse(
                rewards=[0.0] * request.batch_size,
                successes=[False] * request.batch_size,
                errors=["No executors configured"] * request.batch_size,
                compute_time=0.0,
            )

        start_time = time.time()

        if len(self.executors) == 1:
            return self.executors[0].compute_rewards(request)

        responses = []
        for executor in self.executors:
            try:
                resp = executor.compute_rewards(request)
                responses.append((resp, executor))
            except Exception as e:
                logger.error("Executor %s failed: %s", executor.get_model_name(), e)
                error_resp = RewardResponse(
                    rewards=[0.0] * request.batch_size,
                    successes=[False] * request.batch_size,
                    errors=[str(e)] * request.batch_size,
                    compute_time=0.0,
                )
                responses.append((error_resp, executor))

        batch_size = responses[0][0].batch_size if responses else 0
        return aggregate(
            self.reward_aggregation_method,
            responses,
            batch_size,
            time.time() - start_time,
        )

    @property
    def preferred_input_kind(self) -> str:
        """Return the media kind expected by the configured executor set."""
        kinds = {
            str(getattr(executor, "preferred_input_kind", "image") or "image").strip().lower()
            for executor in self.executors
        }
        kinds.discard("")
        if not kinds:
            return "image"
        if len(kinds) > 1:
            raise ValueError(
                f"Mixed reward input kinds in one service are not supported. Configured kinds={sorted(kinds)}."
            )
        return next(iter(kinds))

    def is_available(self) -> bool:
        return any(executor.is_available() for executor in self.executors)

    def offload(self) -> None:
        for executor in self.executors:
            executor.offload()
        logger.debug("RewardService offloaded %d executor(s)", len(self.executors))

    def onload(self) -> None:
        for executor in self.executors:
            executor.onload()
        logger.debug("RewardService onloaded %d executor(s)", len(self.executors))

    def dispose(self) -> None:
        for executor in self.executors:
            executor.dispose()
        self.executors = []
        logger.info("RewardService disposed")


__all__ = [
    "RewardService",
]
