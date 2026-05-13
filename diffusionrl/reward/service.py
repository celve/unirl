"""Reward service: dispatch + lifecycle over a fixed set of executors."""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from omegaconf import DictConfig

from diffusionrl.config.instantiate import build, materialize

from .aggregation import aggregate
from .base import BaseRewardExecutor, BaseRewardScorer, InProcessRewardExecutor, RewardRequest, RewardResponse

logger = logging.getLogger(__name__)


class RewardService:
    """Owns per-component executors for one runtime host; dispatches and aggregates."""

    def __init__(
        self,
        executors: Optional[List[BaseRewardExecutor]] = None,
        aggregation_method: str = "weighted_sum",
    ) -> None:
        self.executors: List[BaseRewardExecutor] = list(executors or [])
        self.reward_aggregation_method = str(aggregation_method)

        logger.info(
            "RewardService initialized with %d executor(s), aggregation=%s",
            len(self.executors),
            self.reward_aggregation_method,
        )

    @classmethod
    def from_configs(cls, reward: DictConfig) -> "RewardService":
        """Build a RewardService from the raw ``cfg.reward`` DictConfig.

        Materializes the parent for top-level fields (``aggregation_method``,
        ``base_device``) plus per-component weights, then dispatches each
        component via :func:`diffusionrl.config.instantiate.build`. Scorer
        results are wrapped in :class:`InProcessRewardExecutor`; executor
        results pass through (HTTP).
        """
        rc = materialize(reward)
        executors: List[BaseRewardExecutor] = []
        for cfg_node, spec in zip(reward.components, rc.components):
            built = build(cfg_node, base_device=rc.base_device)
            if isinstance(built, BaseRewardScorer):
                built = InProcessRewardExecutor(built, weight=spec.weight)
            executors.append(built)
        return cls(
            executors=executors,
            aggregation_method=rc.aggregation_method,
        )

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
            except Exception as e:
                raise RuntimeError(
                    f"Reward executor {executor.get_model_name()!r} failed during compute_rewards "
                    f"(batch_size={request.batch_size}): {type(e).__name__}: {e}"
                ) from e
            responses.append((resp, executor))

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
