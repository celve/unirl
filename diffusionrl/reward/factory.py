"""Factory for constructing reward executors from a RewardSpec."""

from __future__ import annotations

import logging
from typing import List

import torch

from diffusionrl.reward.base import (
    BaseRewardExecutor,
    BaseRewardScorer,
    RewardRequest,
    RewardResponse,
)
from diffusionrl.reward.config import RewardSpec
from diffusionrl.reward.scorers.registry import resolve_builtin_reward_scorer_class
from diffusionrl.utils import load_function

logger = logging.getLogger(__name__)


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


def build_executors(spec: RewardSpec) -> List[BaseRewardExecutor]:
    """Construct executor instances based on the spec's execution plan."""
    if not isinstance(spec, RewardSpec):
        raise TypeError(f"build_executors requires RewardSpec, got: {type(spec).__name__}")

    return _build_local_executors(spec)


def _build_local_executors(spec: RewardSpec) -> List[BaseRewardExecutor]:
    definition = spec.to_definition()
    provider = spec.to_provider_config()
    execution_plan = spec.to_execution_plan()

    device = _resolve_local_device(execution_plan.local_device)
    if device == "cuda":
        logger.warning("Local reward scorer is running on CUDA in-process; this can contend with sampling GPUs.")

    reward_dotpath = provider.reward_dotpath

    def _create_executor(model_name: str, weight: float) -> BaseRewardExecutor:
        if reward_dotpath:
            scorer_cls = load_function(reward_dotpath)
        else:
            scorer_cls = resolve_builtin_reward_scorer_class(model_name)

        if not isinstance(scorer_cls, type) or not issubclass(scorer_cls, BaseRewardScorer):
            logger.warning(
                "Local reward scorer %s does not inherit BaseRewardScorer; treating it as a scorer via duck typing.",
                reward_dotpath or scorer_cls,
            )

        scorer = scorer_cls(
            model_name=model_name,
            device=device,
            batch_size=provider.batch_size,
            timeout=provider.timeout,
        )
        return InProcessRewardExecutor(scorer=scorer, weight=weight)

    executors: List[BaseRewardExecutor] = []
    component_names = definition.component_names
    component_weights = definition.component_weights_list

    if component_names:
        weights = component_weights or [1.0] * len(component_names)

        for i, model in enumerate(component_names):
            weight = weights[i] if i < len(weights) else 1.0
            executor = _create_executor(model_name=model, weight=weight)
            executors.append(executor)
            logger.info(
                "Added in-process reward executor: %s via %s (weight=%s)",
                model,
                type(executor.scorer).__name__,
                weight,
            )

    else:
        executor = _create_executor(
            model_name=definition.default_model_name,
            weight=1.0,
        )
        executors.append(executor)
        logger.info(
            "Added in-process reward executor: %s via %s",
            definition.default_model_name,
            type(executor.scorer).__name__,
        )

    return executors


def _resolve_local_device(local_device_pref: str) -> str:
    pref = str(local_device_pref or "cpu").strip().lower()
    if pref == "cpu":
        return "cpu"
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("local_reward_device=cuda requested but CUDA is not available. Falling back to CPU.")
        return "cpu"
    logger.warning(
        "Unknown local_reward_device=%s. Falling back to CPU.",
        local_device_pref,
    )
    return "cpu"


__all__ = [
    "InProcessRewardExecutor",
    "build_executors",
]
