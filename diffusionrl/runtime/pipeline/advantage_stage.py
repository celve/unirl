"""Advantage stage helpers for RolloutManager."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

import torch

logger = logging.getLogger(__name__)


def get_reward_component_weights(
    reward_components: Dict[str, List[float]],
    reward_workers: Optional[Iterable[Any]] = None,
) -> Dict[str, float]:
    """Map reward component name to configured worker weight."""
    default_weights = {name: 1.0 for name in reward_components.keys()}
    if reward_workers is None:
        return default_weights

    for worker in reward_workers:
        model_name = worker.get_model_name()
        if model_name in default_weights:
            default_weights[model_name] = float(worker.get_weight())
    return default_weights


def compute_advantages(
    *,
    algorithm: Any,
    num_samples_per_prompt: int,
    reward_mix_mode: str,
    rewards: torch.Tensor,
    prompts: List[str],
    reward_components: Optional[Dict[str, List[float]]] = None,
    reward_workers: Optional[Iterable[Any]] = None,
) -> torch.Tensor:
    """
    Compute advantages from reward tensor.

    Default path (`reward_mix_mode=reward_aggr`) uses aggregated rewards directly.
    Optional path (`reward_mix_mode=advantage_aggr`) computes advantages per reward
    component and aggregates them with reward worker weights.
    """
    if reward_mix_mode != "advantage_aggr" or not reward_components:
        return algorithm.compute_advantages(
            rewards=rewards,
            num_samples_per_prompt=num_samples_per_prompt,
            prompts=prompts,
        )

    weights = get_reward_component_weights(reward_components, reward_workers)
    weighted_advantages = torch.zeros_like(rewards)
    total_weight = 0.0

    for component_name, component_rewards in reward_components.items():
        component_tensor = torch.tensor(
            component_rewards,
            dtype=rewards.dtype,
            device=rewards.device,
        )
        if component_tensor.shape != rewards.shape:
            logger.warning(
                "Skipping reward component %s due to shape mismatch: expected %s, got %s",
                component_name,
                tuple(rewards.shape),
                tuple(component_tensor.shape),
            )
            continue

        component_advantages = algorithm.compute_advantages(
            rewards=component_tensor,
            num_samples_per_prompt=num_samples_per_prompt,
            prompts=prompts,
        )
        weight = float(weights.get(component_name, 1.0))
        weighted_advantages += component_advantages * weight
        total_weight += weight

    if total_weight <= 0:
        logger.warning(
            "reward_mix_mode=advantage_aggr but no valid reward components; "
            "falling back to aggregated reward advantages."
        )
        return algorithm.compute_advantages(
            rewards=rewards,
            num_samples_per_prompt=num_samples_per_prompt,
            prompts=prompts,
        )

    return weighted_advantages / total_weight
