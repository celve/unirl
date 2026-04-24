"""Aggregation of multi-component reward responses."""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import torch

from diffusionrl.reward.base import BaseRewardExecutor
from diffusionrl.types.reward import RewardResponse

logger = logging.getLogger(__name__)

_Responses = List[Tuple[RewardResponse, BaseRewardExecutor]]
_Aggregator = Callable[[_Responses, int, float], RewardResponse]


def aggregate(
    method: str,
    responses: _Responses,
    batch_size: int,
    total_time: float,
) -> RewardResponse:
    """Aggregate multi-executor responses into one RewardResponse."""
    if not responses:
        return RewardResponse(
            rewards=[],
            successes=[],
            errors=[],
            compute_time=total_time,
        )

    reducer = AGGREGATORS.get(method)
    if reducer is None:
        logger.warning(
            "Unknown aggregation '%s', using weighted_sum",
            method,
        )
        reducer = _weighted_sum
    return reducer(responses, batch_size, total_time)


def _merge_successes_errors(
    responses: _Responses,
    batch_size: int,
) -> Tuple[List[bool], List[Optional[str]]]:
    successes = [True] * batch_size
    errors: List[Optional[str]] = [None] * batch_size
    for resp, _ in responses:
        for i, (success, error) in enumerate(zip(resp.successes, resp.errors)):
            if not success:
                successes[i] = False
                errors[i] = error
    return successes, errors


def _component_rewards_dict(responses: _Responses) -> Dict[str, List[float]]:
    return {executor.get_model_name(): resp.rewards for resp, executor in responses}


def _weighted_sum(
    responses: _Responses,
    batch_size: int,
    total_time: float,
) -> RewardResponse:
    total = torch.zeros(batch_size)
    total_weight = 0.0
    component_rewards: Dict[str, List[float]] = {}

    for resp, executor in responses:
        weight = executor.get_weight()
        rewards_tensor = torch.tensor(resp.rewards)
        total += rewards_tensor * weight
        total_weight += weight
        component_rewards[executor.get_model_name()] = resp.rewards

    final_rewards = (total / total_weight).tolist() if total_weight > 0 else total.tolist()
    successes, errors = _merge_successes_errors(responses, batch_size)
    return RewardResponse(
        rewards=final_rewards,
        component_rewards=component_rewards,
        successes=successes,
        errors=errors,
        compute_time=total_time,
    )


def _mean(
    responses: _Responses,
    batch_size: int,
    total_time: float,
) -> RewardResponse:
    total = torch.zeros(batch_size)
    component_rewards: Dict[str, List[float]] = {}

    for resp, executor in responses:
        total += torch.tensor(resp.rewards)
        component_rewards[executor.get_model_name()] = resp.rewards

    final_rewards = (total / len(responses)).tolist()
    successes, errors = _merge_successes_errors(responses, batch_size)
    return RewardResponse(
        rewards=final_rewards,
        component_rewards=component_rewards,
        successes=successes,
        errors=errors,
        compute_time=total_time,
    )


def _min(
    responses: _Responses,
    batch_size: int,
    total_time: float,
) -> RewardResponse:
    stacked = torch.stack([torch.tensor(resp.rewards) for resp, _ in responses])
    final_rewards = stacked.min(dim=0)[0].tolist()
    successes, errors = _merge_successes_errors(responses, batch_size)
    return RewardResponse(
        rewards=final_rewards,
        component_rewards=_component_rewards_dict(responses),
        successes=successes,
        errors=errors,
        compute_time=total_time,
    )


def _max(
    responses: _Responses,
    batch_size: int,
    total_time: float,
) -> RewardResponse:
    stacked = torch.stack([torch.tensor(resp.rewards) for resp, _ in responses])
    final_rewards = stacked.max(dim=0)[0].tolist()
    successes, errors = _merge_successes_errors(responses, batch_size)
    return RewardResponse(
        rewards=final_rewards,
        component_rewards=_component_rewards_dict(responses),
        successes=successes,
        errors=errors,
        compute_time=total_time,
    )


def _concat(
    responses: _Responses,
    batch_size: int,
    total_time: float,
) -> RewardResponse:
    successes, errors = _merge_successes_errors(responses, batch_size)
    return RewardResponse(
        rewards=responses[0][0].rewards,
        component_rewards=_component_rewards_dict(responses),
        successes=successes,
        errors=errors,
        compute_time=total_time,
    )


AGGREGATORS: Dict[str, _Aggregator] = {
    "weighted_sum": _weighted_sum,
    "mean": _mean,
    "min": _min,
    "max": _max,
    "concat": _concat,
}


__all__ = [
    "AGGREGATORS",
    "aggregate",
]
