"""
Rollout pipeline — unified stage helpers for RolloutManager.

This module keeps the rollout-side shared stages in a single file for reduced
file-jumping when reading the rollout→train data flow:

- **Sampling stage**: distributed_sample
- **Advantage stage**: get_reward_component_weights, compute_advantages
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from diffusionrl.types.sampling import RolloutOutput, RolloutRequest

logger = logging.getLogger(__name__)


# =========================================================================
# Sampling stage
# =========================================================================


def distributed_sample(
    *,
    actor_group: Any,
    request: RolloutRequest,
) -> List[RolloutOutput]:
    """
    Sample across distributed rollout actors.

    This is the natural construction point where scattered parameters are
    bundled into a :class:`RolloutRequest` before being dispatched.

    Args:
        batch: Batch containing text prompts (prompt-only input contract)
        sde_indices: Set of timestep indices for SDE sampling (MixGRPO).
            If None, all timesteps use SDE (standard GRPO).

    Returns:
        List of RolloutOutput.
    """
    if actor_group is None:
        raise RuntimeError("No sampling actors available")

    prompts = request.prompts
    if not isinstance(prompts, list) or len(prompts) == 0:
        raise ValueError(
            "distributed_sample requires non-empty text prompts. "
            "Prompt-embedding-only input is no longer supported in rollout sampling."
        )

    outputs = actor_group.generate(request)

    merged_outputs: List[RolloutOutput] = []
    for output in outputs:
        if isinstance(output, RolloutOutput):
            merged_outputs.append(output)
            continue

        if isinstance(output, (list, tuple)):
            for item in output:
                if not isinstance(item, RolloutOutput):
                    raise TypeError(
                        "Sampling stage expects RolloutOutput from actors, "
                        f"got {type(item).__name__} inside {type(output).__name__}."
                    )
                merged_outputs.append(item)
            continue

        raise TypeError(
            "Sampling stage expects RolloutOutput from actors, "
            f"got {type(output).__name__}."
        )

    return merged_outputs


# =========================================================================
# Advantage stage
# =========================================================================

def get_reward_component_weights(
    reward_components: Dict[str, List[float]],
    reward_component_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Map reward component name to configured semantic component weights."""
    default_weights = {name: 1.0 for name in reward_components.keys()}
    if reward_component_weights:
        for name, value in reward_component_weights.items():
            if name in default_weights:
                default_weights[name] = float(value)
    return default_weights


def compute_advantages(
    *,
    algorithm: Any,
    rewards: torch.Tensor,
    group_ids: Optional[List[str]] = None,
    reward_components: Optional[Dict[str, List[float]]] = None,
    reward_component_weights: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """Compute advantages from reward tensor.

    Delegates reward-mixing and advantage policy to the algorithm.
    """
    resolved_weights = reward_component_weights
    if reward_components:
        resolved_weights = get_reward_component_weights(
            reward_components,
            reward_component_weights,
        )
    return algorithm.compute_advantages_with_components(
        rewards=rewards,
        group_ids=group_ids,
        reward_components=reward_components,
        reward_component_weights=resolved_weights,
    )
