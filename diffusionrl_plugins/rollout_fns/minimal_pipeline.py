"""Minimal function-style rollout pipeline example."""

from __future__ import annotations

from typing import Any, List

from diffusionrl.types.sampling import RolloutOutput
from diffusionrl.types.training_batch import TrainingBatch


def minimal_pipeline(
    prompts: List[str],
    engine: Any,
    reward_and_advantage_fn: Any,
    assemble_batch: Any,
    **_: Any,
) -> TrainingBatch:
    """
    Function-style rollout pipeline.

    Expected injected callables from RolloutManager:
    - engine.generate(prompts, ...)
    - reward_and_advantage_fn(outputs, ...)
    - assemble_batch(outputs, rewards=..., advantages=..., ...)
    """
    outputs: List[RolloutOutput] = engine.generate(prompts)
    rewards, advantages, _reward_components = reward_and_advantage_fn(outputs)
    return assemble_batch(outputs, rewards=rewards, advantages=advantages)

