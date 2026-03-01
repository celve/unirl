"""Sampling stage helpers for RolloutManager."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from diffusionrl.types.sampling import RolloutOutput, RolloutRequest


def expand_batch_for_sampling(
    batch: Dict[str, Any],
    *,
    num_samples_per_prompt: int,
) -> Tuple[Dict[str, Any], Optional[List[str]]]:
    """
    Expand batch for K-repeat sampling using prompt-major order.

    This repeats prompts along the batch dimension so that
    sampling generates num_samples_per_prompt outputs per unique prompt.

    Returns:
        (expanded_batch, train_prompts)
    """
    k = int(num_samples_per_prompt)
    if k <= 1:
        return batch, batch.get("prompts")

    prompts = batch.get("prompts")
    base_size = len(prompts) if prompts is not None else None

    if base_size is None or base_size == 0:
        return batch, prompts

    train_prompts: Optional[List[str]] = None
    if prompts is not None:
        train_prompts = [p for p in prompts for _ in range(k)]

    expanded: Dict[str, Any] = dict(batch)
    if prompts is not None:
        expanded["prompts"] = train_prompts
    if "metadata" in expanded and isinstance(expanded["metadata"], list):
        metadata = expanded["metadata"]
        if len(metadata) == base_size:
            expanded["metadata"] = [m for m in metadata for _ in range(k)]

    def _repeat(value: Any) -> Any:
        if torch.is_tensor(value) and value.shape[0] == base_size:
            return value.repeat_interleave(k, dim=0)
        return value

    for key in ("latents",):
        if key in expanded:
            expanded[key] = _repeat(expanded[key])

    return expanded, train_prompts


def distributed_sample(
    *,
    actor_group: Any,
    batch: Dict[str, Any],
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    num_frames: int,
    init_same_noise: bool,
    num_samples_per_prompt: int,
    sde_indices: Optional[Set[int]] = None,
    extra_generate_kwargs: Optional[Dict[str, Any]] = None,
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

    prompts = batch.get("prompts", [])
    if not isinstance(prompts, list) or len(prompts) == 0:
        raise ValueError(
            "distributed_sample requires non-empty text prompts. "
            "Prompt-embedding-only input is no longer supported in rollout sampling."
        )

    extra_kwargs: Dict[str, Any] = {}
    extra_kwargs["init_same_noise"] = init_same_noise
    extra_kwargs["num_samples_per_prompt"] = num_samples_per_prompt
    if isinstance(extra_generate_kwargs, dict) and extra_generate_kwargs:
        extra_kwargs.update(extra_generate_kwargs)

    request = RolloutRequest(
        prompts=prompts,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        num_frames=num_frames,
        sde_indices=sde_indices,
        decode_for_reward=True,
        latents=batch.get("latents"),
        kwargs=extra_kwargs,
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
