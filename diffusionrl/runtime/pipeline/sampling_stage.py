"""Sampling stage helpers for RolloutManager."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
from diffusionrl.types.sampling import SamplerOutput


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
    batch: Union[List[str], Dict[str, Any]],
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    num_frames: int,
    init_same_noise: bool,
    num_samples_per_prompt: int,
    sde_indices: Optional[Set[int]] = None,
) -> List[SamplerOutput]:
    """
    Sample across distributed inference actors.

    Args:
        batch: Either:
            - List[str]: List of text prompts (legacy)
            - Dict: Batch containing text prompts (prompt-only input contract)
        sde_indices: Set of timestep indices for SDE sampling (MixGRPO).
            If None, all timesteps use SDE (standard GRPO).

    Returns:
        List of SamplerOutput.
    """
    if actor_group is None:
        raise RuntimeError("No sampling actors available")

    if isinstance(batch, list):
        batch = {"prompts": batch}

    prompts = batch.get("prompts", [])
    if not isinstance(prompts, list) or len(prompts) == 0:
        raise ValueError(
            "distributed_sample requires non-empty text prompts. "
            "Prompt-embedding-only input is no longer supported in rollout sampling."
        )

    gen_kwargs = dict(
        prompts=prompts,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        num_frames=num_frames,
        sde_indices=sde_indices,
        decode_for_reward=True,
        init_same_noise=init_same_noise,
        num_samples_per_prompt=num_samples_per_prompt,
    )

    if hasattr(actor_group, "sample_batch"):
        outputs = actor_group.sample_batch(**gen_kwargs)
    else:
        outputs = actor_group.generate(**gen_kwargs)

    merged_outputs: List[SamplerOutput] = []
    for output in outputs:
        if isinstance(output, SamplerOutput):
            merged_outputs.append(output)
            continue

        if isinstance(output, (list, tuple)):
            for item in output:
                if not isinstance(item, SamplerOutput):
                    raise TypeError(
                        "Sampling stage expects SamplerOutput from actors, "
                        f"got {type(item).__name__} inside {type(output).__name__}."
                    )
                merged_outputs.append(item)
            continue

        raise TypeError(
            "Sampling stage expects SamplerOutput from actors, "
            f"got {type(output).__name__}."
        )

    return merged_outputs
