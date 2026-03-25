"""Default rollout/eval functions and reward hook."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Tuple

import torch

from diffusionrl.rollout.base_types import (
    RewardHookResult,
    RolloutContext,
    RolloutFunctionResult,
)
from diffusionrl.rollout.primitives import (
    build_sampler_output_validator,
    execute_request_batches,
)
from diffusionrl.rollout.service_interface import build_eval_request_batch

logger = logging.getLogger(__name__)


def _build_reward_prompts(
    *,
    prompts: List[str],
    sample_count: int,
) -> List[str]:
    candidate = list(prompts)
    if not candidate:
        return []
    expanded: List[str] = []
    while len(expanded) < sample_count:
        expanded.extend(candidate)
    return expanded[:sample_count]


def _build_wandb_media_preview(
    *,
    sampler_outputs: List[Any],
    reward_prompts: List[str],
    rewards: torch.Tensor,
    max_items: int,
) -> Dict[str, Any] | None:
    limit = max(1, int(max_items))
    rewards_flat: List[float] = []
    if torch.is_tensor(rewards) and rewards.numel() > 0:
        rewards_flat = [float(v) for v in rewards.detach().cpu().reshape(-1).tolist()]

    images: List[Any] = []
    prompts: List[str] = []
    reward_values: List[float] = []
    global_sample_idx = 0

    for output in sampler_outputs:
        latents = getattr(output, "latents", None)
        batch_size = int(latents.shape[0]) if torch.is_tensor(latents) else 0
        decoded_images = list(getattr(output, "aux", {}).get("decoded_images") or [])
        for image_idx, image in enumerate(decoded_images):
            if len(images) >= limit:
                break
            if not hasattr(image, "save"):
                continue
            sample_idx = global_sample_idx + image_idx
            images.append(image)
            prompt = reward_prompts[sample_idx] if sample_idx < len(reward_prompts) else ""
            prompts.append(str(prompt))
            reward_val = rewards_flat[sample_idx] if sample_idx < len(rewards_flat) else 0.0
            reward_values.append(float(reward_val))
        if len(images) >= limit:
            break
        global_sample_idx += batch_size

    if not images:
        return None

    return {
        "images": images,
        "prompts": prompts,
        "rewards": reward_values,
    }

def score_rewards_hook(
    *,
    services: Any,
    request: Any,
    samples: List[Any],
    rollout_id: int,
) -> RewardHookResult:
    """Default reward hook backed by the configured reward service/pipeline."""
    rewards, reward_components = services.score_rewards(
        request=request,
        sampler_outputs=samples,
        samples_per_prompt_override=max(1, int(request.sampling.get("samples_per_prompt", 1))),
    )
    return RewardHookResult(
        rewards=rewards,
        reward_components=reward_components,
    )


def prepare_default_rollout_plan(
    *,
    services: Any,
) -> Tuple[Dict[str, Any], List[Tuple[int, Any]]]:
    """Load one prompt batch and plan request batches for the default rollout path."""
    batch = services.load_prompt_batch()
    request_batches = services.plan_request_batches(
        batch=batch,
        samples_per_prompt=services.samples_per_prompt,
    )
    return batch, request_batches


def finalize_default_rollout(
    *,
    services: Any,
    reward_hook: Callable[..., RewardHookResult],
    context: RolloutContext,
    batch: Dict[str, Any],
    request_batches: List[Tuple[int, Any]],
    request: Any,
    sampler_outputs: List[Any],
) -> RolloutFunctionResult:
    """Finish the default rollout pipeline after sampling has produced outputs."""
    prompts = batch.get("prompts", []) or []
    train_prompts = list(request.prompts)
    train_prompt_ids = request.meta.get("prompt_ids")
    sample_ids = request.meta.get("sample_ids")
    group_ids = request.meta.get("group_ids")

    reward_result = reward_hook(
        services=services,
        request=request,
        samples=sampler_outputs,
        rollout_id=int(context.rollout_id),
    )
    metadata: Dict[str, Any] = {}
    if context.collect_media_preview:
        reward_prompts = _build_reward_prompts(
            prompts=train_prompts,
            sample_count=int(reward_result.rewards.shape[0]),
        )
        media_preview = _build_wandb_media_preview(
            sampler_outputs=sampler_outputs,
            reward_prompts=reward_prompts,
            rewards=reward_result.rewards,
            max_items=context.media_max_items,
        )
        if media_preview is not None:
            metadata["wandb_media_preview"] = media_preview

    if context.debug_trace is not None:
        reward_prompts = _build_reward_prompts(
            prompts=train_prompts,
            sample_count=int(reward_result.rewards.shape[0]),
        )
        context.debug_trace.update(
            {
                "rollout_id": int(context.rollout_id),
                "debug_mode": str(services.debug_mode),
                "prompts": list(prompts),
                "train_prompts": list(train_prompts),
                "prompt_ids": list(train_prompt_ids or []),
                "sample_ids": list(sample_ids or []),
                "group_ids": list(group_ids or []),
                "reward_prompts": reward_prompts,
                "request_batches": list(request_batches),
                "sde_indices": sorted(int(v) for v in (context.sde_indices or [])),
                "sampler_outputs": sampler_outputs,
                "rewards": reward_result.rewards,
                "reward_components": reward_result.reward_components,
            }
        )

    return RolloutFunctionResult(
        request=request,
        sampler_outputs=list(sampler_outputs),
        rewards=reward_result.rewards,
        reward_components=dict(reward_result.reward_components or {}),
        metadata=metadata,
    )


def generate_rollout(
    *,
    services: Any,
    reward_hook: Callable[..., RewardHookResult],
    context: RolloutContext,
) -> RolloutFunctionResult:
    """Default rollout function: prompt batch -> sample -> reward hook."""
    batch, request_batches = prepare_default_rollout_plan(
        services=services,
    )
    if context.sde_indices is not None:
        logger.debug(
            "Received explicit rollout SDE indices: %s",
            sorted(int(i) for i in context.sde_indices)[:5],
        )

    request, sampler_outputs = execute_request_batches(
        request_batches=request_batches,
        rollout_id=context.rollout_id,
        execute_sampling_request=lambda request: services.execute_sampling_request(
            request=request,
            sde_indices=context.sde_indices,
            requirements=services.sampling_requirements,
            sampling_overrides={
                "_keep_reward_media_for_driver": bool(context.collect_media_preview),
            },
        ),
        validate_sampler_outputs=build_sampler_output_validator(
            requirements=services.sampling_requirements,
            validation_config=services.sampler_validation_config,
        ),
    )
    return finalize_default_rollout(
        services=services,
        reward_hook=reward_hook,
        context=context,
        batch=batch,
        request_batches=request_batches,
        request=request,
        sampler_outputs=sampler_outputs,
    )


def evaluate_rollout(
    *,
    services: Any,
    reward_hook: Callable[..., RewardHookResult],
    rollout_id: int,
) -> Dict[str, Any]:
    """Default evaluation function backed by the same rollout service surface."""
    request_batch = build_eval_request_batch(
        data_source=services.data_source,
        prompt_batch_size=services.prompt_batch_size,
        evaluation_settings=services.evaluation_settings,
    )
    prompts: List[str] = list(request_batch.get("prompts", []) or [])
    if not prompts:
        return {
            "rollout_id": int(rollout_id),
            "num_samples": 0,
            "mean_reward": 0.0,
            "std_reward": 0.0,
            "prompts": [],
        }

    request = services.build_request(
        batch=request_batch,
        samples_per_prompt=services.samples_per_prompt,
    )
    sampler_outputs = services.execute_sampling_request(
        request=request,
        sde_indices=None,
    )
    reward_result = reward_hook(
        services=services,
        request=request,
        samples=sampler_outputs,
        rollout_id=int(rollout_id),
    )
    rewards = reward_result.rewards
    return {
        "rollout_id": int(rollout_id),
        "num_samples": len(prompts),
        "mean_reward": rewards.mean().item(),
        "std_reward": rewards.std().item(),
        "prompts": prompts,
    }
