"""Reward-side request assembly and rollout-output scoring helpers."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from diffusionrl.types.reward import RewardRequest
from diffusionrl.types.sampling import RolloutOutput

logger = logging.getLogger(__name__)


def extract_images_from_output(output: RolloutOutput) -> List[Any]:
    """Extract decoded images from one rollout output."""
    if not isinstance(output, RolloutOutput):
        raise TypeError(
            "Reward stage expects RolloutOutput, "
            f"got {type(output).__name__}."
        )

    if output.decoded_images is not None:
        return (
            output.decoded_images
            if isinstance(output.decoded_images, list)
            else [output.decoded_images]
        )
    raise ValueError(
        "Reward stage requires decoded_images on RolloutOutput for image rewards. "
        "Sampler output did not include decoded media."
    )


def extract_videos_from_output(output: RolloutOutput) -> List[torch.Tensor]:
    """Extract decoded videos from one rollout output."""
    if not isinstance(output, RolloutOutput):
        raise TypeError(
            "Reward stage expects RolloutOutput, "
            f"got {type(output).__name__}."
        )

    decoded_videos = None
    metadata = output.metadata
    if isinstance(metadata, dict):
        decoded_videos = metadata.get("decoded_videos")

    if torch.is_tensor(decoded_videos):
        if decoded_videos.dim() >= 5:
            return [video for video in decoded_videos]
        if decoded_videos.dim() == 4:
            return [decoded_videos]
    raise ValueError(
        "Reward stage requires decoded_videos metadata on RolloutOutput for video rewards. "
        "Sampler output did not include decoded video media."
    )


def resolve_reward_input_kind(*, reward_service: Any) -> str:
    """Resolve the decoded media kind required by the reward executor."""
    preferred = str(
        getattr(reward_service, "preferred_input_kind", "") or ""
    ).strip().lower()
    if preferred in {"image", "video"}:
        return preferred
    raise ValueError(
        "Reward executor must expose preferred_input_kind as 'image' or 'video'. "
        f"Got {preferred!r} from {type(reward_service).__name__}."
    )


def _read_precomputed_reward_payload(
    output: RolloutOutput,
) -> Optional[Tuple[List[float], Dict[str, List[float]]]]:
    metadata = output.metadata if isinstance(output.metadata, dict) else {}
    raw_rewards = metadata.get("precomputed_rewards")
    if raw_rewards is None:
        return None
    rewards = [float(v) for v in list(raw_rewards)]
    reward_components = dict(metadata.get("precomputed_reward_components") or {})
    normalized_components: Dict[str, List[float]] = {}
    for name, values in reward_components.items():
        normalized_components[str(name)] = [float(v) for v in list(values or [])]
    return rewards, normalized_components


def collect_precomputed_rewards(
    *,
    sampler_outputs: List[RolloutOutput],
) -> Optional[Tuple[torch.Tensor, Dict[str, List[float]]]]:
    """Collect precomputed rewards from rollout outputs when available."""
    if not sampler_outputs:
        return None

    saw_precomputed = False
    saw_missing = False
    all_rewards: List[float] = []
    reward_components: Dict[str, List[float]] = {}

    for output in sampler_outputs:
        payload = _read_precomputed_reward_payload(output)
        if payload is None:
            saw_missing = True
            continue
        saw_precomputed = True
        rewards, components = payload
        if len(rewards) != int(output.batch_size):
            raise ValueError(
                "Precomputed rewards length must match rollout output batch_size. "
                f"Got rewards={len(rewards)} batch_size={int(output.batch_size)}."
            )
        all_rewards.extend(rewards)
        for name, values in components.items():
            if len(values) != len(rewards):
                raise ValueError(
                    "Precomputed reward component length must match reward length. "
                    f"Got component={name} len={len(values)} rewards={len(rewards)}."
                )
            reward_components.setdefault(name, []).extend(values)

    if saw_precomputed and saw_missing:
        raise ValueError(
            "Mixed rollout reward execution is not supported: some sampler outputs contain "
            "precomputed rewards while others do not."
        )
    if not saw_precomputed:
        return None
    return torch.tensor(all_rewards, dtype=torch.float32), reward_components


def normalize_prompt_metadata(
    *,
    prompt_metadata: Optional[List[Optional[Dict[str, Any]]]],
    prompts: List[str],
    prompt_ids: Optional[List[str]] = None,
    samples_per_prompt: Optional[int] = None,
) -> Optional[List[Optional[Dict[str, Any]]]]:
    """Normalize prompt metadata to sample-aligned layout."""
    if not isinstance(prompt_metadata, list) or not prompt_metadata:
        return None

    sample_count = len(prompts)
    if sample_count <= 0:
        return None

    if len(prompt_metadata) == sample_count:
        return list(prompt_metadata)

    if isinstance(prompt_ids, list) and len(prompt_ids) == sample_count:
        ordered_prompt_ids: List[str] = []
        seen: set[str] = set()
        for raw_prompt_id in prompt_ids:
            prompt_id = str(raw_prompt_id).strip()
            if not prompt_id or prompt_id in seen:
                continue
            seen.add(prompt_id)
            ordered_prompt_ids.append(prompt_id)
        if len(prompt_metadata) == len(ordered_prompt_ids):
            metadata_by_prompt_id = {
                prompt_id: prompt_metadata[idx]
                for idx, prompt_id in enumerate(ordered_prompt_ids)
            }
            return [
                metadata_by_prompt_id.get(str(raw_prompt_id).strip())
                for raw_prompt_id in prompt_ids
            ]

    raise ValueError(
        "Prompt metadata must already be sample-aligned or expand via explicit prompt_ids. "
        f"Got prompts={sample_count}, metadata={len(prompt_metadata)}, "
        f"prompt_ids={len(prompt_ids) if isinstance(prompt_ids, list) else None}, "
        f"samples_per_prompt={samples_per_prompt}."
    )


def build_request_from_rollout_outputs(
    *,
    reward_service: Any,
    samples_per_prompt: int,
    sampler_outputs: List[RolloutOutput],
    prompts: List[str],
    prompt_ids: Optional[List[str]] = None,
    sample_ids: Optional[List[str]] = None,
    group_ids: Optional[List[str]] = None,
    prompt_metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
) -> RewardRequest:
    """Build a RewardRequest from sample-aligned rollout outputs."""
    if not isinstance(prompts, list) or len(prompts) == 0:
        raise ValueError(
            "Reward request assembly requires a non-empty sample-aligned prompts list."
        )

    all_images: List[Any] = []
    all_videos: List[torch.Tensor] = []
    all_prompts: List[str] = []
    all_prompt_ids: List[str] = []
    all_sample_ids: List[str] = []
    all_group_ids: List[str] = []
    all_metadata: List[Optional[Dict[str, Any]]] = []
    sample_idx = 0

    normalized_prompt_metadata = normalize_prompt_metadata(
        prompt_metadata=prompt_metadata,
        prompts=prompts,
        prompt_ids=prompt_ids,
        samples_per_prompt=samples_per_prompt,
    )

    def _append_media(items: List[Any], target: List[Any]) -> None:
        nonlocal sample_idx
        for item in items:
            if sample_idx >= len(prompts):
                raise IndexError(
                    "Reward media count exceeded prompt count while assembling RewardRequest. "
                    f"sample_idx={sample_idx}, prompts={len(prompts)}"
                )
            target.append(item)
            all_prompts.append(prompts[sample_idx])
            if prompt_ids is not None and sample_idx < len(prompt_ids):
                all_prompt_ids.append(str(prompt_ids[sample_idx]))
            if sample_ids is not None and sample_idx < len(sample_ids):
                all_sample_ids.append(str(sample_ids[sample_idx]))
            if group_ids is not None and sample_idx < len(group_ids):
                all_group_ids.append(str(group_ids[sample_idx]))
            all_metadata.append(
                normalized_prompt_metadata[sample_idx]
                if normalized_prompt_metadata is not None
                else None
            )
            sample_idx += 1

    reward_input_kind = resolve_reward_input_kind(reward_service=reward_service)
    if reward_input_kind == "video":
        for output in sampler_outputs:
            _append_media(extract_videos_from_output(output), all_videos)
    else:
        for output in sampler_outputs:
            _append_media(extract_images_from_output(output), all_images)

    if not all_images and not all_videos:
        raise RuntimeError(
            "Reward stage could not assemble any decoded media from sampler outputs."
        )

    request_kwargs: Dict[str, Any] = {
        "prompts": all_prompts,
        "metadata": all_metadata if any(m is not None for m in all_metadata) else None,
    }
    if len(all_prompt_ids) == len(all_prompts):
        request_kwargs["prompt_ids"] = all_prompt_ids
    if len(all_sample_ids) == len(all_prompts):
        request_kwargs["sample_ids"] = all_sample_ids
    if len(all_group_ids) == len(all_prompts):
        request_kwargs["group_ids"] = all_group_ids
    if all_videos:
        request_kwargs["videos"] = all_videos
    else:
        request_kwargs["images"] = all_images
    return RewardRequest(**request_kwargs)


def score_from_rollout_outputs(
    *,
    reward_service: Any,
    samples_per_prompt: int,
    sampler_outputs: List[RolloutOutput],
    prompts: List[str],
    prompt_ids: Optional[List[str]] = None,
    sample_ids: Optional[List[str]] = None,
    group_ids: Optional[List[str]] = None,
    prompt_metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
    """Score rollout outputs through the reward subsystem."""
    precomputed = collect_precomputed_rewards(sampler_outputs=sampler_outputs)
    if precomputed is not None:
        return precomputed
    if reward_service is None:
        raise RuntimeError(
            "Reward executor is not initialized and sampler outputs do not include precomputed rewards."
        )

    request = build_request_from_rollout_outputs(
        reward_service=reward_service,
        samples_per_prompt=samples_per_prompt,
        sampler_outputs=sampler_outputs,
        prompts=prompts,
        prompt_ids=prompt_ids,
        sample_ids=sample_ids,
        group_ids=group_ids,
        prompt_metadata=prompt_metadata,
    )
    response = reward_service.compute_rewards(request)
    return (
        torch.tensor(response.rewards, dtype=torch.float32),
        response.reward_components,
    )


__all__ = [
    "extract_images_from_output",
    "extract_videos_from_output",
    "resolve_reward_input_kind",
    "collect_precomputed_rewards",
    "normalize_prompt_metadata",
    "build_request_from_rollout_outputs",
    "score_from_rollout_outputs",
]
