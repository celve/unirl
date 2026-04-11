"""Reward scoring pipeline: actor-side computation and driver-side reading."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from diffusionrl.types.reward import RewardRequest
from diffusionrl.types.sampling import RolloutSamples


logger = logging.getLogger(__name__)


def extract_images_from_output(output: RolloutSamples) -> List[Any]:
    """Extract decoded images from one rollout output."""
    if not isinstance(output, RolloutSamples):
        raise TypeError(
            "Reward stage expects RolloutSamples, "
            f"got {type(output).__name__}."
        )

    decoded_images = output.aux.get("decoded_images")
    if decoded_images is not None:
        return decoded_images if isinstance(decoded_images, list) else [decoded_images]
    raise ValueError(
        "Reward stage requires decoded_images on RolloutSamples for image rewards. "
        "Sampler output did not include decoded media."
    )


def extract_videos_from_output(output: RolloutSamples) -> List[torch.Tensor]:
    """Extract decoded videos from one rollout output."""
    if not isinstance(output, RolloutSamples):
        raise TypeError(
            "Reward stage expects RolloutSamples, "
            f"got {type(output).__name__}."
        )

    decoded_videos = None
    metadata = output.aux.get("metadata")
    if isinstance(metadata, dict):
        decoded_videos = metadata.get("decoded_videos")

    if torch.is_tensor(decoded_videos):
        if decoded_videos.dim() >= 5:
            return [video for video in decoded_videos]
        if decoded_videos.dim() == 4:
            return [decoded_videos]
    raise ValueError(
        "Reward stage requires decoded_videos metadata on RolloutSamples for video rewards. "
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


def _read_reward_payload(
    output: RolloutSamples,
) -> Optional[Tuple[List[float], Dict[str, List[float]]]]:
    raw_metadata = output.aux.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    raw_rewards = metadata.get("rewards")
    if raw_rewards is None:
        return None
    rewards = [float(v) for v in list(raw_rewards)]
    raw_components = metadata.get("component_rewards")
    normalized_components: Dict[str, List[float]] = {}
    for name, values in dict(raw_components or {}).items():
        normalized_components[str(name)] = [float(v) for v in list(values or [])]
    return rewards, normalized_components


def read_rewards(
    *,
    sampler_outputs: List[RolloutSamples],
) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
    """Read rewards from sampler output metadata (written by score_and_attach_rewards)."""
    if not sampler_outputs:
        raise RuntimeError(
            "Reward read requires at least one rollout output with scored rewards."
        )

    all_rewards: List[float] = []
    component_rewards: Dict[str, List[float]] = {}

    for output in sampler_outputs:
        payload = _read_reward_payload(output)
        if payload is None:
            raise RuntimeError(
                "Reward read requires scored rewards on all sampler outputs."
            )
        rewards, components = payload
        if len(rewards) != int(output.latents.shape[0]):
            raise ValueError(
                "Rewards length must match rollout output batch_size. "
                f"Got rewards={len(rewards)} batch_size={int(output.latents.shape[0])}."
            )
        all_rewards.extend(rewards)
        for name, values in components.items():
            if len(values) != len(rewards):
                raise ValueError(
                    "Reward component length must match reward length. "
                    f"Got component={name} len={len(values)} rewards={len(rewards)}."
                )
            component_rewards.setdefault(name, []).extend(values)

    return torch.tensor(all_rewards, dtype=torch.float32), component_rewards


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
    sampler_outputs: List[RolloutSamples],
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


def compute_rewards_from_rollout_outputs(
    *,
    reward_service: Any,
    samples_per_prompt: int,
    sampler_outputs: List[RolloutSamples],
    prompts: List[str],
    prompt_ids: Optional[List[str]] = None,
    sample_ids: Optional[List[str]] = None,
    group_ids: Optional[List[str]] = None,
    prompt_metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
    """Compute rewards from decoded rollout outputs on the sampling actor."""
    if reward_service is None:
        raise RuntimeError("Reward executor is not initialized for actor-side reward compute.")
    if any(
        _read_reward_payload(output) is not None
        for output in sampler_outputs
    ):
        raise RuntimeError(
            "Actor-side reward compute does not accept precomputed rewards on sampler "
            "outputs."
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
        response.component_rewards,
    )


def score_and_attach_rewards(
    *,
    reward_service: Any,
    output: RolloutSamples,
    prompts: List[str],
    prompt_ids: Optional[List[str]],
    sample_ids: Optional[List[str]],
    group_ids: Optional[List[str]],
    prompt_metadata: Optional[List[Optional[Dict[str, Any]]]],
    collect_media_preview: bool,
    samples_per_prompt: int,
) -> RolloutSamples:
    """Score one sampler output and write rewards into its metadata.

    After scoring, decoded media is dropped unless ``collect_media_preview``
    is set, so that only scalar rewards travel back to the driver.
    """
    rewards, component_rewards = compute_rewards_from_rollout_outputs(
        reward_service=reward_service,
        samples_per_prompt=max(1, int(samples_per_prompt)),
        sampler_outputs=[output],
        prompts=list(prompts),
        prompt_ids=prompt_ids,
        sample_ids=sample_ids,
        group_ids=group_ids,
        prompt_metadata=prompt_metadata,
    )
    raw_meta = output.aux.get("metadata")
    meta = dict(raw_meta or {})
    meta["rewards"] = [float(v) for v in rewards.tolist()]
    meta["component_rewards"] = {
        str(name): [float(v) for v in list(values or [])]
        for name, values in dict(component_rewards or {}).items()
    }
    output.aux["metadata"] = meta
    if not collect_media_preview:
        output.aux.pop("decoded_images", None)
        if isinstance(output.aux.get("metadata"), dict):
            output.aux["metadata"].pop("decoded_videos", None)
    return output


__all__ = [
    "compute_rewards_from_rollout_outputs",
    "extract_images_from_output",
    "extract_videos_from_output",
    "read_rewards",
    "resolve_reward_input_kind",
    "normalize_prompt_metadata",
    "build_request_from_rollout_outputs",
    "score_and_attach_rewards",
]
