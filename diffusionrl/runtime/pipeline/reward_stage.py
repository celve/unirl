"""Reward stage helpers for RolloutManager."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from diffusionrl.types.reward import RewardRequest
from diffusionrl.types.sampling import SamplerOutput

logger = logging.getLogger(__name__)


def extract_images_from_output(output: SamplerOutput) -> List[Any]:
    """
    Extract images from a sampler output.

    Args:
        output: Typed sampler output.

    Returns:
        List of images (PIL.Image or tensors)
    """
    if not isinstance(output, SamplerOutput):
        raise TypeError(
            "Reward stage expects SamplerOutput, "
            f"got {type(output).__name__}."
        )

    if output.decoded_images is not None:
        return output.decoded_images if isinstance(output.decoded_images, list) else [output.decoded_images]

    if isinstance(output.latents, torch.Tensor) and output.latents.dim() >= 3:
        return [lat for lat in output.latents]
    if output.latents is not None:
        return [output.latents]

    return []


def extract_videos_from_output(output: SamplerOutput) -> List[torch.Tensor]:
    """Extract decoded videos (preferred) or 5D tensors as video payload."""
    if not isinstance(output, SamplerOutput):
        raise TypeError(
            "Reward stage expects SamplerOutput, "
            f"got {type(output).__name__}."
        )

    decoded_videos = None
    metadata = output.metadata
    if isinstance(metadata, dict):
        decoded_videos = metadata.get("decoded_videos")
    latents = output.latents

    if torch.is_tensor(decoded_videos):
        if decoded_videos.dim() >= 5:
            return [video for video in decoded_videos]
        if decoded_videos.dim() == 4:
            return [decoded_videos]

    if torch.is_tensor(latents):
        if latents.dim() >= 5:
            return [video for video in latents]
        if latents.dim() == 4:
            return [latents]

    return []


def reward_prefers_video_inputs(reward_path: str) -> bool:
    """Best-effort switch for video-native reward workers."""
    reward_path = str(reward_path or "")
    return "VideoRewardWorker" in reward_path


def compute_rewards(
    *,
    reward_service: Any,
    reward_path: str,
    num_samples_per_prompt: int,
    sampler_outputs: List[SamplerOutput],
    prompts: List[str],
    prompt_metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
    """
    Compute rewards for generated samples using batch processing.
    Uses RewardService to compute rewards in a batched manner.

    Args:
        sampler_outputs: List of sampler outputs
        prompts: List of text prompts

    Returns:
        Tuple of:
            - Tensor of rewards [batch_size]
            - Reward components by worker/model name
    """
    all_images: List[Any] = []
    all_videos: List[torch.Tensor] = []
    all_prompts: List[str] = []
    all_metadata: List[Optional[Dict[str, Any]]] = []

    sample_idx = 0

    def _append_media(items: List[Any], target: List[Any]) -> None:
        nonlocal sample_idx
        for item in items:
            prompt_idx = sample_idx // num_samples_per_prompt
            prompt = prompts[prompt_idx % len(prompts)] if prompts else ""
            target.append(item)
            all_prompts.append(prompt)
            if prompt_metadata and len(prompt_metadata) > 0:
                all_metadata.append(prompt_metadata[prompt_idx % len(prompt_metadata)])
            else:
                all_metadata.append(None)
            sample_idx += 1

    prefer_video_inputs = reward_prefers_video_inputs(reward_path)
    if prefer_video_inputs:
        all_videos = []
        for output in sampler_outputs:
            videos = extract_videos_from_output(output)
            if not videos:
                prefer_video_inputs = False
                all_videos = []
                all_prompts = []
                all_metadata = []
                sample_idx = 0
                break
            _append_media(videos, all_videos)

    if not prefer_video_inputs:
        for output in sampler_outputs:
            images = extract_images_from_output(output)
            _append_media(images, all_images)

    if not all_images and not all_videos:
        logger.warning("No media extracted from sampler outputs")
        return torch.zeros(len(sampler_outputs), dtype=torch.float32), {}

    request_kwargs: Dict[str, Any] = {
        "prompts": all_prompts,
        "metadata": all_metadata if any(m is not None for m in all_metadata) else None,
    }
    if prefer_video_inputs and all_videos:
        request_kwargs["videos"] = all_videos
    else:
        request_kwargs["images"] = all_images
    request = RewardRequest(**request_kwargs)

    response = reward_service.compute_rewards(request)
    return torch.tensor(response.rewards, dtype=torch.float32), response.reward_components
