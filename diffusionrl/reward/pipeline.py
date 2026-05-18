"""Reward scoring pipeline: actor-side computation and driver-side reading."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from omegaconf import DictConfig

from diffusionrl.reward.service import RewardService
from diffusionrl.types.reward import RewardRequest
from diffusionrl.types.sample import RolloutSamples

if TYPE_CHECKING:
    from diffusionrl.types.response import RolloutResponse


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stateless helpers
# ---------------------------------------------------------------------------


def _extract_images_from_output(output: RolloutSamples) -> List[Any]:
    if output.decoded_images is None:
        raise ValueError(
            "Reward stage requires decoded_images on RolloutSamples for image rewards. "
            "Sampler output did not include decoded media."
        )
    # HF image processors (CLIPProcessor and friends) default to do_rescale=True
    # which divides by 255 — correct for uint8 PIL, but already-rescaled for
    # tensor [0,1]. Convert tensors back to PIL here so every reward consumer
    # sees the pre-refactor uint8 RGB contract.
    from diffusionrl.utils.media import tensor_frame_to_pil

    items: List[Any] = []
    for img in output.decoded_images:
        if torch.is_tensor(img):
            items.append(tensor_frame_to_pil(img))
        else:
            items.append(img)
    return items


def _extract_videos_from_output(output: RolloutSamples) -> List[torch.Tensor]:
    decoded_videos = output.decoded_videos
    if torch.is_tensor(decoded_videos):
        if decoded_videos.dim() >= 5:
            return [video for video in decoded_videos]
        if decoded_videos.dim() == 4:
            return [decoded_videos]
    raise ValueError(
        "Reward stage requires decoded_videos on RolloutSamples for video rewards. "
        "Sampler output did not include decoded video media."
    )


def _normalize_prompt_metadata(
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
                prompt_id: prompt_metadata[idx] for idx, prompt_id in enumerate(ordered_prompt_ids)
            }
            return [metadata_by_prompt_id.get(str(raw_prompt_id).strip()) for raw_prompt_id in prompt_ids]

    raise ValueError(
        "Prompt metadata must already be sample-aligned or expand via explicit prompt_ids. "
        f"Got prompts={sample_count}, metadata={len(prompt_metadata)}, "
        f"prompt_ids={len(prompt_ids) if isinstance(prompt_ids, list) else None}, "
        f"samples_per_prompt={samples_per_prompt}."
    )


def _read_reward_payload(output: RolloutSamples):
    if output.rewards is None:
        return None
    rewards = [float(v) for v in output.rewards.tolist()]
    normalized_components: Dict[str, List[float]] = {}
    for name, values in dict(output.component_rewards or {}).items():
        normalized_components[str(name)] = [float(v) for v in values.tolist()]
    return rewards, normalized_components


def _build_request_for_samples(
    *,
    reward_input_kind: str,
    samples_per_prompt: int,
    sampler_outputs: List[RolloutSamples],
    prompts: List[str],
    prompt_ids: Optional[List[str]] = None,
    sample_ids: Optional[List[str]] = None,
    group_ids: Optional[List[str]] = None,
    input_media_refs: Optional[List[List[Any]]] = None,
    prompt_metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
) -> RewardRequest:
    """Assemble a RewardRequest from sample-aligned rollout outputs."""
    if not isinstance(prompts, list) or len(prompts) == 0:
        raise ValueError("Reward request assembly requires a non-empty sample-aligned prompts list.")

    all_images: List[Any] = []
    all_videos: List[torch.Tensor] = []
    all_prompts: List[str] = []
    all_prompt_ids: List[str] = []
    all_sample_ids: List[str] = []
    all_group_ids: List[str] = []
    all_input_media_refs: List[List[Any]] = []
    all_metadata: List[Optional[Dict[str, Any]]] = []
    sample_idx = 0

    normalized_prompt_metadata = _normalize_prompt_metadata(
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
            if input_media_refs is not None and sample_idx < len(input_media_refs):
                all_input_media_refs.append(list(input_media_refs[sample_idx]))
            all_metadata.append(
                normalized_prompt_metadata[sample_idx] if normalized_prompt_metadata is not None else None
            )
            sample_idx += 1

    if reward_input_kind == "video":
        for output in sampler_outputs:
            _append_media(_extract_videos_from_output(output), all_videos)
    else:
        for output in sampler_outputs:
            _append_media(_extract_images_from_output(output), all_images)

    if not all_images and not all_videos:
        raise RuntimeError("Reward stage could not assemble any decoded media from sampler outputs.")

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
    if len(all_input_media_refs) == len(all_prompts):
        request_kwargs["input_media_refs"] = all_input_media_refs
    if all_videos:
        request_kwargs["videos"] = all_videos
    else:
        request_kwargs["images"] = all_images
    return RewardRequest(**request_kwargs)


# ---------------------------------------------------------------------------
# RewardPipeline — actor-side adapter binding RolloutResponse <-> RewardService
# ---------------------------------------------------------------------------


class RewardPipeline:
    """Actor-side reward adapter: scores RolloutResponses with a RewardService."""

    def __init__(self, reward_service: RewardService) -> None:
        self.reward_service = reward_service

    @classmethod
    def from_configs(cls, reward: DictConfig) -> "RewardPipeline":
        return cls(RewardService.from_configs(reward))

    @property
    def preferred_input_kind(self) -> str:
        """Return the decoded media kind required by the underlying executors."""
        preferred = str(getattr(self.reward_service, "preferred_input_kind", "") or "").strip().lower()
        if preferred in {"image", "video"}:
            return preferred
        raise ValueError(
            "Reward service must expose preferred_input_kind as 'image' or 'video'. "
            f"Got {preferred!r} from {type(self.reward_service).__name__}."
        )

    def score_and_attach(self, response: "RolloutResponse") -> "RolloutResponse":
        """Score one response's samples and attach rewards to typed fields in-place.

        Assumes response.samples.decoded_images (or .decoded_videos) is already
        populated — decoding remains the actor's responsibility since it owns
        the sampling engine.

        Fail-fast on per-sample failure flags so partial/corrupt rewards
        cannot silently enter advantage computation. Successes are computed
        against each sample's own requested reward set, so future per-sample
        required_rewards (multi-turn) will not raise spuriously here.
        """
        if _read_reward_payload(response.samples) is not None:
            raise RuntimeError("Actor-side reward compute does not accept precomputed rewards on sampler outputs.")
        prompts = response.request.prompts
        samples_per_prompt = max(1, int(response.request.sampling_params.num_samples_per_prompt))
        request = _build_request_for_samples(
            reward_input_kind=self.preferred_input_kind,
            samples_per_prompt=samples_per_prompt,
            sampler_outputs=[response.samples],
            prompts=prompts.prompts,
            prompt_ids=prompts.prompt_ids,
            sample_ids=prompts.sample_ids,
            group_ids=prompts.group_ids,
            input_media_refs=prompts.media_refs,
            prompt_metadata=prompts.prompt_metadata,
        )
        reward_response = self.reward_service.compute_rewards(request)

        failed = [(i, e) for i, (ok, e) in enumerate(zip(reward_response.successes, reward_response.errors)) if not ok]
        if failed:
            raise RuntimeError(
                f"Reward computation flagged {len(failed)} of "
                f"{len(reward_response.successes)} sample(s) as failure. First few: {failed[:3]}"
            )

        response.samples.rewards = torch.tensor(reward_response.rewards, dtype=torch.float32)
        response.samples.component_rewards = {
            str(name): torch.tensor(list(values or []), dtype=torch.float32)
            for name, values in dict(reward_response.component_rewards or {}).items()
        }
        return response

    def offload(self) -> None:
        self.reward_service.offload()

    def onload(self) -> None:
        self.reward_service.onload()

    def dispose(self) -> None:
        self.reward_service.dispose()

    def is_available(self) -> bool:
        return self.reward_service.is_available()


__all__ = [
    "RewardPipeline",
]
