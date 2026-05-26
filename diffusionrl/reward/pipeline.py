"""Reward scoring pipeline: actor-side computation against a ``RolloutTrack``."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
from omegaconf import DictConfig

from diffusionrl.reward.service import RewardService
from diffusionrl.types.primitives import Images, Videos
from diffusionrl.types.reward import RewardRequest
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutTrack
from diffusionrl.types.sampling import get_ar_params, get_diffusion_params

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stateless helpers
# ---------------------------------------------------------------------------


def _images_from_track(track: RolloutTrack) -> List[Any]:
    """Extract per-sample PIL images from ``track.decoded``.

    HF image processors (``CLIPProcessor`` and friends) default to
    ``do_rescale=True`` which divides by 255 — correct for uint8 PIL,
    but already-rescaled for tensor ``[0, 1]``. Convert tensors back to
    PIL here so every reward consumer sees the uint8 RGB contract.
    """
    decoded = track.decoded
    if not isinstance(decoded, Images):
        kind = type(decoded).__name__ if decoded is not None else "None"
        raise ValueError(f"Reward stage requires Images on track.decoded for image rewards; got {kind}.")
    pixels = getattr(decoded, "pixels", None)
    if pixels is None:
        raise ValueError("Reward stage: Images.pixels is None on track.decoded.")

    from diffusionrl.utils.media import tensor_frame_to_pil

    return [tensor_frame_to_pil(img) for img in pixels.unbind(0)]


def _videos_from_track(track: RolloutTrack) -> List[torch.Tensor]:
    """Extract per-sample 4D ``[C, T, H, W]`` video tensors from ``track.decoded``."""
    decoded = track.decoded
    if not isinstance(decoded, Videos):
        kind = type(decoded).__name__ if decoded is not None else "None"
        raise ValueError(f"Reward stage requires Videos on track.decoded for video rewards; got {kind}.")
    out: List[torch.Tensor] = []
    for idx, video in enumerate(decoded.to_list()):
        frames = video.frames
        if not torch.is_tensor(frames):
            raise ValueError(f"Reward stage: track.decoded video[{idx}].frames is not a Tensor.")
        if frames.dim() != 4:
            raise ValueError(
                f"Reward stage: track.decoded video[{idx}].frames must be 4D [T, C, H, W]; "
                f"got shape {tuple(frames.shape)}."
            )
        out.append(frames.permute(1, 0, 2, 3).contiguous())
    return out


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


def _build_request_for_track(
    *,
    reward_input_kind: str,
    samples_per_prompt: int,
    track: RolloutTrack,
    prompts: List[str],
    prompt_ids: List[str],
    sample_ids: List[str],
    group_ids: List[str],
    prompt_metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
) -> RewardRequest:
    """Assemble a ``RewardRequest`` from one track + its request-side texts."""
    if not prompts:
        raise ValueError("Reward request assembly requires a non-empty prompts list.")

    if reward_input_kind == "video":
        media = _videos_from_track(track)
        media_key = "videos"
    else:
        media = _images_from_track(track)
        media_key = "images"

    if len(media) != len(prompts):
        raise RuntimeError(f"Reward stage: track decoded count {len(media)} != prompts count {len(prompts)}.")

    normalized_metadata = _normalize_prompt_metadata(
        prompt_metadata=prompt_metadata,
        prompts=prompts,
        prompt_ids=prompt_ids,
        samples_per_prompt=samples_per_prompt,
    )

    request_kwargs: Dict[str, Any] = {
        "prompts": list(prompts),
        "prompt_ids": list(prompt_ids),
        "sample_ids": list(sample_ids),
        "group_ids": list(group_ids),
        "metadata": (
            normalized_metadata
            if normalized_metadata is not None and any(m is not None for m in normalized_metadata)
            else None
        ),
        media_key: media,
    }
    return RewardRequest(**request_kwargs)


# ---------------------------------------------------------------------------
# RewardPipeline — actor-side adapter binding RolloutTrack <-> RewardService
# ---------------------------------------------------------------------------


class RewardPipeline:
    """Actor-side reward adapter: scores one ``RolloutTrack`` with a ``RewardService``."""

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

    def score_and_attach(self, *, req: RolloutReq, track: RolloutTrack) -> None:
        """Score one track's decoded media and write rewards onto the track in-place.

        Reads texts off ``req.primitives['text']`` (raises if missing) and pairs
        them with ``track.sample_ids`` / ``track.group_ids``. Synthesizes
        ``prompt_ids`` from sample ids — no live scorer reads them as anything
        besides opaque strings.

        Fail-fast on per-sample failure flags so partial/corrupt rewards
        cannot silently enter advantage computation. Successes are computed
        against each sample's own requested reward set, so future per-sample
        required_rewards (multi-turn) will not raise spuriously here.
        """
        if track.rewards is not None:
            raise RuntimeError("Actor-side reward compute does not accept precomputed rewards on the track.")

        text_prim = req.primitives.get("text")
        if text_prim is None or not getattr(text_prim, "texts", None):
            raise RuntimeError(
                "RewardPipeline.score_and_attach: req.primitives['text'] must be non-empty for reward scoring."
            )
        texts = list(text_prim.texts)
        sample_ids = list(track.sample_ids)
        if len(texts) != len(sample_ids):
            # PE-joint case: the composed engine expands each prompt by N*M
            # (N LLM rewrites × M diffusion samples) before producing the
            # final track. ``req.primitives['text']`` carries the original
            # P prompts unexpanded; the leaf track has P*N*M samples per
            # actor. Replicate each prompt by the integer expansion factor
            # so scoring stays sample-aligned. This preserves the
            # "score against the original user intent" semantic for PE
            # training (reward grounds on user prompt vs final image,
            # giving the LLM rewriter a signal toward improving alignment).
            if len(sample_ids) > 0 and len(texts) > 0 and len(sample_ids) % len(texts) == 0:
                factor = len(sample_ids) // len(texts)
                # Cross-check the divisibility heuristic against the request's
                # explicit branching factors (N=PE-rewrites × M=samples-per-PE).
                # Without this, an accidentally divisible mismatch
                # (e.g. resp=2× when N×M=4) would silently mis-replicate texts.
                ar_params = get_ar_params(req.sampling_params)
                diff_params = get_diffusion_params(req.sampling_params)
                _N = int(ar_params.samples_per_prompt) if ar_params is not None else 1
                _M = int(diff_params.samples_per_prompt) if diff_params is not None else 1
                expected_factor = _N * _M
                if factor != expected_factor:
                    raise RuntimeError(
                        f"RewardPipeline.score_and_attach: implicit expansion factor "
                        f"{factor} (len(sample_ids)={len(sample_ids)} / len(texts)={len(texts)}) "
                        f"does not match sampling_params N*M={expected_factor} "
                        f"(N={_N}, M={_M}). "
                        f"Sample alignment is ambiguous."
                    )
                texts = [t for t in texts for _ in range(factor)]
            else:
                raise RuntimeError(
                    f"RewardPipeline.score_and_attach: text count {len(texts)} != "
                    f"track.sample_ids count {len(sample_ids)} and not an integer "
                    "multiple — sample alignment broken."
                )
        # Each track shard reaches reward as a single GRPO group (the mixin
        # group-splits before calling), so samples_per_prompt == group size.
        samples_per_prompt = max(1, len(sample_ids))

        request = _build_request_for_track(
            reward_input_kind=self.preferred_input_kind,
            samples_per_prompt=samples_per_prompt,
            track=track,
            prompts=texts,
            prompt_ids=[str(sid) for sid in sample_ids],
            sample_ids=sample_ids,
            group_ids=list(track.group_ids),
            prompt_metadata=None,
        )
        reward_response = self.reward_service.compute_rewards(request)

        failed = [(i, e) for i, (ok, e) in enumerate(zip(reward_response.successes, reward_response.errors)) if not ok]
        if failed:
            raise RuntimeError(
                f"Reward computation flagged {len(failed)} of "
                f"{len(reward_response.successes)} sample(s) as failure. First few: {failed[:3]}"
            )

        track.rewards = torch.tensor(reward_response.rewards, dtype=torch.float32)
        track.component_rewards = {
            str(name): torch.tensor(list(values or []), dtype=torch.float32)
            for name, values in dict(reward_response.component_rewards or {}).items()
        }

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
