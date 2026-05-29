"""Reward scoring pipeline: actor-side computation against a ``RolloutTrack``."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
from omegaconf import DictConfig

from diffusionrl.distributed.group.dispatch import Dispatch, distributed
from diffusionrl.distributed.group.remote import Remote
from diffusionrl.reward.service import RewardService
from diffusionrl.types.reward import RewardRequest
from diffusionrl.types.rollout_req import PrimitiveValue, RolloutReq
from diffusionrl.types.rollout_resp import RolloutTrack, _track_with_field
from diffusionrl.types.sampling import get_ar_params, get_diffusion_params

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stateless helpers
# ---------------------------------------------------------------------------

_KIND_TO_KEY = {"image": "image", "video": "video", "text": "text"}


def _normalize_prompt_metadata(
    *,
    prompt_metadata: Optional[List[Optional[Dict[str, Any]]]],
    sample_count: int,
    prompt_ids: Optional[List[str]] = None,
    samples_per_prompt: Optional[int] = None,
) -> Optional[List[Optional[Dict[str, Any]]]]:
    """Normalize prompt metadata to sample-aligned layout."""
    if not isinstance(prompt_metadata, list) or not prompt_metadata:
        return None

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
        f"Got sample_count={sample_count}, metadata={len(prompt_metadata)}, "
        f"prompt_ids={len(prompt_ids) if isinstance(prompt_ids, list) else None}, "
        f"samples_per_prompt={samples_per_prompt}."
    )


def _build_request_for_track(
    *,
    reward_input_kind: str,
    samples_per_prompt: int,
    track: RolloutTrack,
    req_primitives: Dict[str, PrimitiveValue],
    prompt_ids: List[str],
    sample_ids: List[str],
    group_ids: List[str],
    prompt_metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
) -> RewardRequest:
    """Assemble a ``RewardRequest`` from one track + its request-side primitives."""
    decoded = track.decoded
    if decoded is None:
        raise ValueError("Reward request assembly requires non-None track.decoded.")

    gen_key = _KIND_TO_KEY.get(reward_input_kind)
    if gen_key is None:
        raise ValueError(f"Unknown reward_input_kind={reward_input_kind!r}. Expected one of {sorted(_KIND_TO_KEY)}.")

    normalized_metadata = _normalize_prompt_metadata(
        prompt_metadata=prompt_metadata,
        sample_count=len(sample_ids),
        prompt_ids=prompt_ids,
        samples_per_prompt=samples_per_prompt,
    )

    return RewardRequest(
        primitives=dict(req_primitives),
        generated={gen_key: decoded},
        prompt_ids=list(prompt_ids),
        sample_ids=list(sample_ids),
        group_ids=list(group_ids),
        metadata=(
            normalized_metadata
            if normalized_metadata is not None and any(m is not None for m in normalized_metadata)
            else None
        ),
    )


# ---------------------------------------------------------------------------
# RewardPipeline — actor-side adapter binding RolloutTrack <-> RewardService
# ---------------------------------------------------------------------------


class RewardPipeline(Remote):
    """Actor-side reward adapter: scores one ``RolloutTrack`` with a ``RewardService``."""

    def __init__(self, reward_service: RewardService) -> None:
        super().__init__()
        self.reward_service = reward_service

    @classmethod
    def from_configs(cls, reward: DictConfig) -> "RewardPipeline":
        return cls(RewardService.from_configs(reward))

    @property
    def preferred_input_kind(self) -> str:
        """Return the decoded media kind required by the underlying executors."""
        preferred = str(getattr(self.reward_service, "preferred_input_kind", "") or "").strip().lower()
        if preferred in {"image", "video", "text"}:
            return preferred
        raise ValueError(
            "Reward service must expose preferred_input_kind as 'image', 'video', or 'text'. "
            f"Got {preferred!r} from {type(self.reward_service).__name__}."
        )

    @distributed(dispatch_mode=Dispatch.DP_ALL)
    def score_and_attach(self, *, req: RolloutReq, track: RolloutTrack) -> RolloutTrack:
        """Score one track's decoded media and return a copy with rewards attached.

        Copies ``req.primitives`` (input context) into the reward request and
        pairs with ``track.decoded`` (generated output). For PE-joint tracks
        where the request has fewer samples than the track (N×M expansion),
        each primitive is replicated by the expansion factor.

        Returns a new :class:`RolloutTrack` with ``rewards`` and
        ``component_rewards`` populated; the input track is left unchanged so
        the result can flow back through Handle dispatch (pytree_merge across
        DP shards) without relying on worker-local mutation.

        Fail-fast on per-sample failure flags so partial/corrupt rewards
        cannot silently enter advantage computation.
        """
        if track.rewards is not None:
            raise RuntimeError("Actor-side reward compute does not accept precomputed rewards on the track.")

        sample_ids = list(track.sample_ids)
        req_primitives: Dict[str, PrimitiveValue] = dict(req.primitives)

        # Determine the request-side batch size from any primitive.
        req_batch = 0
        for v in req_primitives.values():
            if v is not None:
                req_batch = len(v)
                break

        _expanded_metadata = None  # set in the expansion branch below if needed

        if req_batch > 0 and req_batch != len(sample_ids):
            # PE-joint expansion: req has P prompts, track has P*N*M samples.
            if len(sample_ids) % req_batch == 0:
                factor = len(sample_ids) // req_batch
                ar_params = get_ar_params(req.sampling_params)
                diff_params = get_diffusion_params(req.sampling_params)
                _N = int(ar_params.samples_per_prompt) if ar_params is not None else 1
                _M = int(diff_params.samples_per_prompt) if diff_params is not None else 1
                expected_factor = _N * _M
                if factor != expected_factor:
                    raise RuntimeError(
                        f"RewardPipeline.score_and_attach: implicit expansion factor "
                        f"{factor} (track={len(sample_ids)} / req={req_batch}) "
                        f"does not match sampling_params N*M={expected_factor} "
                        f"(N={_N}, M={_M}). Sample alignment is ambiguous."
                    )
                req_primitives = {k: v.repeat_interleave(factor) for k, v in req_primitives.items()}
                # Expand metadata in sync with primitives so prompt_metadata
                # aligns with sample_ids (one entry per sample).
                if req.metadata:
                    _expanded_metadata = [m for m in req.metadata for _ in range(factor)]
            else:
                raise RuntimeError(
                    f"RewardPipeline.score_and_attach: req batch {req_batch} != "
                    f"track.sample_ids count {len(sample_ids)} and not an integer "
                    "multiple — sample alignment broken."
                )

        samples_per_prompt = max(1, len(sample_ids))

        _final_metadata = (
            _expanded_metadata if _expanded_metadata is not None else (list(req.metadata) if req.metadata else None)
        )

        request = _build_request_for_track(
            reward_input_kind=self.preferred_input_kind,
            samples_per_prompt=samples_per_prompt,
            track=track,
            req_primitives=req_primitives,
            prompt_ids=[str(sid) for sid in sample_ids],
            sample_ids=sample_ids,
            group_ids=list(track.group_ids),
            prompt_metadata=_final_metadata,
        )
        reward_response = self.reward_service.compute_rewards(request)

        failed = [(i, e) for i, (ok, e) in enumerate(zip(reward_response.successes, reward_response.errors)) if not ok]
        if failed:
            raise RuntimeError(
                f"Reward computation flagged {len(failed)} of "
                f"{len(reward_response.successes)} sample(s) as failure. First few: {failed[:3]}"
            )

        rewards = torch.tensor(reward_response.rewards, dtype=torch.float32)
        component_rewards = {
            str(name): torch.tensor(list(values or []), dtype=torch.float32)
            for name, values in dict(reward_response.component_rewards or {}).items()
        }
        track = _track_with_field(track, "rewards", rewards)
        return _track_with_field(track, "component_rewards", component_rewards)

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
