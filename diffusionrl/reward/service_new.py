"""Merged reward service: own per-component executors, dispatch + aggregate,
and score one ``RolloutTrack`` against ``(RolloutReq, decoded)`` in-place.

Side-by-side staging for the ``RewardService`` + ``RewardPipeline`` merge.
Once callers migrate from ``RewardPipeline`` to this class, the old files go
away and this gets renamed to ``service.py`` / ``RewardService``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import torch
from omegaconf import DictConfig

from diffusionrl.config.instantiate import build, materialize
from diffusionrl.distributed.group.dispatch import Dispatch, distributed
from diffusionrl.distributed.group.remote import Remote
from diffusionrl.types.reward import RewardRequest, RewardResponse
from diffusionrl.types.rollout_req import PrimitiveValue, RolloutReq
from diffusionrl.types.rollout_resp import RolloutTrack, _track_with_field
from diffusionrl.types.sampling import get_ar_params, get_diffusion_params

from .aggregation import aggregate
from .base import BaseRewardExecutor, BaseRewardScorer, InProcessRewardExecutor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stateless helpers (lifted from pipeline.py)
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
# RewardServiceNew — executors + dispatch + actor-side adapter
# ---------------------------------------------------------------------------


class RewardServiceNew(Remote):
    """Owns per-component executors; dispatches + aggregates a ``RewardRequest``,
    and scores one ``RolloutTrack`` against ``(req, decoded)`` in-place.

    Side-by-side staging class for the eventual merge of ``RewardService`` and
    ``RewardPipeline``. Self-contained: does **not** wrap an existing
    ``RewardService``.
    """

    def __init__(
        self,
        executors: Optional[List[BaseRewardExecutor]] = None,
        aggregation_method: str = "weighted_sum",
    ) -> None:
        super().__init__()
        self.executors: List[BaseRewardExecutor] = list(executors or [])
        self.reward_aggregation_method = str(aggregation_method)

        logger.info(
            "RewardServiceNew initialized with %d executor(s), aggregation=%s",
            len(self.executors),
            self.reward_aggregation_method,
        )

    @classmethod
    def from_configs(cls, reward: DictConfig) -> "RewardServiceNew":
        """Build a RewardServiceNew from the raw ``cfg.reward`` DictConfig.

        Materializes the parent for top-level fields (``aggregation_method``,
        ``base_device``) plus per-component weights, then dispatches each
        component via :func:`diffusionrl.config.instantiate.build`. Scorer
        results are wrapped in :class:`InProcessRewardExecutor`; executor
        results pass through (HTTP).
        """
        rc = materialize(reward)
        executors: List[BaseRewardExecutor] = []
        for key in reward.components:
            cfg_node = reward.components[key]
            spec = rc.components[key]
            built = build(cfg_node, base_device=rc.base_device)
            if isinstance(built, BaseRewardScorer):
                built = InProcessRewardExecutor(built, weight=spec.weight)
            executors.append(built)
        return cls(
            executors=executors,
            aggregation_method=rc.aggregation_method,
        )

    @property
    def preferred_input_kind(self) -> str:
        """Return the decoded media kind required by the configured executors."""
        kinds = {
            str(getattr(executor, "preferred_input_kind", "") or "").strip().lower() for executor in self.executors
        }
        kinds.discard("")
        if not kinds:
            raise ValueError("RewardServiceNew has no executors; cannot determine preferred_input_kind.")
        if len(kinds) > 1:
            raise ValueError(
                f"Mixed reward input kinds in one service are not supported. Configured kinds={sorted(kinds)}."
            )
        kind = next(iter(kinds))
        if kind not in {"image", "video", "text"}:
            raise ValueError(
                f"Reward executor must expose preferred_input_kind as 'image', 'video', or 'text'. Got {kind!r}."
            )
        return kind

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        """Compute rewards using configured executors."""
        if not self.executors:
            return RewardResponse(
                rewards=[0.0] * request.batch_size,
                successes=[False] * request.batch_size,
                errors=["No executors configured"] * request.batch_size,
                compute_time=0.0,
            )

        start_time = time.time()

        if len(self.executors) == 1:
            return self.executors[0].compute_rewards(request)

        responses = []
        for executor in self.executors:
            try:
                resp = executor.compute_rewards(request)
            except Exception as e:
                raise RuntimeError(
                    f"Reward executor {executor.get_model_name()!r} failed during compute_rewards "
                    f"(batch_size={request.batch_size}): {type(e).__name__}: {e}"
                ) from e
            responses.append((resp, executor))

        batch_size = responses[0][0].batch_size if responses else 0
        return aggregate(
            self.reward_aggregation_method,
            responses,
            batch_size,
            time.time() - start_time,
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
        the result flows back through Handle dispatch (pytree_merge across DP
        shards) without relying on worker-local mutation.

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
                        f"RewardServiceNew.score_and_attach: implicit expansion factor "
                        f"{factor} (track={len(sample_ids)} / req={req_batch}) "
                        f"does not match sampling_params N*M={expected_factor} "
                        f"(N={_N}, M={_M}). Sample alignment is ambiguous."
                    )
                req_primitives = {k: v.repeat_interleave(factor) for k, v in req_primitives.items()}
            else:
                raise RuntimeError(
                    f"RewardServiceNew.score_and_attach: req batch {req_batch} != "
                    f"track.sample_ids count {len(sample_ids)} and not an integer "
                    "multiple — sample alignment broken."
                )

        samples_per_prompt = max(1, len(sample_ids))

        request = _build_request_for_track(
            reward_input_kind=self.preferred_input_kind,
            samples_per_prompt=samples_per_prompt,
            track=track,
            req_primitives=req_primitives,
            prompt_ids=[str(sid) for sid in sample_ids],
            sample_ids=sample_ids,
            group_ids=list(track.group_ids),
            prompt_metadata=list(req.metadata) if req.metadata else None,
        )
        reward_response = self.compute_rewards(request)

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

    def is_available(self) -> bool:
        return any(executor.is_available() for executor in self.executors)

    def offload(self) -> None:
        for executor in self.executors:
            executor.offload()
        logger.debug("RewardServiceNew offloaded %d executor(s)", len(self.executors))

    def onload(self) -> None:
        for executor in self.executors:
            executor.onload()
        logger.debug("RewardServiceNew onloaded %d executor(s)", len(self.executors))

    def dispose(self) -> None:
        for executor in self.executors:
            executor.dispose()
        self.executors = []
        logger.info("RewardServiceNew disposed")


__all__ = [
    "RewardServiceNew",
]
