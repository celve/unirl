"""Rollout pipeline mixin for the ``RolloutReq`` / ``RolloutResp`` path.

Operates on ``RolloutResp`` directly and leans on
:func:`diffusionrl.rollout.engine.types_compat.resp_to_samples` for the
reward-pipeline boundary — the reward pipeline still consumes the
``RolloutSamples`` shape, and a direct ``RolloutResp``-aware reward
pipeline is a follow-up.

Host class contract
-------------------
The host must provide:

- ``self.engine`` — a :class:`BaseRolloutEngine`
- ``self._rollout_plan`` — a ``RolloutPlan`` (forward_batch_size only used by chunking)
- ``self.algorithm`` — a ``GRPORolloutControl`` (owns ``compute_advantages``)
- ``self.generate(req: RolloutReq) → RolloutResp`` — provided by the host actor
- ``self.put_buffer`` / ``self.get_buffer`` / ``self.pop_buffer`` (from ``Buffer``)
- ``self._ensure_reward_pipeline()`` — returns ``RewardPipeline``

Each ``generate_buffered`` call splits the resp by group and pairs each shard
with a per-group ``RolloutReq`` shard. The pairing is held on a per-actor
``_handle_state`` dict keyed by handle id so we don't mutate ``RolloutResp`` to
carry runtime metadata across Ray.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple

import torch

from diffusionrl.distributed.transfer_queue import TransferQueueRuntime, tqbridge
from diffusionrl.rollout.engine.types_compat import resp_to_samples
from diffusionrl.types.prompts import Prompts
from diffusionrl.types.request import RolloutRequest
from diffusionrl.types.response import RolloutResponse
from diffusionrl.types.sampling import SamplingParams
from diffusionrl.utils.batched import Batched, concat_field

if TYPE_CHECKING:
    from diffusionrl.transfer.buffer import BufferHandle
    from diffusionrl.types.rollout_req import RolloutReq
    from diffusionrl.types.rollout_resp import RolloutResp


logger = logging.getLogger(__name__)


@dataclass
class _RolloutRespMeta(Batched):
    """Per-handle key for buffered RolloutResp shards.

    Mirrors :class:`diffusionrl.types.response.RolloutResponseMeta` for the
    new types path. Carried on ``BufferHandle.key`` for handle introspection
    and any future driver-side routing.
    """

    sample_ids: List[str] = concat_field(default_factory=list)
    group_ids: List[str] = concat_field(default_factory=list)


def _make_meta(resp_shard: "RolloutResp") -> _RolloutRespMeta:
    return _RolloutRespMeta(
        sample_ids=list(resp_shard.sample_ids),
        group_ids=list(resp_shard.group_ids),
    )


def _stamp_actor_reward_total(responses: List["RolloutResp"]) -> None:
    """Replace each per-handle ``reward_compute_s`` with the per-actor sum.

    Mirrors the legacy ``_stamp_actor_reward_total`` in
    ``ray/mixins/rollout_pipeline.py`` for the new shape.
    """
    actor_total = sum(float(r.reward_compute_s) for r in responses)
    for r in responses:
        r.reward_compute_s = actor_total


def _responses_to_cpu(responses: List["RolloutResp"]) -> List["RolloutResp"]:
    """Stage rollout payloads on CPU before Ray serializes them.

    With TransferQueue off (the default), returned CUDA tensors deserialize
    onto the driver's GPU 0 — every actor's shard collapses there under
    ``CUDA_VISIBLE_DEVICES`` isolation — so the driver gather + ``aggregate``
    OOMs as the global batch grows. Staging on CPU keeps that in host RAM;
    train actors move their shard back to device in ``_train_resp``. With TQ
    on, ``@tqbridge(put=True)`` dehydrates these already-CPU payloads into the
    store, so the two paths compose.
    """
    return [response.to_device("cpu") for response in responses]


def _build_legacy_response_view(
    req_shard: "RolloutReq",
    resp_shard: "RolloutResp",
    *,
    sampling_params: SamplingParams,
) -> RolloutResponse:
    """Synthesize a legacy ``RolloutResponse`` view for one resp shard.

    The reward pipeline reads ``response.request.prompts.{prompts, prompt_ids,
    sample_ids, group_ids, prompt_metadata}`` and
    ``response.request.sampling_params.num_samples_per_prompt``; everything
    else on the legacy ``RolloutRequest`` is unused. Decoded media arrive
    on ``resp.decoded`` keyed by modality slot (``"image"`` for SD3 / HI3
    t2i, ``"video"`` for WAN T2V); ``resp_to_samples`` dispatches them
    by Primitive type onto ``samples.decoded_images`` /
    ``samples.decoded_videos`` so the reward pipeline's per-kind
    extractors see the right data without an extra decode pass.
    """
    text_prim = req_shard.primitives.get("text")
    if text_prim is None or not getattr(text_prim, "texts", None):
        raise RuntimeError("RolloutPipelineMixin: req.primitives['text'] must be non-empty for reward scoring.")
    texts = list(text_prim.texts)
    sids = list(req_shard.sample_ids)
    gids = list(req_shard.group_ids)
    if len(texts) != len(sids):
        raise RuntimeError(
            f"RolloutPipelineMixin: text count {len(texts)} != sample_ids count {len(sids)}; "
            f"sample alignment broken on req shard."
        )
    # Stable synthetic prompt_ids derived from sample_ids — reward scorers that
    # don't consume them ignore them; those that do (e.g. metadata routing) see
    # a deterministic per-sample string.
    prompt_ids = [str(sid) for sid in sids]
    prompts_obj = Prompts(
        prompts=texts,
        prompt_ids=prompt_ids,
        sample_ids=sids,
        group_ids=gids,
        noise_group_ids=list(prompt_ids),
        prompt_metadata=[{} for _ in sids],
    )
    legacy_request = RolloutRequest(
        prompts=prompts_obj,
        sampling_params=sampling_params,
        collect_media_preview=False,
        media_max_items=8,
    )
    legacy_samples = resp_to_samples(resp_shard, request=legacy_request)
    return RolloutResponse(request=legacy_request, samples=legacy_samples)


class RolloutPipelineMixin:
    """Reusable generate → reward → advantage pipeline for ``RolloutReq``/``RolloutResp``."""

    # ---- per-handle state --------------------------------------------------

    def _ensure_handle_state(self) -> Dict[str, Tuple["RolloutReq", SamplingParams]]:
        if not hasattr(self, "_handle_state"):
            self._handle_state: Dict[str, Tuple["RolloutReq", SamplingParams]] = {}
        return self._handle_state

    def _split_req_by_group(self, req: "RolloutReq") -> List["RolloutReq"]:
        """Mirror :meth:`RolloutResp.split` for ``RolloutReq``.

        Inlined here (not a method on ``RolloutReq``) because the symmetry
        is local to the buffered-pipeline lifecycle — the public
        ``RolloutReq`` API stays leaner.
        """
        if not req.group_ids:
            return [req]
        groups: Dict[str, List[int]] = {}
        for i, gid in enumerate(req.group_ids):
            groups.setdefault(gid, []).append(i)
        results: List["RolloutReq"] = []
        for gid in dict.fromkeys(req.group_ids):
            indices = torch.tensor(groups[gid], dtype=torch.long)
            results.append(req.select(indices))
        return results

    # ---- buffered generation -----------------------------------------------

    def generate_buffered(self, req: "RolloutReq") -> List["BufferHandle"]:
        """Generate, split per group, buffer per shard. Pairs each handle with
        its originating per-group ``RolloutReq`` shard via ``_handle_state``.
        """
        full_resp = self.generate(req)
        resp_shards = full_resp.split()
        req_shards = self._split_req_by_group(req)
        if len(req_shards) != len(resp_shards):
            raise RuntimeError(
                f"RolloutPipelineMixin.generate_buffered: req split has "
                f"{len(req_shards)} shards but resp split has {len(resp_shards)}; "
                f"req.group_ids and resp.group_ids must align."
            )

        state = self._ensure_handle_state()
        handles: List["BufferHandle"] = []
        for req_shard, resp_shard in zip(req_shards, resp_shards):
            sampling_params = SamplingParams(num_samples_per_prompt=int(len(resp_shard.sample_ids)))
            handle = self.put_buffer(_make_meta(resp_shard), resp_shard)
            state[handle.id] = (req_shard, sampling_params)
            handles.append(handle)
        return handles

    # ---- reward attachment -------------------------------------------------

    def attach_reward(self, handle: "BufferHandle") -> None:
        """Score the buffered ``RolloutResp`` via the legacy reward pipeline.

        No ``decode_latents`` call — vllm-omni ships
        ``resp.decoded["image"].pixels`` already and ``resp_to_samples``
        surfaces them onto ``samples.decoded_images``. After scoring, copies
        ``rewards`` / ``component_rewards`` / ``media_preview`` back to the
        ``RolloutResp``.
        """
        resp: "RolloutResp" = self.get_buffer(handle)
        state = self._ensure_handle_state()
        try:
            req_shard, sampling_params = state[handle.id]
        except KeyError as exc:
            raise RuntimeError(
                f"RolloutPipelineMixin.attach_reward: handle {handle.id} has "
                f"no recorded state. Was generate_buffered called on this actor?"
            ) from exc

        legacy_view = _build_legacy_response_view(req_shard, resp, sampling_params=sampling_params)
        score_t0 = time.perf_counter()
        self._ensure_reward_pipeline().score_and_attach(legacy_view)
        elapsed = time.perf_counter() - score_t0

        resp.rewards = legacy_view.samples.rewards
        resp.component_rewards = legacy_view.samples.component_rewards
        resp.reward_compute_s = float(elapsed)

        # Media preview: gated by stage_params['collect_media_preview']. The
        # decoded pixels are already on legacy_view.samples.decoded_images, so
        # this never re-runs the VAE.
        collect_media = bool(req_shard.stage_params.get("collect_media_preview", False))
        if collect_media:
            max_items = int(req_shard.stage_params.get("media_max_items", 8))
            legacy_view.attach_media_preview(max_items=max_items)
            resp.media_preview = legacy_view.samples.media_preview
        else:
            resp.media_preview = None

    # ---- advantage computation --------------------------------------------

    def compute_advantages(self, handle: "BufferHandle") -> None:
        """Compute advantages for one buffered shard. Used outside the fused
        ``run_rollout_pipeline`` flow when callers attach reward + advantages
        per handle. The fused entrypoint normalizes across all shards instead.
        """
        resp: "RolloutResp" = self.get_buffer(handle)
        if resp.rewards is None:
            raise RuntimeError("Cannot compute advantages: rewards not attached.")
        resp.advantages = self.algorithm.compute_advantages(
            rewards=resp.rewards,
            group_ids=list(resp.group_ids),
        )

    # ---- fused pipelines ---------------------------------------------------

    @tqbridge(get=False, put=True)
    def run_rollout_pipeline(self, req: "RolloutReq") -> List["RolloutResp"]:
        """Fused actor-side rollout: generate + reward + cross-shard advantages.

        Mirrors :meth:`RolloutPipelineMixin.run_rollout_pipeline` (legacy):
        advantages are computed once across every group this actor sees so
        ``algorithm.use_global_std=True`` sees the full reward distribution
        for this shard. Per-group mean is preserved via ``group_ids``.
        """
        handles = self.generate_buffered(req)
        for h in handles:
            self.attach_reward(h)
        responses: List["RolloutResp"] = [self.pop_buffer(h) for h in handles]
        # Free per-handle state regardless of downstream success.
        state = self._ensure_handle_state()
        for h in handles:
            state.pop(h.id, None)

        _stamp_actor_reward_total(responses)
        for r in responses:
            if r.rewards is None:
                raise RuntimeError("Cannot compute advantages: rewards not attached.")
        all_rewards = torch.cat([r.rewards for r in responses])
        all_group_ids: List[str] = [gid for r in responses for gid in r.group_ids]
        all_advantages = self.algorithm.compute_advantages(
            rewards=all_rewards,
            group_ids=all_group_ids,
        )
        offset = 0
        for r in responses:
            n = int(r.rewards.shape[0])
            r.advantages = all_advantages[offset : offset + n]
            offset += n
        return _responses_to_cpu(responses)

    def run_eval_pipeline(self, req: "RolloutReq") -> List["RolloutResp"]:
        """Eval pipeline: generate + reward, no advantages."""
        handles = self.generate_buffered(req)
        for h in handles:
            self.attach_reward(h)
        responses: List["RolloutResp"] = [self.pop_buffer(h) for h in handles]
        state = self._ensure_handle_state()
        for h in handles:
            state.pop(h.id, None)
        _stamp_actor_reward_total(responses)
        return _responses_to_cpu(responses)

    # ---- TransferQueue lifecycle (parity with legacy mixin) ---------------

    def init_transferqueue_client(self, *, handoff: dict) -> None:
        """Install per-actor TransferQueue client (called from the driver)."""
        self.tq_handoff = handoff
        self.tq_runtime = TransferQueueRuntime().install()
        self.tq_client = self.tq_runtime.create_client("Actor", handoff)

    def reset_zero_copy_buffer_free(self) -> None:
        """Reset the actor's zero-copy buffer-free state at the top of each rollout step."""
        self.tq_runtime.reset_zero_copy_buffer_free()


__all__ = ["RolloutPipelineMixin"]
