"""SGLang ``GenerationResult`` list → ``RolloutResp`` translator.

Single free function ``_to_rollout_resp(req, results, *, cfg, num_steps, shift,
sde_indices, use_native_logprob)`` produces:

- ``resp.rollout_traces['image']`` = :class:`LatentSegment` with ``latents``,
  ``sigmas``, ``indices``, ``sample_indices`` always populated; ``sde_logp`` +
  ``sde_indices`` populated when ``use_native_logprob`` and the algorithm
  requested SDE log-probs.
- ``resp.decoded['image']`` = :class:`Images` (``float32 [B, C, H, W]`` in
  ``[0, 1]``) built from SGLang's per-result ``samples`` output. Video samples
  surface as ``[C, T, H, W]`` and are deferred (TODO — first video consumer).
- ``resp.conditions['text']`` + (when CFG active) ``resp.conditions[
  'negative_text']`` populated from SGLang's ``prompt_embeds`` /
  ``pooled_prompt_embeds`` / ``encoder_attention_mask`` / ``negative_*``
  outputs when ``cfg.populate_conditions=True``. The slot key
  ``"negative_text"`` matches the SD3 / Mochi typed-container convention so
  trainer-side ``*DiffusionConditions.from_dict(resp.conditions)`` consumers
  pick up the negative branch automatically.

Trajectory validation (T+1 invariant, sigma-schedule cross-check) and
selective-trim heuristics (``compute_trajectory_positions``) port verbatim
from legacy ``samplers/sglang/response.py``. Model-typed
``ForwardContext`` build does NOT port over — trainer-side replay
reconstructs the typed container from ``resp.conditions``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

import torch

from diffusionrl.config.require import require
from diffusionrl.rollout.engine.sglang._sample_decode import decode_sample
from diffusionrl.rollout.engine.sglang._text_fusion import fuse_text_encoder_outputs
from diffusionrl.rollout.engine.sglang.config import SGLangEngineConfig
from diffusionrl.rollout.engine.sigma_verify import verify_engine_used_sigmas
from diffusionrl.types.conditions.text import TextEmbedCondition
from diffusionrl.types.primitives import Images
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutResp
from diffusionrl.types.segments.latent import LatentSegment, make_image_segment
from diffusionrl.types.trajectory_store import compute_trajectory_positions

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.utils import GenerationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trajectory alignment + sigma cross-check (ported from legacy)
# ---------------------------------------------------------------------------


def _derive_timestep_alignment(
    *,
    trajectories_tensor: torch.Tensor,
    expected_sigmas: torch.Tensor,
    results: Sequence[Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate T+1 trajectory shape and verify SGLang used the σ we sent.

    ``expected_sigmas`` is the σ schedule the engine pinned on
    ``RolloutReq.sigmas`` and forwarded via SGLang's
    ``SamplingParams.sigmas`` → ``set_timesteps(sigmas=...)``. SGLang
    echoes the same values back via ``trajectory_timesteps`` per
    result; :func:`verify_engine_used_sigmas` asserts elementwise
    equality (with dynamic scale-normalization for sglang builds that
    emit raw ``sigma * num_train_timesteps`` instead of [0, 1] —
    handled inside the helper). Together these guarantee SGLang
    rollout and training-side replay (which reads ``segment.sigmas``)
    used numerically identical σ schedules.

    Design note: main-unified-base commit ``707cc609`` switched to
    "extract σ from sglang's trajectory_timesteps as SOT" (with a
    warning-only drift check). That approach silently keeps training
    going when sglang deviates from main-repo intent — the GRPO
    invariant holds but the model trains under un-intended σ. We keep
    the **main-repo-as-SOT + fatal drift assert** direction here
    because (a) celve fork ``2c5a2ecec`` ensures sglang honors the
    sigmas we send, (b) any deviation surfaces a real supply-chain
    bug we want loud, not silent.
    """
    traj_len = int(trajectories_tensor.shape[1])
    expected_len = int(expected_sigmas.shape[0])
    require(
        traj_len == expected_len,
        f"SGLang trajectory length {traj_len} != expected_sigmas length {expected_len}. "
        f"Modern SGLang prepends initial latents at "
        f"sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py so "
        f"trajectory carries T+1 latents; expected_sigmas (from req.sigmas) is T+1 "
        f"too. Upgrade SGLang or fix the sampler to emit a T+1 trajectory.",
    )
    expected_cpu = expected_sigmas.detach().to(torch.float32).cpu()
    step_indices = torch.arange(expected_len, dtype=torch.long)
    for i, result in enumerate(results):
        verify_engine_used_sigmas(
            getattr(result, "trajectory_timesteps", None),
            expected=expected_cpu,
            engine_name=f"sglang (result {i})",
        )
    return expected_cpu, step_indices


# ---------------------------------------------------------------------------
# Segment build
# ---------------------------------------------------------------------------


def _build_image_segment(
    results: Sequence["GenerationResult"],
    *,
    expected_sigmas: torch.Tensor,
    num_steps: int,
    sde_indices: Optional[List[int]],
    use_native_logprob: bool,
) -> LatentSegment:
    """Pack per-result trajectory tensors into one batched ``LatentSegment``."""
    trajectory_items: List[torch.Tensor] = []
    for result in results:
        traj = getattr(result, "trajectory_latents", None)
        require(traj is not None, "SGLang result missing trajectory_latents")
        trajectory_items.append(traj.detach().cpu())
    trajectories_tensor = torch.cat(trajectory_items, dim=0)

    sigmas, step_indices = _derive_timestep_alignment(
        trajectories_tensor=trajectories_tensor,
        expected_sigmas=expected_sigmas,
        results=results,
    )

    # NOTE: the warning-only ``_verify_sglang_timesteps`` block from
    # main-unified-base commit ``707cc609`` is deliberately *not*
    # ported. ``_derive_timestep_alignment`` above already runs the
    # fatal-on-mismatch :func:`verify_engine_used_sigmas` per result;
    # adding a second (warning-level) verify pass would be redundant
    # and weaken the "fail loudly on supply-chain drift" property.

    # Selective trim: when only a subset of trajectory positions is referenced
    # by the SDE step set, drop unused columns to save Ray IPC bandwidth.
    # ``compute_trajectory_positions`` returns only the (i, i+1) pairs for
    # SDE-gated steps — for ``sde_indices={5}`` at T=10 that's just
    # {5, 6}, *not* the terminal clean latent at position T=10. Downstream
    # legacy bridges read ``samples.latents = seg.latents[:, -1]`` and
    # feed that to VAE decode, so we always preserve T as well to keep
    # the clean image latent available.
    traj_len = int(trajectories_tensor.shape[1])
    indices_t: torch.Tensor = step_indices
    if sde_indices is not None and len(sde_indices) < num_steps:
        needed = set(compute_trajectory_positions(set(sde_indices), num_steps))
        needed.add(int(num_steps))  # always preserve terminal clean latent
        keep_cols = sorted(p for p in needed if 0 <= p < traj_len)
        if keep_cols and len(keep_cols) < traj_len:
            trajectories_tensor = trajectories_tensor[:, keep_cols]
            indices_t = torch.tensor(keep_cols, dtype=torch.long)

    # sde_indices: always populated (trainer needs to know which steps to replay).
    # sde_logp: only populated in native mode; replay mode computes it on trainer side.
    sde_indices_t: Optional[torch.Tensor] = (
        torch.tensor(list(sde_indices), dtype=torch.long)
        if sde_indices is not None
        else torch.arange(num_steps, dtype=torch.long)
    )
    sde_logp: Optional[torch.Tensor] = None
    if use_native_logprob:
        per_result_log_probs: List[Optional[torch.Tensor]] = [
            result.trajectory_log_probs.detach().cpu()
            if getattr(result, "trajectory_log_probs", None) is not None
            else None
            for result in results
        ]
        missing = [i for i, lp in enumerate(per_result_log_probs) if lp is None]
        if missing:
            raise RuntimeError(
                f"logprob_source='native' but SGLang did not return usable "
                f"trajectory_log_probs for {len(missing)}/{len(results)} result(s) "
                f"(first missing index={missing[0]}). Pin a SGLang build that emits "
                f"trajectory_log_probs of shape [B, T] or switch logprob_source='replay'."
            )
        log_prob_tensor = torch.cat([lp for lp in per_result_log_probs if lp is not None], dim=0)
        # trajectory_log_probs shape: [B, T] (one entry per SDE transition).
        # When sde_indices is None (rollout SDE mode used full schedule), the
        # second dim equals num_steps. When sde_indices is a subset, SGLang is
        # supposed to emit log-probs at the requested transitions only — but
        # some SGLang builds always emit the full schedule. Tolerate that case
        # by selecting the requested columns; only fail on shapes that can't
        # be reconciled either way.
        s_dim = int(log_prob_tensor.shape[1])
        expected_s = len(sde_indices) if sde_indices is not None else num_steps
        if s_dim == num_steps and sde_indices is not None and expected_s < num_steps:
            # Server emitted full schedule; slice down to the requested SDE indices.
            keep_idx = torch.tensor(sorted(int(i) for i in sde_indices), dtype=torch.long)
            log_prob_tensor = log_prob_tensor.index_select(1, keep_idx)
            s_dim = int(log_prob_tensor.shape[1])
        require(
            s_dim == expected_s,
            f"SGLang trajectory_log_probs shape {tuple(log_prob_tensor.shape)} second "
            f"dim={s_dim} does not match expected SDE-step count {expected_s}. "
            f"sigma_schedule / num_inference_steps / sde_indices drift — fix the "
            f"source rather than fall back to replay silently.",
        )
        sde_logp = log_prob_tensor

    batch_size = int(trajectories_tensor.shape[0])
    return make_image_segment(
        latents=trajectories_tensor,
        sigmas=sigmas,
        indices=indices_t,
        sde_logp=sde_logp,
        sde_indices=sde_indices_t,
        sample_indices=torch.arange(batch_size, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Decoded media
# ---------------------------------------------------------------------------


def _build_decoded_images(
    results: Sequence["GenerationResult"],
) -> Optional[Images]:
    """Stack per-result decoded ``samples`` into ``Images.pixels [B, C, H, W]``."""
    per_sample_tensors: List[torch.Tensor] = []
    skipped_video = False
    for result in results:
        canonical = decode_sample(getattr(result, "samples", None))
        if canonical is None:
            continue
        if canonical.dim() == 3:
            per_sample_tensors.append(canonical.to(torch.float32))
        elif canonical.dim() == 4:
            # [C, T, H, W] — video. TODO: surface as Videos primitive once a
            # video reward consumer exists. For now drop with a warning so
            # reward scoring fails fast rather than silently using the wrong
            # tensor.
            skipped_video = True
        else:
            raise RuntimeError(
                f"_build_decoded_images: unexpected canonical media rank "
                f"{canonical.dim()}; want 3 (image) or 4 (video)."
            )
    if skipped_video:
        logger.warning(
            "SGLang result contained 4D (video) samples — Videos primitive packing "
            "is not yet implemented in the response translator; dropping. "
            "Add a Videos branch when a video reward consumer lands."
        )
    if not per_sample_tensors:
        return None
    return Images(pixels=torch.stack(per_sample_tensors, dim=0))


# ---------------------------------------------------------------------------
# Conditions packing
# ---------------------------------------------------------------------------


def _build_text_conditions(
    results: Sequence["GenerationResult"],
) -> Tuple[Optional[TextEmbedCondition], Optional[TextEmbedCondition]]:
    """Fuse per-result encoder outputs into ``text`` + optional ``negative_text``.

    Returns ``(text_cond, neg_text_cond)``. Either may be ``None`` when the
    corresponding source field was missing across all results (e.g. no CFG →
    no negative branch). Concat is dim-0 across results.
    """
    prompt_embeds_list: List[torch.Tensor] = []
    pooled_list: List[torch.Tensor] = []
    mask_list: List[torch.Tensor] = []
    neg_embeds_list: List[torch.Tensor] = []
    neg_pooled_list: List[torch.Tensor] = []

    for result in results:
        embeds = fuse_text_encoder_outputs(getattr(result, "prompt_embeds", None))
        require(
            embeds is not None,
            "SGLang result missing prompt_embeds — request must pin return_prompt_embeds=True",
        )
        prompt_embeds_list.append(embeds.detach().cpu())

        pooled = fuse_text_encoder_outputs(getattr(result, "pooled_prompt_embeds", None))
        if pooled is not None:
            pooled_list.append(pooled.detach().cpu())

        attn_mask = fuse_text_encoder_outputs(getattr(result, "encoder_attention_mask", None))
        if attn_mask is not None:
            mask_list.append(attn_mask.detach().cpu())

        neg_embeds = fuse_text_encoder_outputs(getattr(result, "negative_prompt_embeds", None))
        if neg_embeds is not None:
            neg_embeds_list.append(neg_embeds.detach().cpu())

        neg_pooled = fuse_text_encoder_outputs(getattr(result, "neg_pooled_prompt_embeds", None))
        if neg_pooled is not None:
            neg_pooled_list.append(neg_pooled.detach().cpu())

    embeds_cat = torch.cat(prompt_embeds_list, dim=0) if prompt_embeds_list else None
    text_cond = (
        TextEmbedCondition(
            embeds=embeds_cat,
            pooled=torch.cat(pooled_list, dim=0) if pooled_list else None,
            attn_mask=torch.cat(mask_list, dim=0) if mask_list else None,
        )
        if embeds_cat is not None
        else None
    )

    neg_embeds_cat = torch.cat(neg_embeds_list, dim=0) if neg_embeds_list else None
    neg_text_cond = (
        TextEmbedCondition(
            embeds=neg_embeds_cat,
            pooled=torch.cat(neg_pooled_list, dim=0) if neg_pooled_list else None,
            attn_mask=None,
        )
        if neg_embeds_cat is not None
        else None
    )

    return text_cond, neg_text_cond


# ---------------------------------------------------------------------------
# Top-level translator
# ---------------------------------------------------------------------------


def _to_rollout_resp(
    req: RolloutReq,
    results: Sequence["GenerationResult"],
    *,
    cfg: SGLangEngineConfig,
    num_steps: int,
    sde_indices: Optional[List[int]],
    use_native_logprob: bool,
) -> RolloutResp:
    """Translate one SGLang batch result into the typed ``RolloutResp`` container."""
    require(bool(results), "_to_rollout_resp: SGLang returned no results")
    require(
        req.sigmas is not None,
        "_to_rollout_resp: req.sigmas must be set (SGLangRolloutEngine populates "
        "it before dispatch). Without it we can't verify SGLang used the same "
        "schedule the trainer will replay against.",
    )

    segment = _build_image_segment(
        results,
        expected_sigmas=req.sigmas,
        num_steps=num_steps,
        sde_indices=sde_indices,
        use_native_logprob=use_native_logprob,
    )

    rollout_traces = {"image": segment}

    decoded: dict = {}
    decoded_images = _build_decoded_images(results)
    if decoded_images is not None:
        decoded["image"] = decoded_images

    conditions: dict = {}
    if cfg.populate_conditions:
        text_cond, neg_text_cond = _build_text_conditions(results)
        if text_cond is not None:
            conditions["text"] = text_cond
        if neg_text_cond is not None:
            conditions["negative_text"] = neg_text_cond

    return RolloutResp(
        sample_ids=list(req.sample_ids),
        group_ids=list(req.group_ids),
        conditions=conditions,
        rollout_traces=rollout_traces,
        decoded=decoded,
    )


__all__ = ["_to_rollout_resp"]
