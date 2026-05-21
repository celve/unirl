"""``Omni.generate`` outputs → ``RolloutResp`` translator.

Single modality-branched function ``_to_rollout_resp(req, per_request_outputs,
*, modality)``. Caller groups ``Omni.generate``'s flat output list into
per-request lists; this function picks the per-stage outputs (Stage 0 AR,
Stage 1 DiT for image modalities) and packs into ``RolloutResp``.

Produces:

- ``resp.decoded["image"]`` (image modalities) — pixels ``[B, C, H, W]``
  in [0, 1] from PIL outputs of the DiT stage.
- ``resp.decoded["text"]`` (AR-only modalities) — ``Texts`` from
  ``request_output.outputs[0].text``.
- ``resp.rollout_traces["image"]`` (image modalities) — ``LatentSegment`` from
  the DiT stage's trajectory tensors.
- ``resp.rollout_traces["ar"]`` (all modalities) — ``TextSegment`` packed by
  ``hi3.ar_capture.extract_ar_segment``.
- ``resp.conditions["fused"]`` (image modalities) —
  ``HunyuanImage3FusedMultimodalCondition`` built from per-request
  ``OmniRequestOutput.custom_output["fused_mm_capture"]`` (written by
  :class:`RLHunyuanImage3Pipeline` on the first per-request
  ``prepare_inputs_for_generation`` call inside the worker; vllm-omni
  routes ``DiffusionOutput.custom_output`` into ``OmniRequestOutput.custom_output``
  on the IPC boundary, see upstream ``diffusion/data.py:841`` and
  ``stage_diffusion_proc.py:182``). t2i scope for the first cut —
  surfaces ``input_ids`` / ``attention_mask`` / ``position_ids`` /
  ``rope_cache`` / ``gen_image_mask`` / ``gen_timestep_scatter_index``;
  the it2i ``cond_*`` fields stay unpopulated. Missing capture is a
  fatal misconfiguration (pipeline subclass not installed, hook
  regression, ...) — ``_to_rollout_resp`` raises at the rollout boundary
  rather than silently emitting empty conditions that would crash the
  trainer-side replay much later.
- ``resp.conditions = {}`` (AR-only modalities) — no diffusion replay
  in scope.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from diffusionrl.models.hunyuan_image3.conditions import (
    HunyuanImage3FusedMultimodalCondition,
)
from diffusionrl.rollout.engine.sigma_verify import verify_engine_used_sigmas
from diffusionrl.rollout.engine.vllm_omni.hi3.ar_capture import extract_ar_segment
from diffusionrl.types.conditions import Condition
from diffusionrl.types.conditions.text import TextEmbedCondition
from diffusionrl.types.primitives import Image, Images, Text, Texts
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutResp
from diffusionrl.types.segments.latent import make_image_segment


def group_by_request(
    flat_outputs: Sequence[Any],
    n: int,
) -> List[List[Any]]:
    """Group ``Omni.generate``'s flat output list into per-request lists.

    ``Omni._run_generation`` builds ``request_ids = [f"{i}_{uuid4()}"
    for i in range(B)]`` (one per prompt). After our YAML flips Stage 0
    to ``final_output: true`` for it2i, each request contributes one
    output per final stage (2 for t2i/it2i, 1 for i2t/t2t).

    The mapping from output back to request index is by the ``i_`` prefix
    on ``request_id``. When the orchestrator's ordering invariant
    changes upstream, the count won't match the expected total and we
    raise — better than silently misaligning.
    """
    grouped: List[List[Any]] = [[] for _ in range(n)]
    for out in flat_outputs:
        rid = getattr(out, "request_id", "") or ""
        if "_" in rid:
            idx_part = rid.split("_", 1)[0]
            try:
                idx = int(idx_part)
            except ValueError:
                continue
            if 0 <= idx < n:
                grouped[idx].append(out)
    return grouped


def _pil_list_to_images(pil_images: Sequence[Any]) -> Images:
    """``[PIL.Image, …] → Images`` (float32 ``[B, C, H, W]`` in [0, 1])."""
    if not pil_images:
        raise ValueError("_pil_list_to_images: empty image list")
    from torchvision.transforms.functional import pil_to_tensor

    items: List[Image] = []
    for pil in pil_images:
        # uint8 [C, H, W] / 255 → float32 [0, 1] per Image(pixels=...) contract.
        t = pil_to_tensor(pil).to(torch.float32) / 255.0
        items.append(Image(pixels=t))
    return Images.from_list(items)


def _pick_stage_output(
    outputs: Sequence[Any],
    *,
    final_output_type: str,
    stage_id: Optional[int] = None,
) -> Optional[Any]:
    """Find the ``OmniRequestOutput`` with the requested ``final_output_type``.

    Falls back to ``stage_id`` match if provided. Returns ``None`` if
    neither matches — callers decide whether that's an error.
    """
    for out in outputs:
        if getattr(out, "final_output_type", None) == final_output_type:
            return out
    if stage_id is not None:
        for out in outputs:
            if getattr(out, "stage_id", None) == stage_id:
                return out
    return None


def _build_image_segment(
    diff_outputs: Sequence[Any],
    *,
    expected_sigmas: Optional[torch.Tensor] = None,
) -> Any:
    """Build ``LatentSegment`` for the DiT stage's outputs.

    Each per-prompt ``OmniRequestOutput`` carries its own
    ``trajectory_latents`` / ``trajectory_log_probs`` for its own request.
    With ``runtime.max_inflight=1`` the diffusion engine processes one
    request at a time, so the per-prompt tensors are NOT shared refs to
    a full-batch tensor — each is shape ``[1, T+1, ...]`` / ``[1, K]``
    where ``K`` is the number of SDE-gated steps (``K`` ranges from
    ``0`` for forward-process / NFT runs up to ``T`` for fully-SDE runs).
    We concatenate across all outputs to recover ``[B, T+1, ...]`` /
    ``[B, K]`` where ``B = len(diff_outputs)``.

    ``sigmas`` / ``indices`` / ``sde_indices`` are sample-shared (the SDE
    schedule and stored-slot indexing are identical across all samples in
    the chunk), so we read them off the first output without concat.

    Output shapes:
    - ``latents`` from ``trajectory_latents`` — ``[B, T+1, ...]`` after
      concat across diff_outputs. ALWAYS dense (every step recorded so
      replay has ``x_t`` at every slot regardless of which steps ran SDE).
    - ``sigmas`` from ``trajectory_timesteps`` — the field name reads
      "timesteps" but our ``RL*Pipeline.forward`` overwrites its contents
      with the true [0, 1] sigma schedule (1D ``[T+1]``) drained from
      ``FlowMatchSDEDiscreteScheduler``. Sample-shared.
    - ``sde_logp`` from ``trajectory_log_probs`` — ``[B, K]`` after concat
      (K = number of SDE-gated steps; can be < T when the algorithm picks
      a sparse subset via ``stage_params["diffusion"]["sde_indices"]``).
    - ``indices`` — dense ``arange(T+1)``: latent-storage slots.
    - ``sde_indices`` — sparse step IDs ``[K]`` read off the worker's
      ``custom_output["sde_step_indices"]`` (echoed there by the pipeline
      subclass; falls back to ``arange(K)`` only if the capture is
      missing, e.g. older pipeline build).
    - ``sample_indices`` — ``arange(B)``.
    """
    if not diff_outputs:
        raise ValueError("_build_image_segment: empty diff_outputs")

    per_latents: List[torch.Tensor] = []
    per_log_probs: List[torch.Tensor] = []
    for diff_out in diff_outputs:
        traj_l = getattr(diff_out, "trajectory_latents", None)
        if traj_l is not None:
            per_latents.append(traj_l)
        traj_lp = getattr(diff_out, "trajectory_log_probs", None)
        if traj_lp is not None:
            per_log_probs.append(traj_lp)

    traj_latents: Optional[torch.Tensor] = torch.cat(per_latents, dim=0) if per_latents else None
    traj_log_probs: Optional[torch.Tensor] = torch.cat(per_log_probs, dim=0) if per_log_probs else None
    # Sigmas / step-index axes are sample-shared — the SDE schedule and
    # stored-slot count don't vary per sample in a single chunk.
    head = diff_outputs[0]
    seg_sigmas = getattr(head, "trajectory_timesteps", None)
    # Engine→worker→response σ contract: the engine pinned ``req.sigmas``
    # before dispatch, the worker should have consumed it via
    # ``set_timesteps(sigmas=...)`` and echoed the same values back via
    # ``trajectory_timesteps``. Assert equality here so a broken wire
    # surfaces immediately rather than silently de-syncing training-side
    # replay from rollout. Caller passes ``expected_sigmas=None`` to skip
    # the check (legacy entry points that don't run ensure_req_sigmas).
    verify_engine_used_sigmas(
        seg_sigmas,
        expected=expected_sigmas,
        engine_name="vllm-omni",
    )
    head_custom = getattr(head, "custom_output", None) or {}
    sde_step_indices_raw = head_custom.get("sde_step_indices")

    indices: Optional[torch.Tensor] = None
    sde_indices: Optional[torch.Tensor] = None
    # K == 0 happens when the algorithm requested zero SDE steps (NFT /
    # forward-process). Treat the empty case identically to "no log_probs
    # at all": clean-latents segment with no sde_logp / sde_indices.
    # Trainer-side `to_training_batch` already branches on
    # ``segment.sde_indices is None`` to take the clean-latents path.
    K = int(traj_log_probs.shape[1]) if traj_log_probs is not None else 0
    if K > 0:
        # ``trajectory_log_probs`` is ``[B, K]`` (one entry per SDE-gated
        # transition; ``K`` equals ``T`` only when every step was SDE).
        # ``trajectory_latents`` is ``[B, T+1, ...]`` (position-0 + T post-step,
        # ALWAYS dense — scheduler captures latent regardless of SDE/ODE).
        # ``indices`` maps step_idx -> storage slot for
        # ``LatentSegment.latents_at``, so it must enumerate every stored
        # slot (0..T). ``sde_indices`` enumerates the SDE-gated step ids
        # only (length K).
        T_plus_1 = int(traj_latents.shape[1]) if traj_latents is not None else K + 1
        indices = torch.arange(T_plus_1, dtype=torch.long)
        if sde_step_indices_raw is not None:
            sde_indices = torch.as_tensor([int(i) for i in sde_step_indices_raw], dtype=torch.long)
            if int(sde_indices.numel()) != K:
                raise RuntimeError(
                    f"_build_image_segment: scheduler reported "
                    f"sde_step_indices of length {int(sde_indices.numel())} "
                    f"but trajectory_log_probs has {K} entries — pipeline "
                    f"subclass produced inconsistent outputs."
                )
        else:
            # Legacy fallback when the pipeline subclass didn't echo the
            # real step IDs (e.g. older HI3 path being upgraded). Only safe
            # when K == T (dense case). For sparse K < T this misaligns
            # replay; raise rather than silently mis-label.
            T = int(traj_latents.shape[1]) - 1 if traj_latents is not None else K
            if K != T:
                raise RuntimeError(
                    "_build_image_segment: trajectory log_probs has K="
                    f"{K} but latents has T={T} steps and worker did not "
                    "expose ``custom_output['sde_step_indices']``. Update "
                    "the pipeline subclass to echo last_sde_step_indices."
                )
            sde_indices = torch.arange(K, dtype=torch.long)
    elif traj_latents is not None:
        # Forward-process case (NFT): still emit ``indices`` so the
        # clean-latents branch on the trainer side can look up the final
        # latent, but leave ``sde_indices`` / ``sde_logp`` as None (drop
        # the ``[B, 0]`` placeholder — it confuses downstream stage
        # ``replay`` paths that read ``sde_logp.shape[1]``).
        traj_log_probs = None
        T_plus_1 = int(traj_latents.shape[1])
        indices = torch.arange(T_plus_1, dtype=torch.long)
        sde_indices = None

    batch_size = 0
    if traj_latents is not None:
        batch_size = int(traj_latents.shape[0])
    elif traj_log_probs is not None:
        batch_size = int(traj_log_probs.shape[0])
    sample_indices = torch.arange(batch_size, dtype=torch.long) if batch_size > 0 else None

    return make_image_segment(
        latents=traj_latents,
        sigmas=seg_sigmas,
        indices=indices,
        sde_logp=traj_log_probs,
        sde_indices=sde_indices,
        sample_indices=sample_indices,
    )


def _decoded_text_from_ar(per_request_outputs: Sequence[Sequence[Any]]) -> Texts:
    """Extract the per-request AR text from Stage 0 outputs."""
    texts: List[Text] = []
    for outputs in per_request_outputs:
        ar = _pick_stage_output(outputs, final_output_type="text", stage_id=0)
        text_str = ""
        if ar is not None:
            ro = getattr(ar, "request_output", None)
            if ro is not None:
                completions = getattr(ro, "outputs", None) or []
                if completions:
                    text_str = getattr(completions[0], "text", "") or ""
        texts.append(Text(text=text_str))
    return Texts.from_list(texts)


def _build_fused_mm_condition(
    diff_outputs: Sequence[Any],
) -> Optional[HunyuanImage3FusedMultimodalCondition]:
    """Concat per-request ``fused_mm_capture`` dicts into one fused condition.

    Reads the capture off ``OmniRequestOutput.custom_output["fused_mm_capture"]``
    — the dataclass-routed dict ``RLHunyuanImage3Pipeline`` writes after
    intercepting ``prepare_inputs_for_generation``. Plain runtime attrs on
    ``DiffusionOutput`` don't survive vllm-omni's IPC boundary.

    Returns ``None`` when any diff output is missing the capture (e.g. the
    worker side hasn't installed :class:`RLHunyuanImage3Pipeline`'s hook,
    or upstream's ``prepare_inputs_for_generation`` was bypassed). Callers
    treat ``None`` as "no conditions surfaced" and emit ``resp.conditions = {}``,
    preserving the pre-patch contract.

    For think_recaption mode, different prompts produce different AR output
    lengths → different ``L`` per capture. This function right-pads shorter
    sequences to ``max_L`` (pad_token_id=0 for input_ids, False for masks,
    0.0 for rope_cache) so ``torch.cat`` on dim 0 works.
    """
    if not diff_outputs:
        return None
    captures = [(getattr(d, "custom_output", None) or {}).get("fused_mm_capture") for d in diff_outputs]
    if any(c is None for c in captures):
        return None

    sequence_lengths = [int(c["input_ids"].shape[-1]) for c in captures]
    max_L = max(sequence_lengths)

    def _pad_to(t: Any, target_L: int, dim: int = -1, value: Any = 0) -> Any:
        if t is None or not isinstance(t, torch.Tensor):
            return t
        cur_L = t.shape[dim]
        if cur_L >= target_L:
            return t
        pad_size = target_L - cur_L
        ndim = t.ndim
        pad_spec = [0] * (2 * ndim)
        actual_dim = dim if dim >= 0 else ndim + dim
        pad_idx = (ndim - 1 - actual_dim) * 2
        pad_spec[pad_idx + 1] = pad_size
        return torch.nn.functional.pad(t, pad_spec, value=value)

    def _pad_attn_mask(mask: Any, target_L: int) -> Any:
        """Pad attention_mask [N, 1, L, L] → [N, 1, target_L, target_L]."""
        if mask is None or not isinstance(mask, torch.Tensor):
            return mask
        if mask.shape[-1] >= target_L:
            return mask
        N, H, L, _ = mask.shape
        padded = torch.zeros(N, H, target_L, target_L, dtype=mask.dtype, device=mask.device)
        padded[:, :, :L, :L] = mask
        return padded

    padded_captures = []
    for c, L_i in zip(captures, sequence_lengths):
        if L_i == max_L:
            padded_captures.append(c)
        else:
            padded_captures.append(
                {
                    "input_ids": _pad_to(c["input_ids"], max_L, dim=-1, value=0),
                    "attention_mask": _pad_attn_mask(c.get("attention_mask"), max_L),
                    "position_ids": _pad_to(c.get("position_ids"), max_L, dim=-1, value=0),
                    "gen_image_mask": _pad_to(c.get("gen_image_mask"), max_L, dim=-1, value=False),
                    "gen_timestep_scatter_index": c.get("gen_timestep_scatter_index"),
                    "rope_cache": (
                        (
                            _pad_to(c["rope_cache"][0], max_L, dim=-2, value=0.0),
                            _pad_to(c["rope_cache"][1], max_L, dim=-2, value=0.0),
                        )
                        if c.get("rope_cache") is not None and isinstance(c["rope_cache"], tuple)
                        else c.get("rope_cache")
                    ),
                }
            )

    fused_dict: Dict[str, Any] = {
        "input_ids": torch.cat([c["input_ids"] for c in padded_captures], dim=0),
        "attention_mask": torch.cat([c["attention_mask"] for c in padded_captures], dim=0),
        "position_ids": torch.cat([c["position_ids"] for c in padded_captures], dim=0),
        "gen_image_mask": torch.cat([c["gen_image_mask"] for c in padded_captures], dim=0),
        "gen_timestep_scatter_index": torch.cat([c["gen_timestep_scatter_index"] for c in padded_captures], dim=0),
    }
    cos_parts = [c["rope_cache"][0] for c in padded_captures]
    sin_parts = [c["rope_cache"][1] for c in padded_captures]
    fused_dict["rope_cache"] = (
        torch.cat(cos_parts, dim=0),
        torch.cat(sin_parts, dim=0),
    )

    # ``from_dict`` skips optional fields when absent; cond_* fields stay
    # ``None`` for t2i (out of scope for the it2i extension).
    return HunyuanImage3FusedMultimodalCondition.from_dict(fused_dict)


def _build_sd3_text_condition(
    diff_outputs: Sequence[Any],
) -> Optional[TextEmbedCondition]:
    """Concat per-request ``text_capture`` dicts into one TextEmbedCondition.

    Reads ``OmniRequestOutput.custom_output["text_capture"]`` — the
    dataclass-routed dict :class:`RLStableDiffusion3Pipeline` writes
    after intercepting ``encode_prompt``. Plain runtime attrs on
    ``DiffusionOutput`` don't survive vllm-omni's IPC boundary.

    Returns ``None`` when any diff output is missing the capture (e.g.
    the worker side hasn't installed
    :class:`RLStableDiffusion3Pipeline`'s hook). The training side
    requires this condition for :meth:`SD3DiffusionStage.replay`; an
    empty conditions dict will surface as ``SD3Conditions.from_dict``
    failure downstream (clear error rather than silent skip).

    For SD3, all per-request encodes share the same ``L`` (T5 padding
    to ``max_sequence_length=256`` is fixed), so a plain concat on dim 0
    suffices.
    """
    if not diff_outputs:
        return None
    captures = [(getattr(d, "custom_output", None) or {}).get("text_capture") for d in diff_outputs]
    if any(c is None for c in captures):
        return None

    return TextEmbedCondition(
        embeds=torch.cat([c["prompt_embeds"] for c in captures], dim=0),
        pooled=torch.cat([c["pooled_prompt_embeds"] for c in captures], dim=0),
        attn_mask=None,  # SD3 uses fixed-length T5 padding; no attn mask needed
    )


def _to_rollout_resp(
    req: RolloutReq,
    per_request_outputs: Sequence[Sequence[Any]],
    *,
    modality: str,
) -> RolloutResp:
    """``Omni.generate`` per-request outputs → ``RolloutResp``."""
    if not per_request_outputs or not any(per_request_outputs):
        raise ValueError("_to_rollout_resp: empty per_request_outputs (Omni.generate returned nothing surfaceable).")

    decoded: dict = {}
    rollout_traces: dict = {}
    conditions: Dict[str, Condition] = {}

    if modality in ("t2i", "it2i", "sd35_t2i", "t2i_think_recaption"):
        # Per-request DiT (image) output. For HI3 (t2i/it2i) it's Stage 1;
        # for SD3.5 (sd35_t2i) the diffusion stage is the only stage so
        # stage_id=0. Either way, ``_pick_stage_output`` matches by
        # ``final_output_type='image'`` first and falls back to stage_id.
        dit_stage_id = 0 if modality == "sd35_t2i" else 1
        diff_outputs: List[Any] = []
        pil_images: List[Any] = []
        for outputs in per_request_outputs:
            diff_out = _pick_stage_output(
                outputs,
                final_output_type="image",
                stage_id=dit_stage_id,
            )
            if diff_out is None:
                raise RuntimeError(
                    f"_to_rollout_resp: no image output for request (modality={modality}); did the DiT stage fail?"
                )
            diff_outputs.append(diff_out)
            imgs = getattr(diff_out, "images", None) or []
            pil_images.extend(imgs)

        if not pil_images:
            raise RuntimeError(
                "_to_rollout_resp: DiT outputs carry no PIL images; "
                "check pipeline forward populated DiffusionOutput.output."
            )
        decoded["image"] = _pil_list_to_images(pil_images)
        rollout_traces["image"] = _build_image_segment(
            diff_outputs,
            expected_sigmas=req.sigmas,
        )

        # Surface conditions captured worker-side. Modality-specific:
        # HI3 (t2i/it2i) captures fused MM tensors via
        # ``prepare_inputs_for_generation``; SD3 (sd35_t2i) captures text
        # embeds via ``encode_prompt``. Both are *required* for the training
        # side's ``replay`` step — silently returning empty conditions makes
        # the trainer crash with ``KeyError``/``from_dict({})`` errors far
        # from the root cause. Fail fast at the rollout→trainer boundary
        # instead so the pipeline hook regression is visible immediately.
        if modality == "sd35_t2i":
            text_cond = _build_sd3_text_condition(diff_outputs)
            if text_cond is None:
                raise RuntimeError(
                    "_to_rollout_resp: SD3 rollout returned no 'text_capture' on "
                    "DiffusionOutput.custom_output. Check that "
                    "RLStableDiffusion3Pipeline._install_encode_prompt_hook ran "
                    "in every DiT worker — the subclass swap may not have taken "
                    "effect (verify custom_pipeline_args.pipeline_class in the "
                    "stage YAML)."
                )
            conditions["text"] = text_cond
        else:
            fused_cond = _build_fused_mm_condition(diff_outputs)
            if fused_cond is None:
                raise RuntimeError(
                    f"_to_rollout_resp: HI3 rollout (modality={modality!r}) "
                    "returned no 'fused_mm_capture' on DiffusionOutput.custom_output. "
                    "Check that RLHunyuanImage3Pipeline.prepare_inputs_for_generation "
                    "hook ran in every DiT worker — the subclass swap may not have "
                    "taken effect (verify custom_pipeline_args.pipeline_class in "
                    "the stage YAML)."
                )
            conditions["fused"] = fused_cond
    elif modality in ("i2t", "t2t"):
        decoded["text"] = _decoded_text_from_ar(per_request_outputs)
    else:
        raise ValueError(f"_to_rollout_resp: unknown modality {modality!r}")

    # Surface AR-generated text for all modalities that run AR (Stage 0).
    # For t2i_think_recaption this is the CoT + recaption text.
    if modality in ("t2i", "it2i", "t2i_think_recaption") and "text" not in decoded:
        try:
            decoded["text"] = _decoded_text_from_ar(per_request_outputs)
        except Exception:
            pass  # best-effort; don't break rollout if AR text extraction fails

    # AR segment is shared by all modalities (Stage 0 always runs).
    ar_segment = extract_ar_segment(per_request_outputs)
    if ar_segment is not None:
        rollout_traces["ar"] = ar_segment

    return RolloutResp(
        sample_ids=list(req.sample_ids),
        group_ids=list(req.group_ids),
        conditions=conditions,
        rollout_traces=rollout_traces,
        decoded=decoded,
        rewards=None,
        advantages=None,
        status=None,
    )


__all__ = ["_to_rollout_resp", "group_by_request"]
