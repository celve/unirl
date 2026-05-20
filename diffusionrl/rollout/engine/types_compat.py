"""Translation between legacy ``RolloutRequest``/``RolloutSamples`` and the
new ``RolloutReq``/``RolloutResp`` containers.

The actor (``diffusionrl.ray.rollout_actor.RolloutActor``) and the rollout
pipeline mixin still speak the legacy types. New-ABC engines speak only the
new types. These helpers bridge the gap so the actor can be migrated
gradually: the actor calls ``request_to_req`` before delegating to a new
engine and ``resp_to_samples`` after.

Design notes:

- Modality-agnostic dispatch. ``resp_to_samples`` finds the first
  ``LatentSegment`` in ``resp.rollout_traces`` regardless of slot key
  (``"image"`` for SD3 / HI3 t2i, ``"video"`` for WAN T2V, future
  modalities), and iterates ``resp.decoded.items()`` to dispatch
  decoded primitives by Python type — ``Images`` to ``decoded_images``,
  ``Videos`` to ``decoded_videos``. AR-only modalities (t2t / i2t) that
  produce no latents/log-probs still raise — the legacy
  ``RolloutSamples`` shape can't represent those payloads. AR modalities
  will be wired straight through the new types once the pipeline mixin
  migrates.
- ``forward_context`` is left ``None`` because vllm-omni's worker
  subprocess does not ship the fused MM condition back (see
  ``diffusionrl/rollout/engine/vllm_omni/response.py`` module docstring,
  "fused MM tensors live in the worker subprocess and aren't shipped
  back"). Replay/training-side flows that require a forward context have
  to recompute it from the surfaced prompt + AR tokens.
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch

from diffusionrl.types.primitives import Images, Texts, Videos
from diffusionrl.types.request import RolloutRequest
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutResp
from diffusionrl.types.sample import LogProbData, RolloutSamples
from diffusionrl.types.segments.latent import LatentSegment
from diffusionrl.types.trajectory_store import Trajectory


def request_to_req(request: RolloutRequest) -> RolloutReq:
    """Translate a legacy ``RolloutRequest`` into the new ``RolloutReq``.

    Carries the prompt strings as a ``Texts`` primitive and packs the
    diffusion sampling knobs into ``stage_params["diffusion"]`` so they
    line up with what ``_to_omni_per_stage`` expects (see
    ``diffusionrl/rollout/engine/vllm_omni/request.py``).
    """
    if request.prompts is None or not request.prompts.prompts:
        raise ValueError("request_to_req: RolloutRequest.prompts must be non-empty.")
    sp = request.sampling_params
    diffusion = {
        "height": int(sp.height),
        "width": int(sp.width),
        "num_inference_steps": int(sp.num_inference_steps),
        "guidance_scale": float(sp.guidance_scale),
        "eta": float(sp.sde_config.eta),
        "seed": int(sp.seed),
    }
    return RolloutReq(
        sample_ids=list(request.prompts.sample_ids),
        group_ids=list(request.prompts.group_ids),
        primitives={"text": Texts(texts=list(request.prompts.prompts))},
        stage_params={"diffusion": diffusion},
    )


def resp_to_samples(
    resp: RolloutResp,
    *,
    request: RolloutRequest,
) -> RolloutSamples:
    """Translate a ``RolloutResp`` (any visual modality) into ``RolloutSamples``.

    Iterates ``resp.rollout_traces.items()`` to find the first
    :class:`LatentSegment` (regardless of slot key — ``"image"`` /
    ``"video"`` / future modalities) and ``resp.decoded.items()`` to
    dispatch decoded primitives into ``decoded_images`` /
    ``decoded_videos`` by isinstance check. Image-only, video-only, and
    mixed-modality payloads are all valid.

    Raises if no rollout_traces / no LatentSegment is present — AR-only
    modalities (t2t / i2t) can't be expressed in the legacy
    RolloutSamples shape (no latents / no log-probs); callers for those
    must migrate off RolloutSamples.

    ``forward_context`` is left ``None`` — callers that require it
    (training-side replay) recompute it from the surfaced prompt + AR
    tokens.
    """
    # Find the first LatentSegment-bearing trace. Most modalities use
    # exactly one slot ("image" for SD3 / HI3 t2i, "video" for WAN T2V);
    # the slot key is informational here — what matters is the segment
    # carries latents + sigmas + log_probs.
    seg = None
    seg_key: Optional[str] = None
    for k, candidate in resp.rollout_traces.items():
        if isinstance(candidate, LatentSegment):
            seg = candidate
            seg_key = k
            break
    if seg is None:
        raise ValueError(
            f"resp_to_samples: RolloutResp has no LatentSegment trace; "
            f"have rollout_traces keys={list(resp.rollout_traces.keys())}. "
            f"AR-only modalities cannot be expressed in the legacy "
            f"RolloutSamples shape (no latents / no log-probs). Migrate "
            f"the caller off RolloutSamples for those modalities."
        )
    if seg.latents is None:
        raise ValueError(
            f"resp_to_samples: {seg_key!r} segment is missing latents — "
            f"vllm-omni must capture trajectory_latents "
            f"(return_trajectory_latents=True)."
        )

    final_latents = seg.latents[:, -1]
    # Sparse SDE (FlowGRPO-Fast et al.): when the segment stores only a
    # subset of step positions, build a *selective* Trajectory so the
    # full σ schedule survives downstream — legacy GRPO loss reads
    # ``td.sigmas[1]`` for ``sigma_max`` (see
    # ``diffusionrl/algorithms/grpo.py:332``) and that index has to map
    # to the GLOBAL step-1 σ, not to "position-1 in the sparse stored
    # set" (which would resolve to, say, sigma[5] for sde_indices=[0,5]).
    # ``Trajectory.from_full`` would force ``TrainingBatch.validate`` to
    # require ``len(timesteps) == num_stored`` and we'd have to crop —
    # ``from_selective`` carries ``total_positions`` separately and
    # validates via the selective branch instead.
    if seg.indices is not None and seg.sigmas is not None and int(seg.indices.shape[0]) != int(seg.sigmas.shape[0]):
        trajectory_store = Trajectory.from_selective(
            seg.latents,
            collected_positions=[int(p) for p in seg.indices.tolist()],
            total_positions=int(seg.sigmas.shape[0]),
        )
    else:
        trajectory_store = Trajectory.from_full(seg.latents)

    log_probs: Optional[LogProbData] = None
    if seg.sde_logp is not None and seg.sde_indices is not None:
        log_probs = LogProbData.from_dict(
            {int(seg.sde_indices[i].item()): seg.sde_logp[:, i] for i in range(seg.sde_logp.shape[1])}
        )

    # Dispatch every decoded primitive by its Python type, NOT by the
    # slot key — keeps this bridge open to "rollout pipeline names its
    # video slot 'frames' instead of 'video'" cases without changing
    # this translator.
    decoded_images: Optional[List[Any]] = None
    decoded_videos: Optional[List[Any]] = None
    for _key, decoded in resp.decoded.items():
        if decoded is None:
            continue
        if isinstance(decoded, Images):
            # ``Images.pixels`` is [B, C, H, W]; legacy RolloutSamples expects
            # a list of 3D [C, H, W] tensors (one per sample) — see
            # ``RolloutSamples.decoded_images`` docstring.
            pixels = getattr(decoded, "pixels", None)
            if pixels is not None:
                decoded_images = list(pixels.unbind(0))
        elif isinstance(decoded, Videos):
            # ``Videos`` is varlen-packed; ``to_list()`` returns per-sample
            # ``Video`` objects whose ``frames`` are ``[T_i, C, H, W]``.
            # Legacy RolloutSamples.decoded_videos is ``Optional[Any]`` so
            # a list of 4D ``[C, T, H, W]`` tensors fits — permute and
            # store per-sample so the reward pipeline can iterate without
            # the framework's PACKED protocol knowledge.
            decoded_videos = [v.frames.permute(1, 0, 2, 3).contiguous() for v in decoded.to_list()]
        # else: unknown primitive type — silently skip. When a new
        # modality lands (Audios, PointClouds, ...), wire it here.

    # Pass the FULL σ schedule through, regardless of dense vs sparse —
    # legacy callers index it by global step id (e.g. ``td.sigmas[1]``
    # for sigma_max), so cropping to a sparse local view would silently
    # mis-map. The selective-Trajectory path above keeps the validation
    # invariant ``len(timesteps) == num_stored`` from firing on the
    # full-store branch (``TrainingBatch.validate`` line 256 only
    # enforces it under ``is_full``).
    #
    # ``step_indices``: only emit it for FULL trajectories (where the
    # "compact-index == global-index" alignment lets
    # ``TrainingBatch.get_position_for_step`` return the right compact
    # slot). For SELECTIVE storage that mapping breaks
    # (``trajectory_store.has_position`` checks GLOBAL position while the
    # legacy method passes the COMPACT-array index), so emit ``None``
    # and let any caller that tries per-step lookups raise loudly. The
    # only live consumer of this bridge today is the reward pipeline,
    # which does not call ``get_position_for_step``; sparse-SDE per-step
    # lookups live on the new path (``stage.replay`` reads
    # ``segment.sigmas`` directly, bypasses this bridge entirely).
    if seg.sigmas is not None:
        timesteps = seg.sigmas
        if trajectory_store.is_selective:
            step_indices = None
        else:
            step_indices = seg.indices.to(dtype=torch.long) if seg.indices is not None else None
    else:
        timesteps = torch.zeros(0, dtype=torch.float32)
        step_indices = None

    return RolloutSamples(
        latents=final_latents,
        timesteps=timesteps,
        sampling_params=request.sampling_params,
        prompts=request.prompts,
        trajectories=trajectory_store,
        log_probs=log_probs,
        forward_context=None,
        step_indices=step_indices,
        rewards=None,
        advantages=None,
        component_rewards=None,
        decoded_images=decoded_images,
        decoded_videos=decoded_videos,
        media_preview=None,
        reward_compute_s=0.0,
    )


__all__ = ["request_to_req", "resp_to_samples"]
