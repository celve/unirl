"""RL-aware HunyuanImage3 pipeline subclass.

Three behaviors on top of upstream ``HunyuanImage3Pipeline``
(``vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py:300``):

1. Before ``super().forward(req)``: unconditionally install our
   :class:`FlowMatchSDEDiscreteScheduler` on the inner pipeline (replacing
   the upstream ``FlowMatchEulerDiscreteScheduler``), regardless of eta.
   Our scheduler captures the dense latent trajectory; per-step SDE/ODE
   gating is driven by ``req.sampling_params.extra_args["sde_indices"]``.
   When no step is SDE-gated, the scheduler degenerates to pure Euler ODE
   — so installing at eta=0 has no behavioural cost while still keeping
   ``segment.latents`` populated for ``resp_to_samples``.
2. After ``super().forward(req)`` returns: drain the captured
   trajectory off the scheduler and stamp into
   ``DiffusionOutput.trajectory_{latents,timesteps,log_probs}``.
   ``trajectory_timesteps`` is overwritten to carry the **true [0, 1]
   sigma schedule** (1D ``[T+1]``) — that's what replay
   (``HunyuanImage3DiffusionStage.replay``) reads as
   ``segment.sigmas[step_idx]``. The original 1000-scale per-step
   timesteps are dropped: they're trivially regenerable from the
   sigma schedule when needed for diagnostics, and the
   ``DiffusionOutput`` dataclass appears to filter
   runtime-attached attributes during the worker→parent IPC
   (so a separate ``trajectory_sigmas`` attr did not survive).
3. While the per-step kernel runs, wrap the inner pipeline's
   ``model.prepare_inputs_for_generation`` to capture the fused
   multimodal tensors (``input_ids`` / ``attention_mask`` /
   ``position_ids`` / ``rope_cache`` / ``gen_image_mask`` /
   ``gen_timestep_scatter_index``) on the **first** per-request call —
   subsequent steps under KV-cache reuse pass the gathered-down ``L'``
   slice which is not what training-side replay needs. The captured
   dict is detached/cpu'd and written to
   ``DiffusionOutput.custom_output["fused_mm_capture"]``, the
   dataclass-declared dict vllm-omni explicitly forwards into
   ``OmniRequestOutput.custom_output`` (upstream ``diffusion/data.py:841``,
   ``stage_diffusion_proc.py:182``). Plain ``setattr`` on
   ``DiffusionOutput`` would not survive IPC. Parent-side
   ``_to_rollout_resp`` reads it back and surfaces it as
   ``RolloutResp.conditions["fused"]`` so ``DiffusionGRPO`` →
   ``HunyuanImage3DiffusionConditions.from_dict`` can consume it without
   a separate train-side rebuild.

Everything else — system-prompt resolution, AR-bridged ``ar_generated_text``,
it2i conditioning via ``batch_cond_image_info``, generator/seed/CFG, the
denoise loop itself — is handled by upstream's ``forward`` at
``pipeline_hunyuan_image3.py:1262-1347``.

This class is loaded inside vLLM-Omni's worker subprocess via
``custom_pipeline_args.pipeline_class`` injected from our static stage
configs (``stage_configs/hunyuan_image3_t2i_rl.yaml`` and ``..._it2i_rl.yaml``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
    HunyuanImage3Pipeline,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest

from diffusionrl.rollout.engine.vllm_omni.hi3.sde_scheduler import (
    FlowMatchSDEDiscreteScheduler,
)


def _detach_cpu(t: Any) -> Any:
    """Detach + move to CPU for IPC. Pass through ``None`` / non-tensors."""
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        return t.detach().cpu()
    return t


def _detach_cpu_pair(p: Any) -> Any:
    """``(cos, sin)`` rope-cache pair handler. Pass-through for ``None``."""
    if p is None:
        return None
    if isinstance(p, tuple) and len(p) == 2:
        return (_detach_cpu(p[0]), _detach_cpu(p[1]))
    return p


class RLHunyuanImage3Pipeline(HunyuanImage3Pipeline):
    """HunyuanImage3 pipeline with SDE trajectory capture for RL rollout."""

    def __init__(self, od_config: OmniDiffusionConfig) -> None:
        super().__init__(od_config)
        # Stash the upstream scheduler the first time ``_ensure_scheduler_for_eta``
        # runs (it has to materialize ``self.pipeline`` first), so we can
        # build our subclass via the same shift parameters. Our scheduler
        # then runs for every request — regardless of eta — so
        # ``segment.latents`` is always populated for the clean-latents path.
        self._upstream_scheduler = None
        # Fused-MM capture state. ``_fused_capture`` is reset to ``None`` at
        # the top of every ``forward`` and filled by the first per-request
        # ``prepare_inputs_for_generation`` call; ``_prepare_inputs_patched``
        # is the idempotent install flag so the wrap happens exactly once
        # per pipeline instance.
        self._fused_capture: Optional[Dict[str, Any]] = None
        self._prepare_inputs_patched: bool = False

    def _ensure_scheduler_for_eta(self, eta: float) -> None:
        """Install our trajectory-capturing scheduler regardless of ``eta``.

        First call materializes ``self._pipeline`` via upstream's
        ``pipeline`` property (``pipeline_hunyuan_image3.py:429-443``),
        which sets ``self.scheduler`` to the upstream Euler scheduler —
        we stash that for any future restore and swap in our own.

        Swap goes through the inner pipeline's ``set_scheduler`` hook at
        ``hunyuan_image3_transformer.py:2547-2548`` (which calls
        ``register_modules`` so the diffusers component graph stays
        consistent).

        Why always install (even at ``eta == 0``): same reason as SD3 —
        ``resp_to_samples`` needs ``segment.latents`` to be non-empty and
        only this scheduler captures the dense per-step trajectory.
        ``eta == 0`` collapses the SDE branch in ``step`` to pure Euler
        (gated on ``_sde_indices_set``), so NFT-style ``no SDE`` requests
        still get the trajectory they need.
        """
        # Force the inner pipeline + upstream scheduler into existence.
        _ = self.pipeline

        if self._upstream_scheduler is None:
            self._upstream_scheduler = self.scheduler

        flow_shift = float(self.generation_config.flow_shift)

        if isinstance(self.scheduler, FlowMatchSDEDiscreteScheduler):
            # Already installed — just retune eta in place.
            self.scheduler._eta = float(eta)
            return
        sde = FlowMatchSDEDiscreteScheduler(
            num_train_timesteps=1000,
            shift=flow_shift,
            use_dynamic_shifting=False,
            base_shift=0.5,
            max_shift=1.15,
            time_shift_type="exponential",
            stochastic_sampling=False,
            eta=float(eta),
        )
        self.scheduler = sde
        if self._pipeline is not None:
            self._pipeline.set_scheduler(sde)

    def _install_prepare_inputs_hook(self) -> None:
        """Idempotently wrap ``transformer.prepare_inputs_for_generation``.

        The wrapper runs in front of every per-step model-input prep call
        but only writes to ``self._fused_capture`` when it's ``None`` —
        which is reset to ``None`` at the top of every ``forward(req)``
        call. Net effect: capture exactly the **first** per-request call
        (with full sequence length ``L``); ignore the gathered-down ``L'``
        calls that follow when KV-cache reuse kicks in on later steps.

        Field map matches our own per-step kernel
        (``models/hunyuan_image3/diffusion.py:212-231``), which calls
        ``prepare_inputs_for_generation`` with the same kwarg names that
        upstream uses internally.
        """
        if self._prepare_inputs_patched:
            return

        # Force inner pipeline materialization so ``self._pipeline`` and the
        # model ref are live. Mirrors ``_ensure_scheduler_for_eta``.
        #
        # The inner ``HunyuanImage3Text2ImagePipeline`` holds the MoE backbone
        # on ``.model`` (see upstream ``hunyuan_image3_transformer.py:2797``
        # which dispatches ``self.model.prepare_inputs_for_generation(...)``).
        # Despite the diffusers convention of ``pipeline.transformer``, the
        # t2i pipeline class doesn't expose that alias — only the outer
        # vllm-omni ``HunyuanImage3Pipeline`` wrapper aliases
        # ``self.transformer = self.model``, and that's a different object.
        _ = self.pipeline
        transformer = self._pipeline.model

        orig = transformer.prepare_inputs_for_generation
        pipeline_self = self

        def wrapped(*args: Any, **kw: Any) -> Any:
            if pipeline_self._fused_capture is None:
                # ``input_ids`` is the only positional arg upstream passes
                # (matching our own ``predict_noise`` at diffusion.py:213).
                input_ids = args[0] if args else kw.get("input_ids")
                pipeline_self._fused_capture = {
                    "input_ids": _detach_cpu(input_ids),
                    "attention_mask": _detach_cpu(kw.get("attention_mask")),
                    "position_ids": _detach_cpu(kw.get("position_ids")),
                    "rope_cache": _detach_cpu_pair(kw.get("custom_pos_emb")),
                    "gen_image_mask": _detach_cpu(kw.get("image_mask")),
                    "gen_timestep_scatter_index": _detach_cpu(kw.get("gen_timestep_scatter_index")),
                }
            return orig(*args, **kw)

        transformer.prepare_inputs_for_generation = wrapped
        self._prepare_inputs_patched = True

    def forward(self, req: OmniDiffusionRequest, **kwargs) -> DiffusionOutput:
        # Read eta off the typed field (``OmniDiffusionSamplingParams.eta``,
        # data.py:252). ``_ensure_scheduler_for_eta`` installs our scheduler
        # unconditionally (eta=0 still installs but the SDE branch never
        # fires; see the method docstring for why).
        eta = float(getattr(req.sampling_params, "eta", 0.0) or 0.0)
        self._ensure_scheduler_for_eta(eta)

        # Install (or reset) the sparse-SDE step gate on the scheduler.
        # See the matching block in
        # ``diffusionrl/rollout/engine/vllm_omni/sd3/pipeline.py`` for why
        # this MUST re-fire on every request (stale set from a previous
        # request would silently mis-gate SDE steps).
        if isinstance(self.scheduler, FlowMatchSDEDiscreteScheduler):
            extra = getattr(req.sampling_params, "extra_args", None) or {}
            sde_indices = extra.get("sde_indices")
            self.scheduler._sde_indices_set = (
                frozenset(int(i) for i in sde_indices) if sde_indices is not None else None
            )

        # Reset capture state for this request; install the hook lazily
        # (the inner pipeline / transformer may not exist before the first
        # ``_ensure_scheduler_for_eta`` call).
        self._fused_capture = None
        self._install_prepare_inputs_hook()

        # Delegate everything else (prompt construction, system prompt,
        # AR-bridged cot_text, batch_cond_image_info, prepare_model_inputs,
        # _generate) to upstream forward at pipeline_hunyuan_image3.py:1262.
        out = super().forward(req, **kwargs)

        # Drain trajectory off our scheduler. Belt-and-braces isinstance
        # check — ``_ensure_scheduler_for_eta`` always installs us, but a
        # future subclass override could swap it out.
        if isinstance(self.scheduler, FlowMatchSDEDiscreteScheduler):
            traj = self.scheduler.drain_trajectory()
            if traj is not None:
                latents, sigmas, _timesteps, log_probs = traj
                out.trajectory_latents = latents
                # ``trajectory_timesteps`` carries the true [0, 1] sigma
                # schedule — see module docstring. This is what
                # response.py reads as ``LatentSegment.sigmas`` and what
                # ``HunyuanImage3DiffusionStage.replay`` indexes per step.
                out.trajectory_timesteps = sigmas
                out.trajectory_log_probs = log_probs
                # Echo the real sparse SDE step ids via ``custom_output``
                # (runtime attrs on ``DiffusionOutput`` don't survive IPC).
                sde_step_indices = self.scheduler.last_sde_step_indices
                if out.custom_output is None:
                    out.custom_output = {}
                out.custom_output["sde_step_indices"] = sde_step_indices

        # Surface the captured fused MM tensors via ``DiffusionOutput.custom_output``
        # — a dataclass-declared dict that vllm-omni explicitly forwards into
        # ``OmniRequestOutput.custom_output`` (see upstream ``diffusion/data.py:841``
        # and ``stage_diffusion_proc.py:182``). Plain runtime attrs on
        # ``DiffusionOutput`` get filtered during IPC; that filtering is the
        # reason the trajectory_sigmas attr did not survive (see module docstring).
        # ``None`` if upstream's ``prepare_inputs_for_generation`` was never
        # called (unexpected for image modalities) — parent side treats absence
        # as "old-style" empty conditions, see response.py.
        if self._fused_capture is not None:
            if out.custom_output is None:
                out.custom_output = {}
            out.custom_output["fused_mm_capture"] = self._fused_capture
        return out


__all__ = ["RLHunyuanImage3Pipeline"]
