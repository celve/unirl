"""RL-aware Wan2.2 T2V pipeline subclass.

``forward`` follows the RL interception protocol (see
``pipelines/_shared/interception.py``): **install** (once) → **arm** (every
request) → run (upstream) → **harvest**. The interceptions, mapped to
upstream's stages (``vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py``):

- SDE scheduler swap (behavior policy + dense-trajectory recorder) in place
  of the upstream Wan scheduler. Constructed DIRECTLY (not via
  ``make_sde_scheduler``): upstream's default donor is solver-dependent and
  ``WanEulerScheduler.config`` is a ``SimpleNamespace``, not a ConfigMixin
  FrozenDict. ``num_train_timesteps=1000`` matches both Wan solvers and is
  what upstream reads for the expert ``boundary_timestep``.
- A schedule **pin** (``_arm_schedule_pins``): upstream ``forward`` REBUILDS
  ``self.scheduler`` whenever the request's resolved ``sample_solver`` /
  ``flow_shift`` differ from the cached values — which would silently evict
  the SDE scheduler. Pre-syncing the cached values from the same resolvers
  makes the rebuild comparison always false.
- A conditioning **tap** on ``encode_prompt`` (UMT5 single stream, 2-tuple)
  for the trainer-side ``WAN21Conditions`` reconstruction.
- An initial-noise **injection** through the ``prepare_latents`` override
  (driver-authored x_T slice or recipe row replaces upstream's RNG draw) —
  with a shape assert: upstream accepts injected latents unvalidated and
  silently snaps ``num_frames`` to ``4k+1``, so a driver/engine shape
  mismatch would otherwise surface as garbage sampling.
- A σ-schedule **workaround**: like HV1.5, upstream Wan ignores
  ``req.sampling_params.sigmas`` (see :meth:`_sigma_override`).

Dual experts: the high-noise ``transformer`` / low-noise ``transformer_2``
routing (σ ≥/< ``boundary_ratio``) stays entirely upstream — the SDE
scheduler is expert-agnostic. ``_assert_expert_coverage`` refuses requests
whose σ schedule would visit a stage whose transformer is not loaded
(upstream falls back to the OTHER expert silently — a wrong-policy rollout
the trainer could not replay).

This class is loaded inside vLLM-Omni's worker subprocess via
``custom_pipeline_args.pipeline_class`` injected from
``stage_configs/wan22_t2v_rl.yaml``.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
    Wan22Pipeline,
    resolve_wan_flow_shift,
    resolve_wan_sample_solver,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest

from unirl.rollout.engine.vllm_omni.pipelines._shared.flow_match_sde_scheduler import (
    FlowMatchSDEDiscreteScheduler,
)
from unirl.rollout.engine.vllm_omni.pipelines._shared.interception import (
    detach_cpu,
    drain_trajectory_into,
    inject_latents,
    resolve_request_noise,
    stamp_custom_output,
)


class RLWan22Pipeline(Wan22Pipeline):
    """Wan2.2 T2V pipeline with the RL interception protocol installed."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        if self.expand_timesteps:
            raise RuntimeError(
                "RLWan22Pipeline supports the T2V-A14B family only; this checkpoint "
                "declares expand_timesteps (TI2V-5B per-patch timesteps), which the "
                "trainer-side WAN22DiffusionStep does not replay."
            )
        # Conditioning-tap state: armed (reset) every request, filled by the
        # tap's first call; the flag keeps the install idempotent.
        self._captured_conditioning: Optional[Dict[str, Any]] = None
        self._conditioning_tap_installed: bool = False
        # Per-request x_T hand-off (same pattern as SD3/HV1.5).
        self._pending_initial_noise: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    # install — once per pipeline lifetime, idempotent
    # ------------------------------------------------------------------ #

    def _install_sde_scheduler(self) -> None:
        """Swap in the trajectory-capturing SDE scheduler.

        ``shift`` is a fallback only — RL always pins σ via
        :meth:`_sigma_override`, and ``set_timesteps(sigmas=...)``
        neutralizes every shift source. Per-request eta rides ``_arm_sde``.
        """
        if isinstance(self.scheduler, FlowMatchSDEDiscreteScheduler):
            return
        self.scheduler = FlowMatchSDEDiscreteScheduler(
            num_train_timesteps=1000, shift=float(self._flow_shift)
        )

    def _install_conditioning_tap(self) -> None:
        """Wrap ``encode_prompt`` to capture the UMT5 embeddings.

        Wan2.2 returns ``(prompt_embeds, negative_prompt_embeds)``; the
        negative is ``None`` when CFG is off (guidance 1.0 — the parity
        recipes). Embeddings are already truncated-then-zero-padded to
        ``max_sequence_length`` upstream, matching the trainer's
        ``WAN21TextEmbedStage`` convention, so the capture replays directly.
        First-call-only per request (the buffer is re-armed each ``forward``).
        """
        if self._conditioning_tap_installed:
            return

        orig = self.encode_prompt
        pipeline_self = self

        def tapped(*args: Any, **kw: Any) -> Any:
            result = orig(*args, **kw)
            if pipeline_self._captured_conditioning is None:
                prompt_embeds, negative_prompt_embeds = result
                pipeline_self._captured_conditioning = {
                    "prompt_embeds": detach_cpu(prompt_embeds),
                    "negative_prompt_embeds": detach_cpu(negative_prompt_embeds),
                }
            return result

        self.encode_prompt = tapped  # type: ignore[assignment]
        self._conditioning_tap_installed = True

    # ------------------------------------------------------------------ #
    # arm — every request (stale-leak guards)
    # ------------------------------------------------------------------ #

    def _arm_schedule_pins(self, req: OmniDiffusionRequest) -> None:
        """Neutralize upstream's in-forward scheduler rebuild.

        ``forward`` re-resolves ``sample_solver``/``flow_shift`` and REPLACES
        ``self.scheduler`` when either differs from the cached values —
        evicting the SDE scheduler. Resolve with the same module functions
        upstream uses and sync the cache, so the comparison is always false.
        """
        self._sample_solver = resolve_wan_sample_solver(req, default=self._sample_solver)
        self._flow_shift = resolve_wan_flow_shift(req, self.od_config)

    def _arm_sde(self, req: OmniDiffusionRequest) -> None:
        """This request's SDE strength + sparse step gate."""
        eta = float(getattr(req.sampling_params, "eta", 0.0) or 0.0)
        extra = getattr(req.sampling_params, "extra_args", None) or {}
        self.scheduler.arm(eta=eta, sde_indices=extra.get("sde_indices"))

    def _arm_initial_noise(self, req: OmniDiffusionRequest) -> None:
        """This request's driver-authored x_T (batch slice or recipe row)."""
        self._pending_initial_noise = resolve_request_noise(req, caller="RLWan22Pipeline._arm_initial_noise")

    def _arm_conditioning_tap(self) -> None:
        """Fresh capture buffer so the tap records THIS request's first encode."""
        self._captured_conditioning = None

    def _resolve_boundary_ratio(self, req: OmniDiffusionRequest) -> float:
        """Mirror upstream ``forward``'s boundary resolution (engine config >
        request > warned default 0.875)."""
        boundary = self.boundary_ratio
        if boundary is None:
            boundary = getattr(req.sampling_params, "boundary_ratio", None)
        if boundary is None:
            boundary = 0.875
        return float(boundary)

    def _assert_expert_coverage(self, req: OmniDiffusionRequest) -> None:
        """Refuse σ schedules that would hit upstream's silent expert fallback.

        Upstream's per-step routing falls back to whichever transformer IS
        loaded when the stage's own transformer is missing — a different
        policy than the trainer replays, corrupting the ratio without any
        error. Boundary configs 0.0/1.0 (single-expert) stay legal as long as
        the schedule never enters the missing stage.
        """
        sigmas = getattr(req.sampling_params, "sigmas", None)
        if sigmas is None:
            return  # non-RL smoke path; sigma_verify would catch RL misuse
        boundary = self._resolve_boundary_ratio(req)
        step_sigmas = [float(s) for s in sigmas][:-1]  # drop terminal σ=0
        if self.transformer_2 is None and any(s < boundary for s in step_sigmas):
            raise RuntimeError(
                f"RLWan22Pipeline: σ schedule enters the low-noise stage (σ < {boundary}) "
                "but transformer_2 is not loaded (boundary_ratio=0.0 load policy). Upstream "
                "would silently run the high-noise expert there — not replayable."
            )
        if self.transformer is None and any(s >= boundary for s in step_sigmas):
            raise RuntimeError(
                f"RLWan22Pipeline: σ schedule enters the high-noise stage (σ >= {boundary}) "
                "but transformer is not loaded (boundary_ratio=1.0 load policy). Upstream "
                "would silently run the low-noise expert there — not replayable."
            )

    def _assert_parity_guidance(self, req: OmniDiffusionRequest) -> None:
        """Parity mode pins CFG off: the engine runs CFG as two sequential
        forwards while the trainer batches 2M — a geometry seam we do not
        close. Gated on the worker-side parity env set by
        ``VLLMOmniBackend.boot``."""
        if os.environ.get("UNIRL_VLLM_OMNI_PARITY") != "1":
            return
        if getattr(req.sampling_params, "sigmas", None) is None:
            return  # non-RL traffic (engine boot dummy warmup) — no trajectory is replayed
        gs = getattr(req.sampling_params, "guidance_scale", None)
        provided = bool(getattr(req.sampling_params, "guidance_scale_provided", False))
        gs2 = getattr(req.sampling_params, "guidance_scale_2", None)
        if not provided or float(gs) != 1.0 or (gs2 is not None and float(gs2) != 1.0):
            raise RuntimeError(
                f"RLWan22Pipeline parity mode requires guidance_scale == 1.0 for both "
                f"experts (got guidance_scale={gs!r}, guidance_scale_2={gs2!r}). CFG "
                "execution shape differs between engine (two M forwards) and trainer "
                "(one 2M forward)."
            )

    # ------------------------------------------------------------------ #
    # run-phase interceptions
    # ------------------------------------------------------------------ #

    def _assert_noise_shape(self, noise: torch.Tensor, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        def _arg(idx: int, name: str) -> Any:
            return args[idx] if len(args) > idx else kwargs.get(name)

        batch_size = _arg(0, "batch_size")
        num_channels = _arg(1, "num_channels_latents")
        height = _arg(2, "height")
        width = _arg(3, "width")
        num_frames = _arg(4, "num_frames")
        if None in (batch_size, num_channels, height, width, num_frames):
            return  # unexpected partial call shape — let upstream handle it
        expected = (
            int(batch_size),
            int(num_channels),
            (int(num_frames) - 1) // self.vae_scale_factor_temporal + 1,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if tuple(noise.shape) != expected:
            raise RuntimeError(
                f"RLWan22Pipeline: driver-authored x_T shape {tuple(noise.shape)} != upstream "
                f"latent shape {expected} (upstream snaps num_frames to 4k+1 and never "
                "validates injected latents). Check the driver's latent recipe "
                "(WAN22Pipeline.latent_shape) against the request's num_frames/height/width."
            )

    def prepare_latents(self, *args, **kwargs):  # type: ignore[override]
        """Initial-noise injection point (consume-once).

        Upstream calls this with all-keyword args; the explicit positional
        indices (Wan layout: dtype@5, device@6, latents@8) future-proof
        against upstream switching to positional calls.
        """
        noise = self._pending_initial_noise
        if noise is not None:
            self._pending_initial_noise = None
            self._assert_noise_shape(noise, args, kwargs)
            args, kwargs = inject_latents(args, kwargs, noise, dtype_idx=5, device_idx=6, latents_idx=8)
        return super().prepare_latents(*args, **kwargs)

    @contextmanager
    def _sigma_override(self, req: OmniDiffusionRequest) -> Iterator[None]:
        """WORKAROUND: make upstream pick up the engine's σ schedule.

        Upstream Wan2.2 calls ``self.scheduler.set_timesteps(num_steps,
        device=device)`` ignoring ``req.sampling_params.sigmas`` (same defect
        class as HV1.5; sd3/qwen_image/flux honor it). Without this the
        worker runs upstream's own shift grid while the driver pinned a
        (possibly different) schedule, tripping ``sigma_verify`` and aborting
        rollout.

        Patches the scheduler's ``set_timesteps`` for the duration of ONE
        ``forward`` and always restores — the closure must never leak across
        requests. Delete once upstream honors the request sigmas.
        """
        engine_sigmas = getattr(req.sampling_params, "sigmas", None)
        if engine_sigmas is None:
            yield
            return

        sched = self.scheduler
        orig_set_timesteps = sched.set_timesteps

        def _set_timesteps_with_engine_sigmas(*args: Any, **kw: Any) -> Any:
            kw["sigmas"] = engine_sigmas
            return orig_set_timesteps(*args, **kw)

        sched.set_timesteps = _set_timesteps_with_engine_sigmas  # type: ignore[assignment]
        try:
            yield
        finally:
            del sched.set_timesteps

    # ------------------------------------------------------------------ #
    # harvest — export onto the wire
    # ------------------------------------------------------------------ #

    def _harvest_trajectory(self, out: DiffusionOutput) -> None:
        if not isinstance(self.scheduler, FlowMatchSDEDiscreteScheduler):
            raise RuntimeError(
                "RLWan22Pipeline: upstream forward replaced the SDE scheduler "
                f"(now {type(self.scheduler).__name__}) — the _arm_schedule_pins "
                "neutralization no longer covers upstream's rebuild triggers."
            )
        drain_trajectory_into(out, self.scheduler)

    def _harvest_conditioning(self, out: DiffusionOutput) -> None:
        if self._captured_conditioning is not None:
            stamp_custom_output(out, "text_capture", self._captured_conditioning)

    # ------------------------------------------------------------------ #
    # the protocol
    # ------------------------------------------------------------------ #

    def forward(self, req: OmniDiffusionRequest, **kwargs) -> DiffusionOutput:
        self._install_sde_scheduler()
        self._install_conditioning_tap()

        self._arm_schedule_pins(req)
        self._assert_expert_coverage(req)
        self._assert_parity_guidance(req)
        self._arm_sde(req)
        self._arm_initial_noise(req)
        self._arm_conditioning_tap()

        # Delegate to upstream (encode, latent prep, expert-routed denoise
        # loop, VAE decode); the installed tap/injector fire inside.
        with self._sigma_override(req):
            out = super().forward(req, **kwargs)

        # Video PILs do NOT survive the worker->client wire — only tensors in
        # custom_output / trajectory_* cross (LIN-382). Stamp the decoded
        # video tensor ([B, C, F, H, W], [-1, 1]) so ``collect_dit_outputs``
        # can recover frames for the reward.
        decoded = getattr(out, "output", None)
        if decoded is not None:
            stamp_custom_output(out, "rl_decoded_video", detach_cpu(decoded))

        self._harvest_trajectory(out)
        self._harvest_conditioning(out)
        return out


__all__ = ["RLWan22Pipeline"]
