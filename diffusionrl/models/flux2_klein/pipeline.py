"""Flux2KleinPipeline — RolloutReq → RolloutResp end-to-end for FLUX.2-klein-9B.

Implements the typed four-tier flow::

    Texts ──text_embed──▶ Flux2KleinConditions ──diffuse──▶ LatentSegment
                                                              │
                                                              ▼
                                                         vae_decode
                                                              │
                                                              ▼
                                                            Images

Hydra constructs a pipeline via
``Flux2KleinPipeline.from_config(Flux2KleinPipelineConfig)`` (see
``config.py``); ``from_config`` loads the :class:`Flux2KleinBundle`
then constructs the four stages with the precision policy from the
config.

σ schedule contract
-------------------
The hosting engine (``TrainsideRolloutEngine`` / ``SGLangRolloutEngine``
/ ``VLLMOmniRolloutEngine``) pins ``req.sigmas`` via
:func:`diffusionrl.sde.runtime.ensure_req_sigmas` BEFORE calling
``generate(req)``; this pipeline reads ``req.sigmas`` and uses it
verbatim.

FLUX.2-klein-specific override: μ depends on both ``image_seq_len`` AND
``num_inference_steps`` (the linear-interp
:func:`calculate_dynamic_mu` used by SD3 / Qwen-Image only depends on
``image_seq_len``). :meth:`build_schedule_policy` returns a custom
:class:`Flux2KleinSchedulePolicy` that overrides only
:meth:`FlowMatchSchedulePolicy.compute_mu` (the μ value); the shared
:meth:`FlowMatchSchedulePolicy.compute_sigma` builds the schedule for all
models, Klein included.
"""

from __future__ import annotations

import dataclasses as _dc
from dataclasses import dataclass
from typing import Any, Dict, Optional

from diffusionrl.models.types.pipeline import Pipeline
from diffusionrl.sde.kernels import DanceSDEStrategy, StepStrategy
from diffusionrl.sde.runtime import FlowMatchSchedulePolicy
from diffusionrl.types.primitives import Texts
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutResp

from .bundle import Flux2KleinBundle
from .conditions import Flux2KleinConditions
from .config import Flux2KleinPipelineConfig
from .diffusion import (
    Flux2KleinDiffusionParams,
    Flux2KleinDiffusionStage,
    Flux2KleinDiffusionStep,
)
from .flux2_klein_utils import compute_empirical_mu
from .text_embed import Flux2KleinTextEmbedStage
from .vae import Flux2KleinVAEDecodeStage


@dataclass
class Flux2KleinSchedulePolicy(FlowMatchSchedulePolicy):
    """Policy subclass with FLUX.2-klein empirical-μ shifting.

    Klein's μ depends on both ``image_seq_len`` AND ``num_inference_steps``
    (diffusers' ``Flux2KleinPipeline.compute_empirical_mu``: piecewise
    linear in ``num_steps`` for ``image_seq_len ≤ 4300``, single line
    otherwise). The base linear ``calculate_dynamic_mu`` only depends on
    ``image_seq_len``, so we override :meth:`compute_mu` with the empirical
    formula. Everything else — base grid + diffusers time-shift +
    terminal zero — is unchanged and stays shared in
    :meth:`diffusionrl.sde.runtime.FlowMatchSchedulePolicy.compute_sigma`.
    """

    def compute_mu(self, image_seq_len: int, num_inference_steps: int) -> float:
        return compute_empirical_mu(image_seq_len, num_inference_steps)


class Flux2KleinPipeline(Pipeline):
    """FLUX.2-klein-9B generate pipeline.

    Reads from ``RolloutReq``:

    - ``primitives["text"]: Texts`` — required prompts.
    - ``primitives["negative_text"]: Texts`` — optional CFG negatives.
      The canonical Klein recipe runs at ``guidance_scale=1.0`` with
      no negative branch.
    - ``stage_params["diffusion"]: dict`` — kwargs for
      :class:`Flux2KleinDiffusionParams`.
    - ``sigmas: Tensor[T+1]`` — pinned by the engine adapter (required).

    Writes to ``RolloutResp``:

    - ``conditions["text"]: TextEmbedCondition``; plus
      ``conditions["negative_text"]: TextEmbedCondition`` when negative
      prompts were supplied.
    - ``rollout_traces["image"]: LatentSegment`` (patchified spatial
      shape ``[B, K, 128, H_pat, W_pat]``).
    - ``decoded["image"]: Images``.
    """

    def __init__(
        self,
        *,
        bundle: Flux2KleinBundle,
        text_embed: Flux2KleinTextEmbedStage,
        diffusion: Flux2KleinDiffusionStage,
        vae_decode: Flux2KleinVAEDecodeStage,
        shift: float = 1.0,
    ) -> None:
        self.bundle = bundle
        self.text_embed = text_embed
        self.diffusion = diffusion
        self.vae_decode = vae_decode
        # ``shift`` is retained as an attribute for the hosting engine
        # to read when constructing the σ policy. For Klein, the
        # empirical-μ schedule fully replaces static shifting at
        # runtime — the value is only a fallback when the bundle's
        # scheduler can't be loaded.
        self.shift = shift

    def build_schedule_policy(self):
        """Build the Klein-specific schedule policy.

        FLUX.2-klein-9B was trained with an empirical-μ schedule that
        depends on **both** the packed image_seq_len AND the number of
        inference steps. The standard :class:`FlowMatchSchedulePolicy`
        only encodes the image_seq_len → μ mapping linearly
        (``calculate_dynamic_mu``), so we return a Klein-specific subclass
        that overrides :meth:`compute_mu` with the empirical formula. The
        σ application (base grid + diffusers time-shift) is the shared
        dynamic-shift path. ``time_shift_type`` must match the checkpoint's
        ``scheduler_config.json`` (FLUX.2 uses ``"exponential"``).
        """
        return Flux2KleinSchedulePolicy(
            shift=float(self.shift),
            use_dynamic_shifting=True,
            vae_scale_factor=8,
            patch_size=2,
            time_shift_type="exponential",
        )

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> tuple:
        """Per-sample patchified latent shape ``(C_pack=128, H_pat, W_pat)``
        for driver-side noise pre-computation.

        FLUX.2-klein-9B: 32-channel post-VAE latents (``AutoencoderKLFlux2``),
        2×2 channel-packed for the transformer input (128 = 32 × 4),
        post-VAE spatial 8× downsample plus the patchify factor of 2.
        ``Flux2KleinDiffusionStage`` operates directly on the patchified
        shape ``[B, 128, H_pix/16, W_pix/16]``; the driver-shipped
        initial-noise tensor must match this geometry.
        """
        height = int(sampling_spec.height)
        width = int(sampling_spec.width)
        downsample = 8 * 2  # vae_scale_factor × patchify_factor
        if height % downsample != 0 or width % downsample != 0:
            raise ValueError(
                f"Flux2KleinPipeline.latent_shape: height ({height}) and width "
                f"({width}) must be divisible by VAE×patchify downsample "
                f"({downsample})."
            )
        return (128, height // downsample, width // downsample)

    @classmethod
    def from_config(
        cls,
        config: Flux2KleinPipelineConfig,
        *,
        strategy: Optional[StepStrategy] = None,
    ) -> "Flux2KleinPipeline":
        """Build the full pipeline from a config.

        ``strategy`` defaults to :class:`DanceSDEStrategy` — the
        canonical Klein training-script setting
        (``main_flux_bundle/reproduce_scripts/train_grpo_flux2_klein9b_sglang_multinode.sh``
        sets ``SDE_TYPE=dance``). Callers running an alternate SDE
        family pass an explicit strategy.
        """
        bundle = Flux2KleinBundle.from_config(config)
        text_embed = Flux2KleinTextEmbedStage(
            bundle,
            max_sequence_length=config.max_sequence_length,
            extraction_layers=tuple(config.qwen3_extraction_layers),
        )
        step = Flux2KleinDiffusionStep()
        diffusion = Flux2KleinDiffusionStage(
            model=bundle,
            step=step,
            strategy=strategy if strategy is not None else DanceSDEStrategy(),
            autocast_precision=config.autocast_precision,
            trajectory_precision=config.trajectory_precision,
            logprob_precision=config.logprob_precision,
        )
        vae_decode = Flux2KleinVAEDecodeStage(bundle)
        return cls(
            bundle=bundle,
            text_embed=text_embed,
            diffusion=diffusion,
            vae_decode=vae_decode,
            shift=float(config.shift),
        )

    def generate(self, req: RolloutReq) -> RolloutResp:
        """Run FLUX.2-klein-9B t2i end-to-end. Requires ``req.sigmas``
        to be pinned by the hosting engine adapter."""
        if req.sigmas is None:
            raise ValueError(
                "Flux2KleinPipeline.generate: req.sigmas is None. The hosting "
                "engine (Trainside / SGLang / VLLMOmni) must call "
                "diffusionrl.sde.runtime.ensure_req_sigmas(req, policy) before "
                "invoking pipeline.generate; see the σ ownership note in "
                "diffusionrl.models.types.pipeline."
            )
        texts = req.primitives.get("text")
        if not isinstance(texts, Texts):
            raise TypeError(
                f"Flux2KleinPipeline.generate: req.primitives['text'] must be Texts, "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )
        negatives_raw = req.primitives.get("negative_text")
        negatives = negatives_raw if isinstance(negatives_raw, Texts) else None
        if negatives is not None and len(negatives.texts) != len(texts.texts):
            raise ValueError(
                f"Flux2KleinPipeline.generate: negative_text length "
                f"{len(negatives.texts)} != text length {len(texts.texts)}"
            )

        raw_dict: Dict[str, Any] = dict(req.stage_params.get("diffusion") or {})
        allowed = {f.name for f in _dc.fields(Flux2KleinDiffusionParams)}
        params_dict = {k: v for k, v in raw_dict.items() if k in allowed}
        params = Flux2KleinDiffusionParams(**params_dict)

        text_cond = self.text_embed.embed(texts)
        # CFG empty negative: Klein's canonical training-script setting is
        # ``guidance_scale=1.0`` (the script literally hardcodes it; see
        # ``main_flux_bundle/reproduce_scripts/train_grpo_flux2_klein9b_sglang_multinode.sh``).
        # When CFG is OFF, no negative branch is needed and we leave
        # ``negative_text=None`` so the transformer runs only the
        # conditional forward. When a downstream user opts in to
        # ``guidance_scale > 1`` without supplying ``negative_text``,
        # default to an empty string (the Qwen3 chat-template tokenizer
        # is robust to ``""``: no chat-template prefix is stripped, so
        # the resulting embedding is well-defined).
        if negatives is None and float(params.guidance_scale) > 1.0:
            negatives = Texts(texts=[""] * len(texts.texts))
        negative_text_cond = self.text_embed.embed(negatives) if negatives is not None else None
        klein_conds = Flux2KleinConditions(text=text_cond, negative_text=negative_text_cond)

        schedule = req.sigmas.to(self.bundle.device)

        initial_cond = (req.request_conditions or {}).get("initial_latents")
        initial_latents = getattr(initial_cond, "latents", None) if initial_cond is not None else None

        latent_seg = self.diffusion.diffuse(
            klein_conds, schedule=schedule, params=params, initial_latents=initial_latents
        )
        images = self.vae_decode.decode(latent_seg)

        return RolloutResp(
            sample_ids=list(req.sample_ids),
            group_ids=list(req.group_ids),
            conditions=klein_conds.to_dict(),
            rollout_traces={"image": latent_seg},
            decoded={"image": images},
        )


__all__ = ["Flux2KleinPipeline", "Flux2KleinSchedulePolicy"]
