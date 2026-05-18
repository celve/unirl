"""HunyuanVideo-1.5 FSDP-native sampler.

Mirrors the structure of :class:`FSDPWanSampler` — the model-private call goes
through ``model_bundle.forward_denoiser(...)``; the sampler only owns the
sampling loop, latent shape, sigma schedule, trajectory recording and
``ForwardContext`` construction.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from diffusionrl.sde.kernels import DanceSDEStrategy, StepStrategy
from diffusionrl.sde.runtime import denoising_step, sd3_time_shift
from diffusionrl.types.forward_context import HunyuanVeido1p5ForwardContext
from diffusionrl.types.sample import LogProbData
from diffusionrl.types.trajectory_store import TrajectoryBuilder

from ..base import RolloutSamples
from .base_sampler import FSDPBaseSampler

logger = logging.getLogger(__name__)


class FSDPHunyuanVeido1p5Sampler(FSDPBaseSampler):
    """SDE / ODE sampler for HunyuanVideo-1.5 (T2V + I2V).

    Latent shape is derived from the VAE compression ratios exposed by the
    bundle's VAE config; output channels come from the transformer config (32
    by default in upstream checkpoints). The transformer's expected
    ``in_channels`` is ``2 * latent_channels + 1`` because of the
    ``cat([latents, cond_latents, cond_mask], dim=1)`` packing — the bundle
    handles that concat inside ``forward_denoiser``.
    """

    DEFAULT_SPATIAL_DOWNSAMPLE = 16
    DEFAULT_TEMPORAL_DOWNSAMPLE = 4
    DEFAULT_LATENT_CHANNELS = 32

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        text_encoder: Optional[Any] = None,
        vae: Optional[nn.Module] = None,
        eta: float = 1.0,
        strategy: Optional[StepStrategy] = None,
        shift: float = 5.0,
        guidance_scale: float = 6.0,
        autocast_precision: Any = "bf16",
        trajectory_precision: Any = "fp16",
        logprob_precision: Any = "fp32",
        **kwargs: Any,
    ):
        super().__init__(
            model=model,
            text_encoder=text_encoder,
            vae=vae,
            eta=eta,
            strategy=strategy if strategy is not None else DanceSDEStrategy(),
            shift=shift,
            autocast_precision=autocast_precision,
            trajectory_precision=trajectory_precision,
            logprob_precision=logprob_precision,
            **kwargs,
        )
        self.default_guidance_scale = float(guidance_scale)

    # ------------------------------------------------------------------
    # Latent geometry
    # ------------------------------------------------------------------

    def _latent_shape(self, *, height: int, width: int, num_frames: int) -> Tuple[int, int, int]:
        spatial = self.DEFAULT_SPATIAL_DOWNSAMPLE
        temporal = self.DEFAULT_TEMPORAL_DOWNSAMPLE
        if self.vae is not None:
            attr_spatial = getattr(self.vae, "spatial_compression_ratio", None)
            attr_temporal = getattr(self.vae, "temporal_compression_ratio", None)
            if attr_spatial is not None:
                spatial = int(attr_spatial)
            if attr_temporal is not None:
                temporal = int(attr_temporal)
        latent_t = ((int(num_frames) - 1) // temporal) + 1
        latent_h = max(1, int(height) // spatial)
        latent_w = max(1, int(width) // spatial)
        return latent_t, latent_h, latent_w

    def _latent_channels(self) -> int:
        if self.vae is not None:
            channels = getattr(getattr(self.vae, "config", None), "latent_channels", None)
            if channels is not None:
                return int(channels)
        config = getattr(self.model, "config", None)
        out_channels = getattr(config, "out_channels", None)
        return int(out_channels) if out_channels is not None else self.DEFAULT_LATENT_CHANNELS

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        prompts: Optional[List[str]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        prompt_embeds_2: Optional[torch.Tensor] = None,
        prompt_embeds_mask_2: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds_2: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask_2: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        cond_latents: Optional[torch.Tensor] = None,
        cond_mask: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = None,
        height: int = 480,
        width: int = 848,
        num_frames: int = 121,
        latents: Optional[torch.Tensor] = None,
        base_seed: Optional[int] = None,
        sde_indices: Optional[Set[int]] = None,
        init_same_noise: bool = False,
        samples_per_prompt: int = 1,
        noise_group_ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> RolloutSamples:
        del kwargs
        if self.model is None:
            raise RuntimeError("Model not set. Initialize sampler with model parameter.")
        if self.model_bundle is None:
            raise RuntimeError("FSDPHunyuanVeido1p5Sampler requires model_bundle for model-specific forward dispatch")
        if prompt_embeds is None:
            raise ValueError("FSDPHunyuanVeido1p5Sampler requires prompt_embeds (use bundle.encode_inputs)")

        device = self._resolve_runtime_device(prompt_embeds=prompt_embeds, latents=latents)
        batch_size = int(prompt_embeds.shape[0])
        actual_guidance = float(guidance_scale) if guidance_scale is not None else self.default_guidance_scale

        # Cast positive / negative streams to autocast dtype on the right device.
        prompt_embeds = prompt_embeds.to(device=device, dtype=self.autocast_dtype)
        if prompt_embeds_2 is not None:
            prompt_embeds_2 = prompt_embeds_2.to(device=device, dtype=self.autocast_dtype)
        if prompt_embeds_mask is not None:
            prompt_embeds_mask = prompt_embeds_mask.to(device=device)
        if prompt_embeds_mask_2 is not None:
            prompt_embeds_mask_2 = prompt_embeds_mask_2.to(device=device)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=self.autocast_dtype)
        if negative_prompt_embeds_2 is not None:
            negative_prompt_embeds_2 = negative_prompt_embeds_2.to(device=device, dtype=self.autocast_dtype)
        if negative_prompt_embeds_mask is not None:
            negative_prompt_embeds_mask = negative_prompt_embeds_mask.to(device=device)
        if negative_prompt_embeds_mask_2 is not None:
            negative_prompt_embeds_mask_2 = negative_prompt_embeds_mask_2.to(device=device)
        if image_embeds is not None:
            image_embeds = image_embeds.to(device=device, dtype=self.autocast_dtype)
        if cond_latents is not None:
            cond_latents = cond_latents.to(device=device, dtype=self.autocast_dtype)
        if cond_mask is not None:
            cond_mask = cond_mask.to(device=device, dtype=self.autocast_dtype)

        # Latent shape / channels.
        latent_t, latent_h, latent_w = self._latent_shape(
            height=height,
            width=width,
            num_frames=num_frames,
        )
        latent_channels = self._latent_channels()
        trajectory_dtype = self.trajectory_dtype

        if latents is None:
            from ..utils.noise import generate_latents

            latents = generate_latents(
                batch_size=batch_size,
                latent_shape=(latent_channels, latent_t, latent_h, latent_w),
                device=device,
                dtype=trajectory_dtype,
                init_same_noise=init_same_noise,
                samples_per_prompt=samples_per_prompt,
                noise_group_ids=noise_group_ids,
                base_seed=base_seed,
            )
        else:
            latents = latents.to(device=device, dtype=trajectory_dtype)

        # Sigma schedule — flow-matching with sd3_time_shift, matching upstream
        # ``sigmas = np.linspace(1, 0, n+1)[:-1]`` then scheduler set_timesteps.
        sigma_schedule = torch.linspace(1, 0, int(num_inference_steps) + 1)
        sigma_schedule = sd3_time_shift(self.shift, sigma_schedule).to(device)

        if sde_indices is None:
            sde_indices = set() if self.uses_deterministic_solver else set(range(num_inference_steps))

        strategy = self.strategy
        strategy.init_schedule(sigma_schedule)

        trajectory_store = TrajectoryBuilder.for_sde_steps(sde_indices, num_inference_steps)
        trajectory_store.add(0, latents.clone().to(dtype=trajectory_dtype))
        all_log_probs: Dict[int, torch.Tensor] = {}

        autocast_ctx = (
            torch.autocast("cuda", self.autocast_dtype)
            if device.type == "cuda" and self.autocast_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )

        # Build a long-lived ForwardContext used for every step. We keep one
        # instance per call because all conditioning is shared across steps —
        # only ``latents`` and ``sigma`` vary.
        forward_context = HunyuanVeido1p5ForwardContext(
            guidance_scale=actual_guidance,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            prompt_embeds_2=prompt_embeds_2,
            prompt_embeds_mask_2=prompt_embeds_mask_2,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            negative_prompt_embeds_2=negative_prompt_embeds_2,
            negative_prompt_embeds_mask_2=negative_prompt_embeds_mask_2,
            image_embeds=image_embeds,
            cond_latents=cond_latents,
            cond_mask=cond_mask,
            attention_kwargs=attention_kwargs,
        )

        self.model.eval()
        for i in range(num_inference_steps):
            sigma = sigma_schedule[i].to(device)
            sigma_next = sigma_schedule[i + 1].to(device)

            with torch.no_grad():
                with autocast_ctx:
                    noise_pred = self.model_bundle.forward_denoiser(
                        latents=latents,
                        sigma=sigma,
                        ctx=forward_context,
                    )

            step_eta = self.eta if i in sde_indices else 0.0
            latents, log_prob, _ = denoising_step(
                noise_pred=noise_pred,
                sample=latents,
                sigma=sigma,
                sigma_next=sigma_next,
                eta=step_eta,
                sigma_max=sigma_schedule[1].item(),
                strategy=strategy,
                step_index=i,
            )
            latents = latents.to(dtype=trajectory_dtype)
            trajectory_store.add(i + 1, latents)
            if log_prob is not None:
                all_log_probs[i] = log_prob.to(dtype=self.logprob_dtype)

        return RolloutSamples(
            latents=latents.to(torch.float32),
            timesteps=sigma_schedule,
            trajectories=trajectory_store.finalize(),
            log_probs=LogProbData.from_dict(all_log_probs),
            forward_context=forward_context,
            step_indices=torch.arange(sigma_schedule.shape[0], device=sigma_schedule.device, dtype=torch.long),
        )


__all__ = ["FSDPHunyuanVeido1p5Sampler"]
