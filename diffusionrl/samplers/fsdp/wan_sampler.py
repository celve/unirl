"""WAN 2.1 FSDP-native sampler."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from diffusionrl.sde.kernels import DanceSDEStrategy, StepStrategy
from diffusionrl.sde.runtime import denoising_step, sd3_time_shift
from diffusionrl.types.forward_context import WAN21ForwardContext
from diffusionrl.types.sample import LogProbData
from diffusionrl.types.trajectory_store import TrajectoryBuilder

from ..base import RolloutSamples
from .base_sampler import FSDPBaseSampler

logger = logging.getLogger(__name__)


class FSDPWanSampler(FSDPBaseSampler):
    """WAN sampler supporting T2V and I2V conditioning tensors."""

    DEFAULT_SPATIAL_DOWNSAMPLE = 8
    DEFAULT_TEMPORAL_DOWNSAMPLE = 4
    DEFAULT_LATENT_CHANNELS = 16

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        text_encoder: Optional[Any] = None,
        vae: Optional[nn.Module] = None,
        eta: float = 1.0,
        strategy: Optional[StepStrategy] = None,
        shift: float = 5.0,
        guidance_scale: float = 5.0,
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

    def _latent_shape(self, *, height: int, width: int, num_frames: int) -> Tuple[int, int, int]:
        vae_cfg = getattr(self.vae, "config", None)
        spatial_downsample = int(getattr(vae_cfg, "scale_factor_spatial", self.DEFAULT_SPATIAL_DOWNSAMPLE))
        temporal_downsample = int(getattr(vae_cfg, "scale_factor_temporal", self.DEFAULT_TEMPORAL_DOWNSAMPLE))
        latent_t = ((int(num_frames) - 1) // temporal_downsample) + 1
        latent_h = int(height) // spatial_downsample
        latent_w = int(width) // spatial_downsample
        return latent_t, latent_h, latent_w

    def _latent_channels(self) -> int:
        config = getattr(self.model, "config", None)
        out_channels = getattr(config, "out_channels", None)
        return int(out_channels) if out_channels is not None else self.DEFAULT_LATENT_CHANNELS

    def _encode_prompts_if_needed(
        self,
        prompts: Optional[List[str]],
        prompt_embeds: Optional[torch.Tensor],
        negative_prompt_embeds: Optional[torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if prompt_embeds is not None:
            return prompt_embeds, negative_prompt_embeds
        if prompts is None:
            raise ValueError("Either prompts or prompt_embeds must be provided")
        if self.text_encoder is None or not hasattr(self.text_encoder, "encode_prompt"):
            raise RuntimeError("WAN sampler requires prompt_embeds or a compatible text_encoder")

        prompt_embeds, _ = self.text_encoder.encode_prompt(prompts, None)
        prompt_embeds = prompt_embeds.to(device=device)
        if negative_prompt_embeds is None and self.default_guidance_scale > 1.0:
            negative_prompt_embeds, _ = self.text_encoder.encode_prompt([""] * len(prompts), None)
            negative_prompt_embeds = negative_prompt_embeds.to(device=device)
        return prompt_embeds, negative_prompt_embeds

    def sample(
        self,
        prompts: Optional[List[str]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        negative_encoder_hidden_states_image: Optional[torch.Tensor] = None,
        image_conditioning_latents: Optional[torch.Tensor] = None,
        first_frame_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 40,
        guidance_scale: Optional[float] = None,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 81,
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

        device = self._resolve_runtime_device(prompt_embeds=prompt_embeds, latents=latents)
        prompt_embeds, negative_prompt_embeds = self._encode_prompts_if_needed(
            prompts,
            prompt_embeds,
            negative_prompt_embeds,
            device,
        )
        batch_size = int(prompt_embeds.shape[0])
        actual_guidance = float(guidance_scale) if guidance_scale is not None else self.default_guidance_scale

        prompt_embeds = prompt_embeds.to(device=device, dtype=self.autocast_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=self.autocast_dtype)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = encoder_hidden_states_image.to(device=device, dtype=self.autocast_dtype)
        if negative_encoder_hidden_states_image is not None:
            negative_encoder_hidden_states_image = negative_encoder_hidden_states_image.to(
                device=device,
                dtype=self.autocast_dtype,
            )
        if image_conditioning_latents is not None:
            image_conditioning_latents = image_conditioning_latents.to(device=device, dtype=self.autocast_dtype)
        if first_frame_mask is not None:
            first_frame_mask = first_frame_mask.to(device=device, dtype=self.autocast_dtype)

        latent_t, latent_h, latent_w = self._latent_shape(height=height, width=width, num_frames=num_frames)
        latent_channels = self._latent_channels()
        if latents is None:
            from ..noise_utils import generate_latents

            latents = generate_latents(
                batch_size=batch_size,
                latent_shape=(latent_channels, latent_t, latent_h, latent_w),
                device=device,
                dtype=self.trajectory_dtype,
                init_same_noise=init_same_noise,
                samples_per_prompt=samples_per_prompt,
                noise_group_ids=noise_group_ids,
                base_seed=base_seed,
            )
        else:
            latents = latents.to(device=device, dtype=self.trajectory_dtype)

        sigma_schedule = torch.linspace(1, 0, int(num_inference_steps) + 1)
        sigma_schedule = sd3_time_shift(self.shift, sigma_schedule).to(device)
        if sde_indices is None:
            sde_indices = set() if self.uses_deterministic_solver else set(range(num_inference_steps))

        strategy = self.strategy
        strategy.init_schedule(sigma_schedule)
        trajectory_store = TrajectoryBuilder.for_sde_steps(sde_indices, num_inference_steps)
        trajectory_store.add(0, latents.clone().to(dtype=self.trajectory_dtype))
        all_log_probs: Dict[int, torch.Tensor] = {}

        autocast_ctx = (
            torch.autocast("cuda", self.autocast_dtype)
            if device.type == "cuda" and self.autocast_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )

        self.model.eval()
        for i in range(num_inference_steps):
            sigma = sigma_schedule[i].to(device)
            sigma_next = sigma_schedule[i + 1].to(device)
            forward_context = WAN21ForwardContext(
                guidance_scale=actual_guidance,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                encoder_hidden_states_image=encoder_hidden_states_image,
                negative_encoder_hidden_states_image=negative_encoder_hidden_states_image,
                image_conditioning_latents=image_conditioning_latents,
                first_frame_mask=first_frame_mask,
            )

            with torch.no_grad():
                with autocast_ctx:
                    if self.model_bundle is None:
                        raise RuntimeError("WAN sampler requires model_bundle for model-specific forward dispatch")
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
            latents = latents.to(dtype=self.trajectory_dtype)
            trajectory_store.add(i + 1, latents)
            if log_prob is not None:
                all_log_probs[i] = log_prob.to(dtype=self.logprob_dtype)

        final_context = WAN21ForwardContext(
            guidance_scale=actual_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            encoder_hidden_states_image=encoder_hidden_states_image,
            negative_encoder_hidden_states_image=negative_encoder_hidden_states_image,
            image_conditioning_latents=image_conditioning_latents,
            first_frame_mask=first_frame_mask,
        )

        return RolloutSamples(
            latents=latents.to(torch.float32),
            timesteps=sigma_schedule,
            trajectories=trajectory_store.finalize(),
            log_probs=LogProbData.from_dict(all_log_probs),
            forward_context=final_context,
            step_indices=torch.arange(sigma_schedule.shape[0], device=sigma_schedule.device, dtype=torch.long),
        )


__all__ = ["FSDPWanSampler"]
