"""WAN 2.2 FSDP-native sampler with dual-transformer boundary switching."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Set

import torch
import torch.nn as nn

from diffusionrl.sde.kernels import DanceSDEStrategy, StepStrategy
from diffusionrl.sde.runtime import denoising_step, sd3_time_shift
from diffusionrl.types.forward_context import WAN22ForwardContext
from diffusionrl.types.sample import LogProbData
from diffusionrl.types.trajectory_store import TrajectoryBuilder

from ..base import RolloutSamples
from .wan_sampler import FSDPWanSampler

logger = logging.getLogger(__name__)


class FSDPWan22Sampler(FSDPWanSampler):
    """WAN 2.2 sampler with dual-transformer boundary-based switching.

    Extends FSDPWanSampler to pass ``guidance_scale_2`` into ``WAN22ForwardContext``.
    ``boundary_ratio`` lives on ``WAN22ModelBundle`` (config / bundle state); routing
    uses it inside ``WAN22ModelBundle.forward_denoiser()`` / ``_select_guidance_for_sigma``.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        text_encoder: Optional[Any] = None,
        vae: Optional[nn.Module] = None,
        eta: float = 1.0,
        strategy: Optional[StepStrategy] = None,
        shift: float = 5.0,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
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
            guidance_scale=guidance_scale,
            autocast_precision=autocast_precision,
            trajectory_precision=trajectory_precision,
            logprob_precision=logprob_precision,
            **kwargs,
        )
        self.guidance_scale_2 = guidance_scale_2

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
        guidance_scale_2: Optional[float] = None,
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
        if self.model_bundle is None:
            raise RuntimeError("WAN22 sampler requires model_bundle for forward dispatch")

        device = self._resolve_runtime_device(prompt_embeds=prompt_embeds, latents=latents)
        prompt_embeds, negative_prompt_embeds = self._encode_prompts_if_needed(
            prompts,
            prompt_embeds,
            negative_prompt_embeds,
            device,
        )
        batch_size = int(prompt_embeds.shape[0])
        actual_guidance = float(guidance_scale) if guidance_scale is not None else self.default_guidance_scale
        actual_guidance_2 = guidance_scale_2 if guidance_scale_2 is not None else self.guidance_scale_2

        # Cast conditioning tensors to device/dtype once
        cast_dtype = self.autocast_dtype
        prompt_embeds = prompt_embeds.to(device=device, dtype=cast_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=cast_dtype)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = encoder_hidden_states_image.to(device=device, dtype=cast_dtype)
        if negative_encoder_hidden_states_image is not None:
            negative_encoder_hidden_states_image = negative_encoder_hidden_states_image.to(
                device=device,
                dtype=cast_dtype,
            )
        if image_conditioning_latents is not None:
            image_conditioning_latents = image_conditioning_latents.to(device=device, dtype=cast_dtype)
        if first_frame_mask is not None:
            first_frame_mask = first_frame_mask.to(device=device, dtype=cast_dtype)

        latent_t, latent_h, latent_w = self._latent_shape(height=height, width=width, num_frames=num_frames)
        latent_channels = self._latent_channels()
        if latents is None:
            from ..utils.noise import generate_latents

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

        # Hoist loop-invariant values
        sigma_max = sigma_schedule[1].item()

        autocast_ctx = (
            torch.autocast("cuda", cast_dtype)
            if device.type == "cuda" and cast_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )

        # Build forward context once (all fields are loop-invariant)
        forward_context = WAN22ForwardContext(
            guidance_scale=actual_guidance,
            guidance_scale_2=actual_guidance_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            encoder_hidden_states_image=encoder_hidden_states_image,
            negative_encoder_hidden_states_image=negative_encoder_hidden_states_image,
            image_conditioning_latents=image_conditioning_latents,
            first_frame_mask=first_frame_mask,
        )

        self.model.eval()
        with torch.no_grad():
            for i in range(num_inference_steps):
                sigma = sigma_schedule[i]
                sigma_next = sigma_schedule[i + 1]

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
                    sigma_max=sigma_max,
                    strategy=strategy,
                    step_index=i,
                )
                latents = latents.to(dtype=self.trajectory_dtype)
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


__all__ = ["FSDPWan22Sampler"]
