"""
SD3 Image Sampler for GRPO Training (FSDP Engine).

This sampler implements SDE sampling with log probability computation
for Stable Diffusion 3 image models using native PyTorch (FSDP-compatible).
It supports:
- Standard SDE sampling (sde)
- CPS sampling (cps) - recommended for flow_grpo
- Mixed ODE/SDE sampling for MixGRPO

Reference:
- flow_grpo: flow_grpo/diffusers_patch/sd3_sde_with_logprob.py
- DiffusionNFT: DiffusionNFT SD3 implementation
"""

import logging
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Dict, List, Optional, Set, Any, Tuple
import torch
import torch.nn as nn

from ..base import BaseSampler, RolloutOutput
from ..log_prob import compute_sde_log_prob, sde_step_with_log_prob, get_sigma_schedule
from diffusionrl.types import LogProbData, PromptEmbeddings

logger = logging.getLogger(__name__)


@dataclass
class _DPMState:
    order: int
    model_outputs: List[Optional[torch.Tensor]] = None
    lower_order_nums: int = 0

    def __post_init__(self) -> None:
        self.model_outputs = [None] * self.order

    def update(self, model_output: torch.Tensor) -> None:
        for i in range(self.order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

    def update_lower_order(self) -> None:
        if self.lower_order_nums < self.order:
            self.lower_order_nums += 1


def _convert_model_output(model_output: torch.Tensor, sample: torch.Tensor, sigmas: torch.Tensor, step_index: int) -> torch.Tensor:
    sigma_t = sigmas[step_index]
    x0_pred = sample - sigma_t * model_output
    return x0_pred


def _sigma_to_alpha_sigma_t(sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    alpha_t = 1 - sigma
    sigma_t = sigma
    return alpha_t, sigma_t


def _dpm_solver_first_order_update(
    model_output: torch.Tensor,
    sigmas: torch.Tensor,
    step_index: int,
    sample: torch.Tensor,
) -> torch.Tensor:
    sigma_t, sigma_s = sigmas[step_index + 1], sigmas[step_index]
    alpha_t, sigma_t = _sigma_to_alpha_sigma_t(sigma_t)
    alpha_s, sigma_s = _sigma_to_alpha_sigma_t(sigma_s)
    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s = torch.log(alpha_s) - torch.log(sigma_s)

    h = lambda_t - lambda_s
    x_t = (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * model_output
    return x_t


def _multistep_dpm_solver_second_order_update(
    model_output_list: List[torch.Tensor],
    sigmas: torch.Tensor,
    step_index: int,
    sample: torch.Tensor,
) -> torch.Tensor:
    sigma_t, sigma_s0, sigma_s1 = (
        sigmas[step_index + 1],
        sigmas[step_index],
        sigmas[step_index - 1],
    )

    alpha_t, sigma_t = _sigma_to_alpha_sigma_t(sigma_t)
    alpha_s0, sigma_s0 = _sigma_to_alpha_sigma_t(sigma_s0)
    alpha_s1, sigma_s1 = _sigma_to_alpha_sigma_t(sigma_s1)

    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
    lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)

    m0, m1 = model_output_list[-1], model_output_list[-2]

    h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
    r0 = h_0 / h
    D0, D1 = m0, (1.0 / r0) * (m0 - m1)

    x_t = (
        (sigma_t / sigma_s0) * sample
        - (alpha_t * (torch.exp(-h) - 1.0)) * D0
        - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
    )
    return x_t


def _dpm_step(
    order: int,
    model_output: torch.Tensor,
    sample: torch.Tensor,
    step_index: int,
    timesteps: torch.Tensor,
    sigmas: torch.Tensor,
    dpm_state: _DPMState,
) -> torch.Tensor:
    lower_order_final = step_index == len(timesteps) - 1
    lower_order_second = (step_index == len(timesteps) - 2) and len(timesteps) < 15

    model_output = _convert_model_output(model_output, sample, sigmas, step_index=step_index)
    dpm_state.update(model_output)

    sample = sample.to(torch.float32)

    if order == 1 or dpm_state.lower_order_nums < 1 or lower_order_final:
        if step_index == 0 or lower_order_final:
            # DDIM update with eta=0
            t, s = sigmas[step_index + 1], sigmas[step_index]
            noise_pred = (sample - (1 - s) * model_output) / s
            prev_mean = (1 - t) * model_output + torch.sqrt(t**2) * noise_pred
            prev_sample = prev_mean
        else:
            prev_sample = _dpm_solver_first_order_update(model_output, sigmas.to(torch.float64), step_index, sample)
    elif order == 2 or dpm_state.lower_order_nums < 2 or lower_order_second:
        prev_sample = _multistep_dpm_solver_second_order_update(
            dpm_state.model_outputs,
            sigmas.to(torch.float64),
            step_index,
            sample,
        )
    else:
        raise ValueError(f"Unsupported DPM order: {order}")

    dpm_state.update_lower_order()
    return prev_sample.to(model_output.dtype)


class SD3Sampler(BaseSampler):
    """
    SD3 image sampler with log probability computation (FSDP Engine).

    This sampler is designed for Stable Diffusion 3 models and implements:
    - Standard SDE formulation
    - CPS (Coefficient-Preserving Sampling) - recommended for flow_grpo
    - Mixed ODE/SDE sampling for MixGRPO

    Example:
        sampler = SD3Sampler(
            model=sd3_model.transformer,
            text_encoder=sd3_model.text_encoder,
            sde_type="cps",
            eta=0.7,
            shift=3.0,
        )
        output = sampler.sample(
            prompts=["A beautiful sunset"],
            num_inference_steps=28,
            guidance_scale=7.0,
        )
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        text_encoder: Optional[Any] = None,
        text_encoder_2: Optional[Any] = None,
        text_encoder_3: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        tokenizer_2: Optional[Any] = None,
        tokenizer_3: Optional[Any] = None,
        vae: Optional[nn.Module] = None,
        scheduler: Optional[Any] = None,
        eta: float = 0.7,
        sde_type: str = "cps",  # SD3 typically uses "cps" or "sde"
        shift: float = 3.0,    # SD3 uses shift=3.0
        latent_channels: int = 16,  # SD3 uses 16 latent channels
        vae_scale_factor: int = 8,  # VAE 8x compression
    ):
        """
        Initialize SD3 sampler.

        Args:
            model: SD3 transformer model
            text_encoder: CLIP text encoder 1
            text_encoder_2: CLIP text encoder 2
            text_encoder_3: T5 text encoder
            tokenizer: CLIP tokenizer 1
            tokenizer_2: CLIP tokenizer 2
            tokenizer_3: T5 tokenizer
            vae: VAE for encoding/decoding
            eta: Noise level for SDE (controls stochasticity)
            sde_type: SDE formulation ("sde", "cps")
            shift: Time shift parameter (SD3 uses 3.0)
            latent_channels: Number of latent channels (16 for SD3)
            vae_scale_factor: VAE spatial compression factor
        """
        super().__init__(eta=eta, sde_type=sde_type, shift=shift)
        self.model = model
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.text_encoder_3 = text_encoder_3
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.tokenizer_3 = tokenizer_3
        self.vae = vae
        self.scheduler = scheduler
        self.latent_channels = latent_channels
        self.vae_scale_factor = vae_scale_factor

    @property
    def requires_extra_forward_for_log_prob(self) -> bool:
        return False  # Log prob computed during sampling

    @property
    def supports_image(self) -> bool:
        return True

    @property
    def supports_video(self) -> bool:
        return False

    def _predict_noise_with_cfg(
        self,
        *,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor],
        guidance_scale: float,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict noise with SD3 CFG using a single batched forward.

        Matches flow_grpo/DiffusionNFT SD3 behavior:
        - concat [uncond, cond] embeddings
        - concat [latents, latents]
        - single model forward
        - chunk(2) and apply CFG formula
        """
        if guidance_scale > 1.0:
            uncond_prompt_embeds = (
                negative_prompt_embeds
                if negative_prompt_embeds is not None
                else torch.zeros_like(prompt_embeds)
            )
            if pooled_prompt_embeds is not None:
                uncond_pooled_embeds = (
                    negative_pooled_prompt_embeds
                    if negative_pooled_prompt_embeds is not None
                    else torch.zeros_like(pooled_prompt_embeds)
                )
                pooled_batched = torch.cat(
                    [uncond_pooled_embeds, pooled_prompt_embeds], dim=0
                )
            else:
                pooled_batched = None

            noise_pred = self.model(
                hidden_states=torch.cat([latents, latents], dim=0),
                encoder_hidden_states=torch.cat(
                    [uncond_prompt_embeds, prompt_embeds], dim=0
                ),
                timestep=torch.cat([timestep, timestep], dim=0),
                pooled_projections=pooled_batched,
                return_dict=False,
            )[0]
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
            return noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

        return self.model(
            hidden_states=latents,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]

    def sample(
        self,
        prompts: Optional[List[str]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,  # Not used by SD3 but kept for API compatibility
        num_inference_steps: int = 28,
        guidance_scale: float = 7.0,
        height: int = 1024,
        width: int = 1024,
        latents: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        sde_indices: Optional[Set[int]] = None,
        max_sequence_length: int = 256,
        init_same_noise: bool = False,
        num_samples_per_prompt: int = 1,
        **kwargs,
    ) -> RolloutOutput:
        """
        Execute SDE sampling and return trajectories with log probabilities.

        Args:
            prompts: List of text prompts (used if prompt_embeds not provided)
            prompt_embeds: Pre-computed prompt embeddings [B, seq, hidden]
            pooled_prompt_embeds: Pooled prompt embeddings [B, hidden]
            text_ids: Not used by SD3 (kept for API compatibility)
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG guidance scale
            height: Output image height
            width: Output image width
            latents: Initial latents (if None, sampled from noise)
            generator: Random number generator
            sde_indices: Set of timestep indices to use SDE sampling.
                If None, all timesteps use SDE.
            max_sequence_length: Maximum T5 sequence length
            init_same_noise: Share initial noise across K samples for same prompt (DanceGRPO/MixGRPO)
            num_samples_per_prompt: Number of samples per prompt (for init_same_noise)

        Returns:
            RolloutOutput with trajectories, log_probs, etc.
        """
        if self.model is None:
            raise RuntimeError("Model not set. Initialize sampler with model parameter.")

        device = next(self.model.parameters()).device
        # For SD3 with PEFT, always use bfloat16 for inference to match base model
        # LoRA adapters are float32 but base model is bfloat16
        dtype = torch.bfloat16

        # Encode prompts if needed
        if prompt_embeds is None:
            if prompts is None:
                raise ValueError("Either prompts or prompt_embeds must be provided")
            prompt_embeds, pooled_prompt_embeds = self._encode_prompt(
                prompts, max_sequence_length, device, dtype
            )
            if guidance_scale > 1.0 and negative_prompt_embeds is None:
                negative_prompt_embeds, negative_pooled_prompt_embeds = self._encode_prompt(
                    [""] * len(prompts), max_sequence_length, device, dtype
                )

        batch_size = prompt_embeds.shape[0]

        # Move embeddings to device
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        if pooled_prompt_embeds is not None:
            pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
        if negative_pooled_prompt_embeds is not None:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device=device, dtype=dtype)
        if guidance_scale > 1.0 and negative_prompt_embeds is None:
            logger.warning(
                "SD3 CFG: negative_prompt_embeds not provided. "
                "Falling back to zero embeddings."
            )

        # Calculate latent dimensions
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor

        # Initialize latents (with optional shared noise for DanceGRPO/MixGRPO)
        if latents is None:
            from ..noise_utils import generate_latents
            latents = generate_latents(
                batch_size=batch_size,
                latent_shape=(self.latent_channels, latent_height, latent_width),
                device=device,
                dtype=dtype,
                generator=generator,
                init_same_noise=init_same_noise,
                num_samples_per_prompt=num_samples_per_prompt,
            )
        else:
            latents = latents.to(device=device, dtype=dtype)

        # Get sigma schedule (align with diffusers scheduler if available)
        sigmas = self._get_sigma_schedule(num_inference_steps, device)

        # Default: all timesteps use SDE; in deterministic (dpm2) mode, use ODE only.
        if sde_indices is None:
            if self.sde_type == "dpm2":
                sde_indices = set()
            else:
                sde_indices = set(range(num_inference_steps))

        # Storage for trajectory and log probs
        trajectory = [latents.clone()]
        log_probs_dict: Dict[int, torch.Tensor] = {}

        # Denoising loop
        dpm_state: Optional[_DPMState] = None
        if self.sde_type == "dpm2":
            dpm_state = _DPMState(order=2)
            timesteps = sigmas[:-1]
        for i in range(num_inference_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            # Create timestep tensor (SD3 uses 0-1000 range)
            timestep = (sigma * 1000).expand(batch_size).to(dtype=dtype)

            # Forward pass with CFG
            with torch.no_grad():
                noise_pred = self._predict_noise_with_cfg(
                    latents=latents,
                    timestep=timestep,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    guidance_scale=guidance_scale,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                )

            # Check if this step uses SDE
            if self.sde_type == "dpm2":
                latents = _dpm_step(
                    order=2,
                    model_output=noise_pred.float(),
                    sample=latents.float(),
                    step_index=i,
                    timesteps=timesteps,
                    sigmas=sigmas,
                    dpm_state=dpm_state,
                )
                latents = latents.to(dtype=dtype)
            elif i in sde_indices:
                # SDE step with log probability
                latents, log_prob, _ = sde_step_with_log_prob(
                    noise_pred=noise_pred,
                    sample=latents,
                    sigmas=sigmas,
                    step_index=i,
                    eta=self.eta,
                    generator=generator,
                    sde_type=self.sde_type,
                )
                # Cast back to bfloat16 (sde_step_with_log_prob uses float32 internally)
                latents = latents.to(dtype=dtype)
                log_probs_dict[i] = log_prob
            else:
                # Deterministic ODE step (no log_prob)
                dt = sigma_next - sigma
                latents = latents + dt * noise_pred
                latents = latents.to(dtype=dtype)

            trajectory.append(latents.clone())

        # Stack trajectory
        trajectories = torch.stack(trajectory, dim=1)  # [B, T+1, C, H, W]

        # Create embeddings bundle
        embeddings = PromptEmbeddings(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )

        return RolloutOutput(
            latents=latents,
            timesteps=sigmas,
            trajectories=trajectories,
            log_probs=LogProbData.from_dict(log_probs_dict),
            embeddings=embeddings,
            metadata={
                "sde_indices": sde_indices,
                "engine_capabilities": {
                    "supports_logprob": True,
                    "supports_trajectory": True,
                    "supports_prompt_embeddings": True,
                },
                "trajectory_format": "dense_latent",
                "timestep_type": "sigma",
                "timestep_scale": 1.0,
                "height": height,
                "width": width,
                "guidance_scale": guidance_scale,
            },
            step_indices=torch.arange(sigmas.shape[0], device=sigmas.device, dtype=torch.long),
        )

    def compute_log_prob_for_training(
        self,
        latents: torch.Tensor,
        prev_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor],
        timestep_index: int,
        sigma_schedule: torch.Tensor,
        guidance_scale: Optional[float] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute one-step log probability from replayed SD3 trajectory."""
        if self.model is None:
            raise RuntimeError("Model not set. Initialize sampler with model parameter.")

        device = latents.device
        dtype = latents.dtype
        batch_size = latents.shape[0]
        actual_guidance = float(guidance_scale if guidance_scale is not None else 1.0)

        sigma_schedule = sigma_schedule.to(device=device, dtype=torch.float32)
        if timestep_index < 0 or timestep_index >= int(sigma_schedule.shape[0]) - 1:
            raise ValueError(
                "timestep_index out of range for sigma_schedule: "
                f"index={timestep_index}, len={sigma_schedule.shape[0]}"
            )

        sigma = sigma_schedule[timestep_index]
        sigma_next = sigma_schedule[timestep_index + 1]
        timestep = (sigma * 1000).expand(batch_size).to(dtype=dtype)

        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        pooled_prompt_embeds = (
            pooled_prompt_embeds.to(device=device, dtype=dtype)
            if pooled_prompt_embeds is not None
            else None
        )
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
        if negative_pooled_prompt_embeds is not None:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(
                device=device, dtype=dtype
            )

        autocast_ctx = (
            torch.autocast("cuda", torch.bfloat16)
            if latents.is_cuda
            else nullcontext()
        )
        self.model.train()
        with autocast_ctx:
            noise_pred = self._predict_noise_with_cfg(
                latents=latents,
                timestep=timestep,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                guidance_scale=actual_guidance,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            )

        log_prob, _ = compute_sde_log_prob(
            noise_pred=noise_pred,
            sample=latents,
            prev_sample=prev_latents,
            sigma=sigma,
            sigma_next=sigma_next,
            eta=self.eta,
            sde_type=self.sde_type,
            sigma_max=(
                float(sigma_schedule[1].item())
                if int(sigma_schedule.shape[0]) > 1
                else 1.0
            ),
        )
        return log_prob

    def _get_sigma_schedule(
        self,
        num_inference_steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.scheduler is None:
            return get_sigma_schedule(num_inference_steps, self.shift, device)
        try:
            from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
            retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas=None)
            sigmas = self.scheduler.sigmas
            if sigmas is None:
                raise ValueError("Scheduler did not provide sigmas")
            return sigmas.float().to(device)
        except Exception as e:
            raise RuntimeError(
                "Failed to use diffusers scheduler sigmas. "
                "Consider fixing the scheduler setup or explicitly switching "
                "to the internal sd3_time_shift schedule."
            ) from e

    def _encode_prompt(
        self,
        prompts: List[str],
        max_sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple:
        """Encode prompts using SD3's triple text encoder setup."""
        batch_size = len(prompts)

        # Check if we have all encoders
        if self.text_encoder is None or self.text_encoder_2 is None or self.text_encoder_3 is None:
            raise ValueError("All three text encoders required for SD3")
        if self.tokenizer is None or self.tokenizer_2 is None or self.tokenizer_3 is None:
            raise ValueError("All three tokenizers required for SD3")

        # CLIP 1 encoding
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)

        with torch.no_grad():
            clip_output_1 = self.text_encoder(
                text_input_ids,
                output_hidden_states=True,
            )
            clip_embeds_1 = clip_output_1.hidden_states[-2]
            pooled_1 = clip_output_1.text_embeds

        # CLIP 2 encoding
        text_inputs_2 = self.tokenizer_2(
            prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids_2 = text_inputs_2.input_ids.to(device)

        with torch.no_grad():
            clip_output_2 = self.text_encoder_2(
                text_input_ids_2,
                output_hidden_states=True,
            )
            clip_embeds_2 = clip_output_2.hidden_states[-2]
            pooled_2 = clip_output_2.text_embeds

        # Concatenate CLIP embeddings
        clip_embeds = torch.cat([clip_embeds_1, clip_embeds_2], dim=-1)
        pooled_embeds = torch.cat([pooled_1, pooled_2], dim=-1)

        # T5 encoding
        text_inputs_3 = self.tokenizer_3(
            prompts,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids_3 = text_inputs_3.input_ids.to(device)

        with torch.no_grad():
            t5_output = self.text_encoder_3(text_input_ids_3)
            t5_embeds = t5_output.last_hidden_state

        # SD3 transformer expects:
        # - encoder_hidden_states: T5 embeddings only [B, seq_len, 4096]
        # - pooled_projections: CLIP pooled embeddings [B, 2048]
        prompt_embeds = t5_embeds

        return prompt_embeds.to(dtype=dtype), pooled_embeds.to(dtype=dtype)
