"""
HunyuanVideo FSDP Sampler for GRPO Training.

This sampler implements SDE sampling with log probability computation
for HunyuanVideo models using native PyTorch (FSDP-compatible).
It is fully aligned with DanceGRPO's implementation.

Reference:
- DanceGRPO Hunyuan training implementation (lines 54-96, 104-143)

Key alignment points with DanceGRPO:
- sd3_time_shift(): sigma schedule transformation (line 54-55)
- flux_step(): SDE step with log_prob computation (line 57-96)
- run_sample_step(): sampling loop (line 104-143)
- SDE solver correction term (line 76-79)
- Guidance default: 6018.0 (line 129; configurable via runtime args)
- Latent normalization: /0.476986 (line 140)
- Precision: bfloat16 for forward, float32 for SDE step, float16 for trajectory storage
"""

import logging
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Set

import torch
import torch.nn as nn

from diffusionrl.sde.registry import resolve_sde_strategy_class
from diffusionrl.sde.runtime import denoising_step, sd3_time_shift
from diffusionrl.types.forward_context import HunyuanForwardContext
from diffusionrl.types.sample import LogProbData
from diffusionrl.types.trajectory_store import TrajectoryBuilder

from ..base import RolloutSamples
from .base_sampler import FSDPBaseSampler

logger = logging.getLogger(__name__)


class FSDPHunyuanSampler(FSDPBaseSampler):
    """
    HunyuanVideo FSDP sampler with log probability computation.

    This sampler is fully aligned with DanceGRPO's implementation for
    HunyuanVideo models. It uses native PyTorch operations and is
    compatible with FSDP for distributed training.

    Key features:
    - SDE sampling with exact DanceGRPO formulation
    - SDE solver correction for improved sampling
    - bfloat16 forward pass with float32 SDE step for numerical stability
    - float16 trajectory storage for precision (natural images within float16 range)
    - Full trajectory and log probability tracking

    Example:
        sampler = FSDPHunyuanSampler(
            model=hunyuan_transformer,
            eta=1.0,
            shift=1.0,
        )
        output = sampler.sample(
            prompt_embeds=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            num_inference_steps=50,
        )
    """

    # DanceGRPO constants
    SPATIAL_DOWNSAMPLE = 8
    TEMPORAL_DOWNSAMPLE = 4
    IN_CHANNELS = 16
    DEFAULT_GUIDANCE_VALUE = 6018.0  # DanceGRPO line 129
    LATENT_SCALE = 0.476986  # DanceGRPO line 140

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        text_encoder: Optional[Any] = None,
        vae: Optional[nn.Module] = None,
        eta: float = 1.0,
        sde_type: str = "dance",  # Use DanceGRPO formulation
        shift: float = 1.0,  # DanceGRPO default
        guidance_scale: float = DEFAULT_GUIDANCE_VALUE,
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
            sde_type=sde_type,
            shift=shift,
            autocast_precision=autocast_precision,
            trajectory_precision=trajectory_precision,
            logprob_precision=logprob_precision,
            **kwargs,
        )
        self.default_guidance_scale = float(guidance_scale)

    def sample(
        self,
        prompts: Optional[List[str]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = None,
        height: int = 480,
        width: int = 848,
        num_frames: int = 129,
        latents: Optional[torch.Tensor] = None,
        base_seed: Optional[int] = None,
        sde_indices: Optional[Set[int]] = None,
        init_same_noise: bool = False,
        samples_per_prompt: int = 1,
        noise_group_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> RolloutSamples:
        """
        Execute SDE sampling and return trajectories with log probabilities.

        Aligned with DanceGRPO run_sample_step (line 104-143):
        - Sigma schedule with time shift
        - Forward pass with configurable guidance (DanceGRPO default: 6018.0)
        - SDE step with log probability
        - Trajectory stacking

        Args:
            prompts: Not used (embeddings required)
            prompt_embeds: Pre-computed encoder hidden states [B, seq, hidden]
            pooled_prompt_embeds: Not used for HunyuanVideo
            encoder_attention_mask: Attention mask [B, seq]
            num_inference_steps: Number of denoising steps
            guidance_scale: Optional per-request guidance override.
            height: Video height
            width: Video width
            num_frames: Number of video frames
            latents: Initial latents (if None, sampled from noise)
            generator: Random number generator
            sde_indices: Set of timestep indices for SDE (all by default)

        Returns:
            RolloutSamples with trajectories, log_probs, etc.
        """
        if self.model is None:
            raise RuntimeError("Model not set. Initialize sampler with model parameter.")

        if prompt_embeds is None:
            raise ValueError("prompt_embeds (encoder_hidden_states) is required for HunyuanVideo")

        device = self._resolve_runtime_device(prompt_embeds=prompt_embeds, latents=latents)
        batch_size = prompt_embeds.shape[0]

        # Move embeddings to device
        prompt_embeds = prompt_embeds.to(device=device, dtype=self.autocast_dtype)
        if pooled_prompt_embeds is not None:
            pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=self.autocast_dtype)
        else:
            # Newer diffusers Hunyuan forward requires pooled_projections.
            # pooled_prompt_embeds stays in autocast precision for model forward.
            proj_dim = getattr(getattr(self.model, "config", None), "pooled_projection_dim", 768)
            pooled_prompt_embeds = torch.zeros(
                batch_size,
                int(proj_dim),
                device=device,
                dtype=self.autocast_dtype,
            )
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.to(device=device)
        else:
            # Recent diffusers Hunyuan forward expects a valid attention mask.
            encoder_attention_mask = torch.ones(
                batch_size,
                prompt_embeds.shape[1],
                device=device,
                dtype=torch.long,
            )

        # Calculate latent dimensions (DanceGRPO line 199-203)
        latent_t = ((num_frames - 1) // self.TEMPORAL_DOWNSAMPLE) + 1
        latent_h = height // self.SPATIAL_DOWNSAMPLE
        latent_w = width // self.SPATIAL_DOWNSAMPLE

        # Initialize latents (DanceGRPO line 223-229)
        # Use float16 for trajectory storage (more mantissa bits than bfloat16).
        # Natural image latents are within float16 range.
        # Model forward still uses bfloat16 via torch.autocast.
        trajectory_dtype = self.trajectory_dtype

        if latents is None:
            from ..noise_utils import generate_latents

            latents = generate_latents(
                batch_size=batch_size,
                latent_shape=(self.IN_CHANNELS, latent_t, latent_h, latent_w),
                device=device,
                dtype=trajectory_dtype,
                init_same_noise=init_same_noise,
                samples_per_prompt=samples_per_prompt,
                noise_group_ids=noise_group_ids,
                base_seed=base_seed,
            )
        else:
            latents = latents.to(device=device, dtype=trajectory_dtype)

        # Get sigma schedule (DanceGRPO line 188-190)
        sigma_schedule = torch.linspace(1, 0, num_inference_steps + 1)
        sigma_schedule = sd3_time_shift(self.shift, sigma_schedule)
        sigma_schedule = sigma_schedule.to(device)

        # Default: all timesteps use SDE; deterministic mode uses ODE only.
        if sde_indices is None:
            if self.uses_deterministic_solver:
                sde_indices = set()
            else:
                sde_indices = set(range(num_inference_steps))
        actual_guidance = float(guidance_scale) if guidance_scale is not None else float(self.default_guidance_scale)

        strategy = resolve_sde_strategy_class(self.sde_type)()
        strategy.init_schedule(sigma_schedule)

        # Storage for trajectory and log probs (DanceGRPO line 115-116)
        # Selective collection: only store positions needed for SDE step pairs
        trajectory_store = TrajectoryBuilder.for_sde_steps(sde_indices, num_inference_steps)
        trajectory_store.add(0, latents.clone().to(dtype=trajectory_dtype))
        all_log_probs: Dict[int, torch.Tensor] = {}

        autocast_ctx = (
            torch.autocast("cuda", self.autocast_dtype)
            if device.type == "cuda" and self.autocast_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )

        # Denoising loop (DanceGRPO line 117-139)
        for i in range(num_inference_steps):
            sigma = sigma_schedule[i].to(device)
            sigma_next = sigma_schedule[i + 1].to(device)
            # Keep timestep in float32 to avoid precision loss (no int truncation)
            timestep = (sigma.float() * 1000).expand(batch_size)

            # Forward pass (DanceGRPO line 123-135)
            self.model.eval()
            with torch.no_grad():
                with autocast_ctx:
                    model_pred = self.model(
                        hidden_states=latents,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        timestep=timestep,
                        guidance=torch.tensor([actual_guidance], device=device, dtype=self.autocast_dtype),
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]

            # Unified step: eta controls SDE vs ODE behaviour.
            step_eta = self.eta if i in sde_indices else 0.0
            latents, log_prob, prev_sample_mean = denoising_step(
                noise_pred=model_pred,
                sample=latents,
                sigma=sigma,
                sigma_next=sigma_next,
                eta=step_eta,
                sde_type=self.sde_type,
                sigma_max=sigma_schedule[1].item(),
                strategy=strategy,
                step_index=i,
            )
            latents = latents.to(dtype=self.trajectory_dtype)
            trajectory_store.add(i + 1, latents)

            if log_prob is not None:
                all_log_probs[i] = log_prob.to(dtype=self.logprob_dtype)

        # Final latent normalization (DanceGRPO line 140)
        final_latents = latents.to(torch.float32) / self.LATENT_SCALE

        trajectory = trajectory_store.finalize()

        forward_context = HunyuanForwardContext(
            guidance_scale=float(actual_guidance),
            prompt_embeds=prompt_embeds,
            encoder_attention_mask=encoder_attention_mask,
        )

        return RolloutSamples(
            latents=final_latents,
            timesteps=sigma_schedule,
            trajectories=trajectory,
            log_probs=LogProbData.from_dict(all_log_probs),
            forward_context=forward_context,
            step_indices=torch.arange(sigma_schedule.shape[0], device=sigma_schedule.device, dtype=torch.long),
        )

    def compute_log_prob_for_training(
        self,
        latents: torch.Tensor,
        prev_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor],
        timestep_index: int,
        sigma_schedule: torch.Tensor,
        guidance_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute log probability for a single training step.

        This is aligned with DanceGRPO's grpo_one_step (line 146-173).

        Args:
            latents: Current latents x_t
            prev_latents: Previous latents x_{t-1} (from sampling trajectory)
            prompt_embeds: Encoder hidden states
            encoder_attention_mask: Attention mask
            timestep_index: Index in sigma schedule
            sigma_schedule: Full sigma schedule

        Returns:
            log_prob: Log probability [B]
        """
        device = latents.device
        batch_size = latents.shape[0]
        actual_guidance = float(guidance_scale) if guidance_scale is not None else float(self.default_guidance_scale)
        prompt_embeds = prompt_embeds.to(device=device)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                batch_size,
                prompt_embeds.shape[1],
                device=device,
                dtype=torch.long,
            )
        else:
            encoder_attention_mask = encoder_attention_mask.to(device=device)

        sigma = sigma_schedule[timestep_index]
        # Keep timestep in float32 to avoid precision loss (no int truncation)
        timestep = (sigma.float() * 1000).expand(batch_size)

        proj_dim = getattr(getattr(self.model, "config", None), "pooled_projection_dim", 768)
        pooled_prompt_embeds = torch.zeros(
            batch_size,
            int(proj_dim),
            device=device,
            dtype=self.autocast_dtype,
        )

        # Forward pass with gradients (DanceGRPO line 158-171)
        self.model.train()
        autocast_ctx = (
            torch.autocast("cuda", self.autocast_dtype)
            if latents.is_cuda and self.autocast_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        with autocast_ctx:
            model_pred = self.model(
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                timestep=timestep,
                guidance=torch.tensor([actual_guidance], device=device, dtype=self.autocast_dtype),
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]

        # Compute log probability (DanceGRPO line 172)
        # Training side: prev_latents is float16 from trajectory.
        # Upcast to float32 so that compute is on the same truncated value.
        if self.uses_deterministic_solver:
            raise ValueError("Deterministic Hunyuan sampling does not define stochastic log-prob replay.")

        sigma = sigma_schedule[timestep_index].to(device)
        sigma_next = sigma_schedule[timestep_index + 1].to(device)
        _, log_prob, _ = denoising_step(
            noise_pred=model_pred,
            sample=latents,
            sigma=sigma,
            sigma_next=sigma_next,
            eta=self.eta,
            prev_sample=prev_latents,
            sde_type=self.sde_type,
        )

        return log_prob.to(dtype=self.logprob_dtype)
