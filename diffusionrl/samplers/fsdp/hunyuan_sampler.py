"""
HunyuanVideo FSDP Sampler for GRPO Training.

This sampler implements SDE sampling with log probability computation
for HunyuanVideo models using native PyTorch (FSDP-compatible).
It is fully aligned with DanceGRPO's implementation.

Reference:
- DanceGRPO: fastvideo/train_grpo_hunyuan.py (lines 54-96, 104-143)

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
from typing import Dict, List, Optional, Set, Tuple, Any
import torch
import torch.nn as nn

from ..base import BaseSampler, RolloutOutput
from ..log_prob import get_sigma_schedule, sd3_time_shift, flux_sde_step
from diffusionrl.types import LogProbData, PromptEmbeddings

logger = logging.getLogger(__name__)


class FSDPHunyuanSampler(BaseSampler):
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
            use_sde_solver=True,
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
        use_sde_solver: bool = True,  # Enable SDE solver correction
        guidance_scale: float = DEFAULT_GUIDANCE_VALUE,
    ):
        """
        Initialize HunyuanVideo FSDP sampler.

        Args:
            model: HunyuanVideo transformer model
            text_encoder: Reserved for constructor compatibility with the
                shared sampler factory (unused in this sampler).
            vae: VAE for decoding latents to video
            eta: Noise level for SDE (controls stochasticity)
            sde_type: SDE formulation (use "dance" for DanceGRPO alignment)
            shift: Time shift parameter for sigma schedule
            use_sde_solver: Enable SDE solver correction (DanceGRPO line 76-79)
            guidance_scale: Default guidance scale when request does not override.
        """
        super().__init__(eta=eta, sde_type=sde_type, shift=shift)
        self.model = model
        self.text_encoder = text_encoder
        self.vae = vae
        self.use_sde_solver = use_sde_solver
        self.default_guidance_scale = float(guidance_scale)

    @property
    def requires_extra_forward_for_log_prob(self) -> bool:
        return False  # Log prob computed during sampling

    @property
    def supports_image(self) -> bool:
        return False

    @property
    def supports_video(self) -> bool:
        return True

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
        generator: Optional[torch.Generator] = None,
        sde_indices: Optional[Set[int]] = None,
        **kwargs,
    ) -> RolloutOutput:
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
            RolloutOutput with trajectories, log_probs, etc.
        """
        if self.model is None:
            raise RuntimeError("Model not set. Initialize sampler with model parameter.")

        if prompt_embeds is None:
            raise ValueError("prompt_embeds (encoder_hidden_states) is required for HunyuanVideo")

        device = next(self.model.parameters()).device
        batch_size = prompt_embeds.shape[0]

        # Move embeddings to device
        prompt_embeds = prompt_embeds.to(device=device)
        if pooled_prompt_embeds is not None:
            pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=torch.bfloat16)
        else:
            # Newer diffusers Hunyuan forward requires pooled_projections.
            # pooled_prompt_embeds stays bfloat16 for model forward (not trajectory storage).
            proj_dim = getattr(getattr(self.model, "config", None), "pooled_projection_dim", 768)
            pooled_prompt_embeds = torch.zeros(
                batch_size,
                int(proj_dim),
                device=device,
                dtype=torch.bfloat16,
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
        trajectory_dtype = torch.float16

        if latents is None:
            latents = torch.randn(
                batch_size,
                self.IN_CHANNELS,
                latent_t,
                latent_h,
                latent_w,
                device=device,
                dtype=trajectory_dtype,
                generator=generator,
            )
        else:
            latents = latents.to(device=device, dtype=trajectory_dtype)

        # Get sigma schedule (DanceGRPO line 188-190)
        sigma_schedule = torch.linspace(1, 0, num_inference_steps + 1)
        sigma_schedule = sd3_time_shift(self.shift, sigma_schedule)
        sigma_schedule = sigma_schedule.to(device)

        # Default: all timesteps use SDE
        if sde_indices is None:
            sde_indices = set(range(num_inference_steps))
        actual_guidance = (
            float(guidance_scale)
            if guidance_scale is not None
            else float(self.default_guidance_scale)
        )

        # Storage for trajectory and log probs (DanceGRPO line 115-116)
        all_latents = [latents.clone()]
        all_log_probs: Dict[int, torch.Tensor] = {}

        # Denoising loop (DanceGRPO line 117-139)
        for i in range(num_inference_steps):
            sigma = sigma_schedule[i]
            # Keep timestep in float32 to avoid precision loss (no int truncation)
            timestep = (sigma.float() * 1000).expand(batch_size)

            # Forward pass (DanceGRPO line 123-135)
            self.model.eval()
            with torch.no_grad():
                with torch.autocast("cuda", torch.bfloat16):
                    model_pred = self.model(
                        hidden_states=latents,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        timestep=timestep,
                        guidance=torch.tensor(
                            [actual_guidance],
                            device=device,
                            dtype=torch.bfloat16
                        ),
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]

            # Check if this step uses SDE
            if i in sde_indices:
                # SDE step with log probability (DanceGRPO line 136)
                # output_dtype=float16: align log_prob with trajectory storage precision
                # float16 has more mantissa bits than bfloat16
                latents, pred_original, log_prob = flux_sde_step(
                    model_pred,
                    latents.to(torch.float32),
                    sigma_schedule,
                    index=i,
                    eta=self.eta,
                    use_sde_solver=self.use_sde_solver,
                    output_dtype=trajectory_dtype,
                )
                latents = latents.to(trajectory_dtype)
            else:
                # ODE step (deterministic)
                dsigma = sigma_schedule[i + 1] - sigma
                latents = latents + dsigma * model_pred

            all_latents.append(latents.clone())

        # Final latent normalization (DanceGRPO line 140)
        final_latents = pred_original.to(torch.float32) / self.LATENT_SCALE

        # Stack trajectory (DanceGRPO line 141)
        # Under FSDP CPU offload, some intermediates can temporarily reside on CPU.
        # Normalize trajectory tensors to the final latent device before stacking.
        target_device = latents.device
        if any(t.device != target_device for t in all_latents):
            all_latents = [t.to(device=target_device) for t in all_latents]
        trajectories = torch.stack(all_latents, dim=1)  # [B, T+1, C, t, H, W]

        # Create embeddings bundle
        embeddings = PromptEmbeddings(
            prompt_embeds=prompt_embeds,
            encoder_attention_mask=encoder_attention_mask,
        )

        return RolloutOutput(
            latents=final_latents,
            timesteps=sigma_schedule,
            trajectories=trajectories,
            log_probs=LogProbData.from_dict(all_log_probs),
            embeddings=embeddings,
            metadata={
                "sde_indices": sde_indices,
                "engine_capabilities": {
                    "supports_logprob": True,
                    "supports_trajectory": True,
                    "supports_prompt_embeddings": True,
                    "supports_guidance_scale": True,
                },
                "trajectory_format": "video_dense_latent",
                "timestep_type": "sigma",
                "timestep_scale": 1.0,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "guidance_scale": float(actual_guidance),
                "use_sde_solver": self.use_sde_solver,
                "latent_scale": self.LATENT_SCALE,
            },
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
        actual_guidance = (
            float(guidance_scale)
            if guidance_scale is not None
            else float(self.default_guidance_scale)
        )
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
            dtype=torch.bfloat16,
        )

        # Forward pass with gradients (DanceGRPO line 158-171)
        self.model.train()
        with torch.autocast("cuda", torch.bfloat16):
            model_pred = self.model(
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                timestep=timestep,
                guidance=torch.tensor(
                    [actual_guidance],
                    device=device,
                    dtype=torch.bfloat16
                ),
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]

        # Compute log probability (DanceGRPO line 172)
        # Training side: prev_latents is float16 from trajectory.
        # Upcast to float32 so that compute is on the same truncated value.
        _, _, log_prob = flux_sde_step(
            model_pred,
            latents.to(torch.float32),
            sigma_schedule,
            index=timestep_index,
            eta=self.eta,
            use_sde_solver=self.use_sde_solver,
            prev_sample=prev_latents.to(torch.float32),
        )

        return log_prob
