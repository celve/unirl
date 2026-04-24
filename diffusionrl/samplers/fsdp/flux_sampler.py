"""
FLUX Image Sampler for GRPO Training (native direct sampling).

This sampler implements SDE sampling with log probability computation
for FLUX image models using native PyTorch (FSDP-compatible). It is
fully aligned with DanceGRPO's implementation.

Reference:
- DanceGRPO FLUX training implementation

Key alignment points with DanceGRPO:
- pack_latents(): 2x2 patch packing (line 191-196)
- unpack_latents(): Reverse unpacking (line 198-211)
- prepare_latent_image_ids(): Uses latent_h//2, latent_w//2 (line 352)
- text_ids handling: .repeat(seq_len, 1) (line 244)
- timesteps: /1000 normalization (line 238)
- guidance: 3.5 (line 239-243)
- flux_step(): SDE with log_prob (line 131-170)
"""

import logging
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Set

import torch
import torch.nn as nn

from diffusionrl.sde.registry import resolve_sde_strategy_class
from diffusionrl.sde.runtime import denoising_step, sd3_time_shift
from diffusionrl.types.forward_context import FluxForwardContext
from diffusionrl.types.sample import LogProbData
from diffusionrl.types.trajectory_store import TrajectoryBuilder

from ..base import RolloutSamples
from .base_sampler import FSDPBaseSampler

logger = logging.getLogger(__name__)


class FluxSampler(FSDPBaseSampler):
    """
    FLUX image sampler with log probability computation for native direct sampling.

    Fully aligned with DanceGRPO's train_grpo_flux.py implementation.

    Key features:
    - 2x2 patch packing/unpacking for FLUX transformer
    - SDE solver with score correction
    - bfloat16 forward + float32 SDE step
    - float16 trajectory storage for precision (natural images within float16 range)

    Example:
        sampler = FluxSampler(
            model=flux_model.transformer,
            text_encoder=flux_model.text_encoder,
            eta=0.7,
            shift=1.0,
        )
        output = sampler.sample(
            prompts=["A beautiful sunset"],
            num_inference_steps=28,
            guidance_scale=3.5,
        )
    """

    # FLUX constants
    SPATIAL_DOWNSAMPLE = 8  # VAE compression
    IN_CHANNELS = 16  # Latent channels
    DEFAULT_GUIDANCE = 3.5  # DanceGRPO line 239-243
    VAE_SCALE = 0.3611  # DanceGRPO line 381
    VAE_SHIFT = 0.1159  # DanceGRPO line 381

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        text_encoder: Optional[Any] = None,
        vae: Optional[nn.Module] = None,
        eta: float = 0.7,
        sde_type: str = "dance",  # FLUX uses DanceGRPO formulation by default
        shift: float = 1.0,  # FLUX uses shift=1.0
        guidance_scale: float = 3.5,  # Configurable guidance (DanceGRPO default: 3.5)
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
        self.guidance_scale = guidance_scale

    def _pack_latents(
        self,
        latents: torch.Tensor,
        batch_size: int,
        num_channels: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Pack latents using 2x2 patches for FLUX transformer.

        Aligned with DanceGRPO pack_latents (line 191-196):
            latents.view(B, C, H//2, 2, W//2, 2)
            → [B, (H/2)*(W/2), C*4]
        """
        latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)
        return latents

    def _unpack_latents(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Unpack latents from 2x2 patches back to spatial format.

        Aligned with DanceGRPO unpack_latents (line 198-211).
        """
        batch_size, num_patches, channels = latents.shape

        # Account for 2x2 packing
        packed_h = height // 2
        packed_w = width // 2

        latents = latents.view(batch_size, packed_h, packed_w, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // 4, height, width)

        return latents

    def _prepare_latent_image_ids(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Prepare latent image IDs for FLUX positional encoding.

        Aligned with DanceGRPO prepare_latent_image_ids (line 178-189).
        Note: DanceGRPO uses height//2, width//2 AFTER packing (line 352).
        """
        # Use packed dimensions (height and width are already latent_h//2, latent_w//2)
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_ids = latent_image_ids.reshape(height * width, 3)
        return latent_image_ids.to(device=device, dtype=dtype)

    def sample(
        self,
        prompts: Optional[List[str]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,
        num_inference_steps: int = 28,
        guidance_scale: Optional[float] = None,  # Override sampler's guidance_scale
        height: int = 1024,
        width: int = 1024,
        latents: Optional[torch.Tensor] = None,
        base_seed: Optional[int] = None,
        sde_indices: Optional[Set[int]] = None,
        init_same_noise: bool = False,
        samples_per_prompt: int = 1,
        noise_group_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> RolloutSamples:
        """
        Execute SDE sampling aligned with DanceGRPO run_sample_step.

        Args:
            prompts: List of text prompts (used if prompt_embeds not provided)
            prompt_embeds: Pre-computed prompt embeddings [B, seq, hidden]
            pooled_prompt_embeds: Pooled prompt embeddings [B, hidden]
            text_ids: Text position IDs [B, seq, 3]
            num_inference_steps: Number of denoising steps
            guidance_scale: Override sampler's guidance_scale (default: self.guidance_scale)
            height: Output image height
            width: Output image width
            latents: Initial latents (if None, sampled from noise)
            generator: Random number generator
            sde_indices: Set of timestep indices for SDE (all by default)
            init_same_noise: Share initial noise across K samples for same prompt (DanceGRPO/MixGRPO)
            samples_per_prompt: Number of samples per prompt (for init_same_noise)

        Returns:
            RolloutSamples with trajectories, log_probs, etc.
        """
        # Use provided guidance_scale or fall back to instance default
        actual_guidance = guidance_scale if guidance_scale is not None else self.guidance_scale
        if self.model is None:
            raise RuntimeError("Model not set. Initialize sampler with model parameter.")

        device = self._resolve_runtime_device(prompt_embeds=prompt_embeds, latents=latents)
        # Use float16 for trajectory storage (more mantissa bits than bfloat16).
        # Natural image latents are within float16 range.
        # Model forward still uses bfloat16 via torch.autocast.
        trajectory_dtype = self.trajectory_dtype

        # Encode prompts if needed
        if prompt_embeds is None:
            if prompts is None:
                raise ValueError("Either prompts or prompt_embeds must be provided")
            if self.text_encoder is None:
                raise ValueError("text_encoder required when prompts are provided")
            prompt_embeds, pooled_prompt_embeds = self.text_encoder.encode_prompt(prompts)

        batch_size = prompt_embeds.shape[0]

        # Move embeddings to device
        prompt_embeds = prompt_embeds.to(device=device, dtype=self.autocast_dtype)
        if pooled_prompt_embeds is not None:
            pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=self.autocast_dtype)

        # Calculate latent dimensions
        latent_h = height // self.SPATIAL_DOWNSAMPLE
        latent_w = width // self.SPATIAL_DOWNSAMPLE

        # Initialize latents with optional shared noise (DanceGRPO line 346-350)
        if latents is None:
            from ..noise_utils import generate_latents

            latents = generate_latents(
                batch_size=batch_size,
                latent_shape=(self.IN_CHANNELS, latent_h, latent_w),
                device=device,
                dtype=trajectory_dtype,
                init_same_noise=init_same_noise,
                samples_per_prompt=samples_per_prompt,
                noise_group_ids=noise_group_ids,
                base_seed=base_seed,
            )
        else:
            latents = latents.to(device=device, dtype=trajectory_dtype)

        # Pack latents for FLUX transformer (DanceGRPO line 351)
        packed_latents = self._pack_latents(latents, batch_size, self.IN_CHANNELS, latent_h, latent_w)

        # Prepare image IDs with packed dimensions (DanceGRPO line 352)
        # Note: uses latent_h//2, latent_w//2 because packing halves dimensions
        image_ids = self._prepare_latent_image_ids(
            batch_size, latent_h // 2, latent_w // 2, device, self.autocast_dtype
        )

        # Prepare text IDs (DanceGRPO line 244)
        if text_ids is None:
            seq_len = prompt_embeds.shape[1]
            text_ids = torch.zeros(batch_size, seq_len, 3, device=device, dtype=self.autocast_dtype)
        else:
            text_ids = text_ids.to(device=device, dtype=self.autocast_dtype)

        # Get sigma schedule (DanceGRPO line 311-313)
        sigma_schedule = torch.linspace(1, 0, num_inference_steps + 1, device=device)
        sigma_schedule = sd3_time_shift(self.shift, sigma_schedule).to(device=device, dtype=torch.float32)

        # Default: all timesteps use SDE. For deterministic (dpm2) or eta=0 mode, use ODE only.
        if sde_indices is None:
            if self.uses_deterministic_solver:
                sde_indices = set()  # pure ODE / deterministic steps
            else:
                sde_indices = set(range(num_inference_steps))

        strategy = resolve_sde_strategy_class(self.sde_type)()
        strategy.init_schedule(sigma_schedule)

        # Storage for trajectory and log probs (DanceGRPO line 226-227)
        latents = packed_latents.to(dtype=trajectory_dtype)
        # Selective collection: only store positions needed for SDE step pairs
        trajectory_store = TrajectoryBuilder.for_sde_steps(sde_indices, num_inference_steps)
        trajectory_store.add(0, latents)
        all_log_probs: Dict[int, torch.Tensor] = {}

        autocast_ctx = (
            torch.autocast("cuda", self.autocast_dtype)
            if device.type == "cuda" and self.autocast_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )

        # Denoising loop (DanceGRPO line 228-256)
        for i in range(num_inference_steps):
            sigma = sigma_schedule[i].to(device=device)
            sigma_next = sigma_schedule[i + 1].to(device=device)
            # Keep timestep in float32 to avoid precision loss (no int truncation)
            timestep = sigma.float().expand(batch_size)

            # Forward pass (DanceGRPO line 234-249)
            self.model.eval()
            with torch.no_grad():
                with autocast_ctx:
                    pred = self.model(
                        hidden_states=latents,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timestep,
                        guidance=torch.tensor([actual_guidance], device=device, dtype=self.autocast_dtype).expand(
                            batch_size
                        ),
                        txt_ids=text_ids[0],  # [seq, 3] - same for all batches
                        pooled_projections=pooled_prompt_embeds,
                        img_ids=image_ids,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]
                    if pred.device != latents.device:
                        pred = pred.to(device=latents.device)

            # Unified step: eta controls SDE vs ODE behaviour.
            step_eta = self.eta if i in sde_indices else 0.0
            latents, log_prob, prev_sample_mean = denoising_step(
                noise_pred=pred,
                sample=latents,
                sigma=sigma,
                sigma_next=sigma_next,
                eta=step_eta,
                sde_type=self.sde_type,
                sigma_max=sigma_schedule[1].item(),
                strategy=strategy,
                step_index=i,
            )
            latents = latents.to(dtype=trajectory_dtype)
            trajectory_store.add(i + 1, latents)

            if log_prob is not None:
                all_log_probs[i] = log_prob.to(dtype=self.logprob_dtype)

        # Unpack final latents for output
        final_latents = self._unpack_latents(latents, latent_h, latent_w)

        trajectory = trajectory_store.finalize()

        forward_context = FluxForwardContext(
            guidance_scale=actual_guidance,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids,
            image_ids=image_ids,
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
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        image_ids: torch.Tensor,
        timestep_index: int,
        sigma_schedule: torch.Tensor,
        guidance_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute log probability for a single training step.

        Aligned with DanceGRPO's grpo_one_step (line 260-292).

        Args:
            latents: Current packed latents x_t
            prev_latents: Previous packed latents x_{t-1}
            prompt_embeds: Encoder hidden states
            pooled_prompt_embeds: Pooled projections
            text_ids: Text position IDs
            image_ids: Image position IDs
            timestep_index: Index in sigma schedule
            sigma_schedule: Full sigma schedule
            guidance_scale: Override sampler's guidance_scale

        Returns:
            log_prob: Log probability [B]
        """
        device = latents.device
        batch_size = latents.shape[0]

        # Use provided guidance_scale or fall back to instance default
        actual_guidance = guidance_scale if guidance_scale is not None else self.guidance_scale

        sigma_schedule = sigma_schedule.to(device=device, dtype=torch.float32)
        sigma = sigma_schedule[timestep_index]
        # Keep timestep in float32 to avoid precision loss (no int truncation)
        timestep = sigma.float().expand(batch_size)

        # Forward pass with gradients (DanceGRPO line 274-290)
        self.model.train()
        autocast_ctx = (
            torch.autocast("cuda", self.autocast_dtype)
            if latents.is_cuda and self.autocast_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        with autocast_ctx:
            pred = self.model(
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                guidance=torch.tensor([actual_guidance], device=device, dtype=self.autocast_dtype).expand(batch_size),
                txt_ids=text_ids[0],  # [seq, 3] - same for all batches
                pooled_projections=pooled_prompt_embeds,
                img_ids=image_ids.squeeze(0) if image_ids.dim() > 2 else image_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]

        # Compute log probability.
        # Training side: prev_latents is float16 from trajectory.
        # Upcast to float32 for stable log_prob.
        # Must use the same SDE formulation as sampling to keep math consistent.
        if self.uses_deterministic_solver:
            raise ValueError("Deterministic FLUX sampling does not define stochastic log-prob replay.")
        sigma = sigma_schedule[timestep_index].to(device)
        sigma_next = sigma_schedule[timestep_index + 1].to(device)
        _, log_prob, _ = denoising_step(
            noise_pred=pred,
            sample=latents,
            sigma=sigma,
            sigma_next=sigma_next,
            eta=self.eta,
            prev_sample=prev_latents,
            sde_type=self.sde_type,
        )

        return log_prob.to(dtype=self.logprob_dtype)
