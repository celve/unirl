"""
FastVideo sampler for GRPO training (FastVideo Engine).

This sampler uses FastVideo for efficient video generation and computes
log probabilities using an extra forward pass (Phase 1 implementation).

In Phase 2, after FastVideo PR is merged, log_prob will be computed
directly during sampling.
"""

from typing import Dict, List, Optional, Any, Callable, Set
import torch
import torch.nn as nn

from ..base import BaseSampler, RolloutOutput, TrajectoryReplaySampler
from ..log_prob import compute_sde_log_prob, get_sigma_schedule
from diffusionrl.types import LogProbData, PromptEmbeddings


class FastVideoSampler(TrajectoryReplaySampler):
    """
    FastVideo-based sampler for video GRPO training.

    Phase 1 Implementation:
    - Uses FastVideo for sampling (get trajectories)
    - Computes log_prob with extra forward passes after sampling

    Phase 2 (after FastVideo PR):
    - FastVideo computes log_prob during sampling
    - No extra forward passes needed

    Example usage:
        from fastvideo import VideoGenerator

        generator = VideoGenerator(model_path="...")
        sampler = FastVideoSampler(
            generator=generator,
            model=model,  # For log_prob computation
            eta=1.0,
            sde_type="sde",
        )

        output = sampler.sample(
            prompts=["a cat running"],
            num_inference_steps=28,
            guidance_scale=3.5,
        )
        # output.trajectories: [B, T+1, C, T_frames, H, W]
        # output.log_probs: {step_idx: [B]}
    """

    def __init__(
        self,
        model: nn.Module,
        generator: Optional[Any] = None,  # VideoGenerator from FastVideo (optional)
        text_encoder: Optional[nn.Module] = None,
        eta: float = 1.0,
        sde_type: str = "sde",
        shift: float = 3.0,
        model_forward_fn: Optional[Callable] = None,
    ):
        """
        Initialize FastVideo sampler.

        Args:
            model: Diffusion model for log_prob computation
            generator: FastVideo VideoGenerator instance (optional, can be set later)
            text_encoder: Text encoder for prompt encoding (optional)
            eta: Noise level for SDE
            sde_type: SDE formulation ("sde", "cps", "dance")
            shift: Time shift parameter
            model_forward_fn: Custom forward function for the model
                              If None, uses default forward logic
        """
        super().__init__(model=model, eta=eta, sde_type=sde_type, shift=shift)
        self.generator = generator
        self.text_encoder = text_encoder
        self.model_forward_fn = model_forward_fn

    @property
    def supports_video(self) -> bool:
        return True

    @property
    def supports_image(self) -> bool:
        # FastVideo primarily supports video models
        # Image support depends on generator configuration
        return False

    def sample(
        self,
        prompts: List[str],
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        latents: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        sde_indices: Optional[Set[int]] = None,
        **kwargs,
    ) -> RolloutOutput:
        """
        Sample videos using FastVideo and compute log probabilities.

        Args:
            prompts: List of text prompts
            prompt_embeds: Pre-computed prompt embeddings
            pooled_prompt_embeds: Pooled prompt embeddings
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            latents: Initial noise (optional)
            generator: Random generator
            sde_indices: Set of timestep indices to use SDE sampling.
                If None, all timesteps use SDE (standard GRPO).
                For MixGRPO, this is a subset of timesteps.
            **kwargs: Additional FastVideo arguments (height, width, num_frames, etc.)

        Returns:
            RolloutOutput with video trajectories and log_probs
        """
        # Step 1: Run FastVideo sampling with trajectory tracking
        # Pass sde_indices for mixed ODE/SDE if supported
        fastvideo_output = self._run_fastvideo_sampling(
            prompts=prompts,
            prompt_embeds=prompt_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            latents=latents,
            sde_indices=sde_indices,
            **kwargs,
        )

        # Extract from FastVideo output
        trajectories = fastvideo_output["trajectory_latents"]  # [B, T+1, C, ...]
        timesteps = fastvideo_output["trajectory_timesteps"]   # [T+1]
        final_latents = fastvideo_output["latents"]            # [B, C, ...]

        # Step 2: Encode prompts if not provided
        if prompt_embeds is None:
            prompt_embeds, pooled_prompt_embeds = self._encode_prompts(prompts)

        # Step 3: Determine SDE indices
        num_steps = trajectories.shape[1] - 1
        if sde_indices is None:
            # All SDE (standard GRPO)
            actual_sde_indices = set(range(num_steps))
        else:
            # MixGRPO: only specified indices use SDE
            actual_sde_indices = set(i for i in sde_indices if i < num_steps)

        # Step 4: Compute log_probs only for SDE steps
        log_probs_dict = self.compute_log_probs_from_trajectory_mixed(
            trajectories=trajectories,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            guidance_scale=guidance_scale,
            sde_indices=actual_sde_indices,
            **kwargs,
        )

        # Step 5: Create embeddings bundle for downstream use
        embeddings = PromptEmbeddings(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
        )

        return RolloutOutput(
            latents=final_latents,
            trajectories=trajectories,
            log_probs=LogProbData.from_dict(log_probs_dict),
            timesteps=timesteps,
            embeddings=embeddings,
            metadata={
                "generator_type": "fastvideo",
                "model_type": self._get_model_type(),
                "engine_capabilities": {
                    "supports_logprob": True,
                    "supports_trajectory": True,
                    "supports_prompt_embeddings": True,
                },
                "mixed_sampling": sde_indices is not None,
                "num_sde_steps": len(actual_sde_indices),
                "num_ode_steps": num_steps - len(actual_sde_indices),
                "sde_indices": actual_sde_indices,
                "trajectory_format": "video_dense_latent",
                "timestep_type": "sigma",
                "timestep_scale": 1.0,
            },
            step_indices=torch.arange(timesteps.shape[0], device=timesteps.device, dtype=torch.long),
        )

    def compute_log_probs_from_trajectory_mixed(
        self,
        trajectories: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor],
        guidance_scale: float,
        sde_indices: Set[int],
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        """
        Compute log probabilities only for SDE steps (MixGRPO).

        This is more efficient than computing for all steps when using
        mixed ODE/SDE sampling.

        Args:
            trajectories: [B, T+1, C, ...] sampled latents
            timesteps: [T+1] sigma values
            prompt_embeds: [B, seq, hidden]
            pooled_prompt_embeds: [B, hidden] (optional)
            guidance_scale: CFG scale
            sde_indices: Set of timestep indices to compute log_prob for

        Returns:
            Dict mapping step index to log_prob tensor [B]
        """
        log_probs = {}
        num_steps = trajectories.shape[1] - 1
        device = trajectories.device

        with torch.no_grad():
            for t_idx in sde_indices:
                if t_idx >= num_steps:
                    continue

                x_t = trajectories[:, t_idx]
                x_t_minus_1 = trajectories[:, t_idx + 1]
                sigma = timesteps[t_idx].to(device)
                sigma_next = timesteps[t_idx + 1].to(device)

                # Forward pass to get noise_pred
                noise_pred = self._forward_model(
                    x_t, sigma, prompt_embeds, pooled_prompt_embeds,
                    guidance_scale=guidance_scale, **kwargs
                )

                # Compute log probability
                log_prob, _ = compute_sde_log_prob(
                    noise_pred=noise_pred,
                    sample=x_t,
                    prev_sample=x_t_minus_1,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    eta=self.eta,
                    sde_type=self.sde_type,
                )

                log_probs[t_idx] = log_prob

        return log_probs

    def compute_log_prob_for_training(
        self,
        latents: torch.Tensor,
        prev_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timestep_index: int,
        sigma_schedule: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute one-step log probability from replayed FastVideo trajectory."""
        if self.model is None:
            raise RuntimeError(
                "FastVideoSampler replay requires model to be initialized on training actor."
            )

        if timestep_index < 0 or timestep_index >= int(sigma_schedule.shape[0]) - 1:
            raise ValueError(
                "timestep_index out of range for sigma_schedule: "
                f"index={timestep_index}, len={sigma_schedule.shape[0]}"
            )

        device = latents.device
        sigma_schedule = sigma_schedule.to(device=device, dtype=torch.float32)
        sigma = sigma_schedule[timestep_index]
        sigma_next = sigma_schedule[timestep_index + 1]

        prompt_embeds = prompt_embeds.to(device=device)
        if pooled_prompt_embeds is not None:
            pooled_prompt_embeds = pooled_prompt_embeds.to(device=device)
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.to(device=device)
        actual_guidance = (
            float(guidance_scale) if guidance_scale is not None else 1.0
        )

        model_kwargs = dict(kwargs)
        if encoder_attention_mask is not None:
            model_kwargs.setdefault("encoder_attention_mask", encoder_attention_mask)

        with torch.no_grad():
            noise_pred = self._forward_model(
                latents=latents,
                sigma=sigma,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                guidance_scale=actual_guidance,
                **model_kwargs,
            )

            log_prob, _ = compute_sde_log_prob(
                noise_pred=noise_pred,
                sample=latents,
                prev_sample=prev_latents,
                sigma=sigma,
                sigma_next=sigma_next,
                eta=self.eta,
                sde_type=self.sde_type,
            )
        return log_prob

    def _run_fastvideo_sampling(
        self,
        prompts: List[str],
        prompt_embeds: Optional[torch.Tensor],
        num_inference_steps: int,
        guidance_scale: float,
        latents: Optional[torch.Tensor],
        sde_indices: Optional[Set[int]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Run FastVideo sampling with trajectory tracking.

        For MixGRPO, FastVideo can use ODE steps for non-SDE indices,
        which is faster. The sde_indices parameter tells FastVideo which
        steps need stochastic sampling.

        This is a placeholder that should be adapted based on the actual
        FastVideo API. The key requirements are:
        - return_trajectory_latents=True
        - deterministic_indices (optional, for MixGRPO)

        Returns:
            Dict with:
                - trajectory_latents: [B, T+1, C, T_frames, H, W]
                - trajectory_timesteps: [T+1]
                - latents: [B, C, T_frames, H, W]
        """
        # Import FastVideo components
        try:
            from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
        except ImportError:
            raise ImportError(
                "FastVideo is required. Install it with: pip install fastvideo"
            )

        # Determine deterministic (ODE) indices for MixGRPO
        deterministic_indices = None
        if sde_indices is not None:
            # ODE indices = all indices not in sde_indices
            all_indices = set(range(num_inference_steps))
            deterministic_indices = list(all_indices - sde_indices)

        # Build ForwardBatch
        batch_kwargs = {
            "prompt": prompts,
            "return_trajectory_latents": True,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            **kwargs,
        }

        # Add deterministic_indices if FastVideo supports it
        # This enables mixed ODE/SDE sampling
        if deterministic_indices is not None:
            batch_kwargs["deterministic_indices"] = deterministic_indices

        batch = ForwardBatch(**batch_kwargs)

        # Run through FastVideo executor
        # Note: The exact API may vary based on FastVideo version
        output = self.generator.executor.execute_forward(batch)

        # Convert to our format
        # FastVideo returns list of tensors, we stack them
        trajectory_latents = torch.stack(output.trajectory_latents, dim=1)
        trajectory_timesteps = torch.tensor(output.trajectory_timesteps)

        return {
            "trajectory_latents": trajectory_latents,
            "trajectory_timesteps": trajectory_timesteps,
            "latents": output.output,
        }

    def _encode_prompts(
        self,
        prompts: List[str],
    ) -> tuple:
        """
        Encode text prompts to embeddings.

        Returns:
            (prompt_embeds, pooled_prompt_embeds)
        """
        if self.text_encoder is None:
            raise ValueError(
                "text_encoder is required when prompt_embeds is not provided"
            )

        # This should be adapted based on the model type
        # Different models (Wan, HunyuanVideo, etc.) have different encoders
        with torch.no_grad():
            outputs = self.text_encoder(prompts)

        if isinstance(outputs, tuple):
            return outputs[0], outputs[1] if len(outputs) > 1 else None
        return outputs, None

    def _forward_model(
        self,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor],
        guidance_scale: float,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through the model to get noise prediction.

        Handles CFG if guidance_scale > 1.
        """
        if self.model_forward_fn is not None:
            return self.model_forward_fn(
                latents=latents,
                sigma=sigma,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                guidance_scale=guidance_scale,
                **kwargs,
            )

        # Default forward logic
        # Note: This needs to be adapted for specific model architectures
        # (Wan, HunyuanVideo, StepVideo, etc.)

        batch_size = latents.shape[0]
        device = latents.device

        # Prepare timestep
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0)
        timestep = sigma.expand(batch_size).to(device)

        if guidance_scale > 1.0:
            # CFG: run conditional and unconditional forward
            latents_input = torch.cat([latents, latents], dim=0)
            timestep_input = torch.cat([timestep, timestep], dim=0)

            # Assume unconditional embeds are zeros or from a separate encoder
            uncond_embeds = torch.zeros_like(prompt_embeds)
            embeds_input = torch.cat([prompt_embeds, uncond_embeds], dim=0)

            pooled_input = None
            if pooled_prompt_embeds is not None:
                uncond_pooled = torch.zeros_like(pooled_prompt_embeds)
                pooled_input = torch.cat([pooled_prompt_embeds, uncond_pooled], dim=0)

            noise_pred = self.model(
                latents_input,
                timestep_input,
                encoder_hidden_states=embeds_input,
                pooled_projections=pooled_input,
                **kwargs,
            )

            # CFG combination
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
        else:
            noise_pred = self.model(
                latents,
                timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                **kwargs,
            )

        return noise_pred

    def _get_model_type(self) -> str:
        """Get the model type from generator."""
        # This should be adapted based on FastVideo's model detection
        return getattr(self.generator, "model_type", "unknown")


class FastVideoSamplerV2(BaseSampler):
    """
    FastVideo sampler V2 - for use after FastVideo PR is merged.

    This version expects FastVideo to compute log_prob during sampling,
    eliminating the need for extra forward passes.

    NOT YET IMPLEMENTED - placeholder for Phase 2.
    """

    def __init__(self, generator: Any, eta: float = 1.0, sde_type: str = "sde"):
        super().__init__(eta=eta, sde_type=sde_type)
        self.generator = generator

    @property
    def requires_extra_forward_for_log_prob(self) -> bool:
        # After FastVideo PR, no extra forward needed
        return False

    @property
    def supports_video(self) -> bool:
        return True

    def sample(
        self,
        prompts: List[str],
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        latents: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        sde_indices: Optional[Set[int]] = None,
        **kwargs,
    ) -> RolloutOutput:
        """
        Sample using FastVideo with built-in log_prob computation.

        This requires the FastVideo PR to be merged that adds:
        - return_log_probs parameter to ForwardBatch
        - trajectory_log_probs output
        - deterministic_indices for MixGRPO support

        Args:
            prompts: List of text prompts
            prompt_embeds: Pre-computed prompt embeddings
            pooled_prompt_embeds: Pooled prompt embeddings
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            latents: Initial noise (optional)
            generator: Random generator
            sde_indices: Set of timestep indices for SDE (MixGRPO)
            **kwargs: Additional FastVideo arguments

        Raises:
            NotImplementedError: Until FastVideo PR is merged
        """
        raise NotImplementedError(
            "FastVideoSamplerV2 requires FastVideo PR to be merged. "
            "Use FastVideoSampler (extra forward) for now."
        )
