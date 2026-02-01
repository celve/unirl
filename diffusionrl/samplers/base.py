"""
Base sampler interface for GRPO training.

All samplers must inherit from BaseSampler and implement the sample() method.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Any
import torch

# Import SamplerOutput from types.py to avoid duplicate definitions
from diffusionrl.types import SamplerOutput, LogProbData, PromptEmbeddings


class BaseSampler(ABC):
    """
    Abstract base class for GRPO samplers.

    All samplers must implement the sample() method which generates
    trajectories and log probabilities for policy gradient training.

    Key Design Principles:
    1. Log probabilities MUST be computed at sampling time
    2. Trajectories are stored as [B, num_steps+1, C, ...] tensors
    3. Each sampler specifies whether it requires extra forward for log_prob
    """

    def __init__(
        self,
        eta: float = 1.0,
        sde_type: str = "sde",
        shift: float = 3.0,
    ):
        """
        Initialize sampler.

        Args:
            eta: Noise level for SDE (controls stochasticity)
            sde_type: SDE formulation ("sde", "cps", "dance", "flux_dance", "flux_flow")
            shift: Time shift parameter for sigma schedule
        """
        self.eta = eta
        self.sde_type = sde_type
        self.shift = shift

    @abstractmethod
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
    ) -> SamplerOutput:
        """
        Execute sampling and return trajectories with log probabilities.

        Args:
            prompts: List of text prompts
            prompt_embeds: Pre-computed prompt embeddings [B, seq, hidden]
            pooled_prompt_embeds: Pooled prompt embeddings [B, hidden]
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            latents: Initial latents (if None, sample from noise)
            generator: Random number generator for reproducibility
            sde_indices: Set of timestep indices to use SDE sampling.
                If None, all timesteps use SDE (standard GRPO).
                For MixGRPO, this is a subset of timesteps.
                ODE steps (not in sde_indices) are deterministic and faster.
            **kwargs: Additional model-specific arguments

        Returns:
            SamplerOutput with trajectories, log_probs, etc.
            - log_probs will only contain entries for sde_indices
            - sde_indices in output reflects which steps used SDE
        """
        pass

    @property
    @abstractmethod
    def requires_extra_forward_for_log_prob(self) -> bool:
        """
        Whether this sampler requires an extra forward pass to compute log_prob.

        If True, the sampler computes log_prob using a separate forward pass
        after sampling (less efficient but works without modifying inference code).

        If False, the sampler computes log_prob during the sampling loop
        (more efficient, requires inference code modification).
        """
        pass

    @property
    def supports_video(self) -> bool:
        """Whether this sampler supports video models."""
        return False

    @property
    def supports_image(self) -> bool:
        """Whether this sampler supports image models."""
        return True

    def get_sigma_schedule(
        self,
        num_steps: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Get the sigma (noise level) schedule.

        Args:
            num_steps: Number of denoising steps
            device: Device for the schedule

        Returns:
            Tensor of sigmas [num_steps + 1]
        """
        from .log_prob import get_sigma_schedule
        return get_sigma_schedule(num_steps, self.shift, device)


class TrajectoryReplaySampler(BaseSampler):
    """
    Base class for samplers that replay trajectories to compute log probabilities.

    This is used when the inference backend (e.g., FastVideo) doesn't expose
    noise_pred during sampling. The sampler:
    1. Runs inference to get trajectories
    2. Replays the trajectory with extra forward passes to compute log_probs
    """

    def __init__(
        self,
        model: torch.nn.Module,
        eta: float = 1.0,
        sde_type: str = "sde",
        shift: float = 3.0,
    ):
        """
        Initialize replay sampler.

        Args:
            model: The diffusion model for computing log probabilities
            eta: Noise level for SDE
            sde_type: SDE formulation
            shift: Time shift parameter
        """
        super().__init__(eta=eta, sde_type=sde_type, shift=shift)
        self.model = model

    @property
    def requires_extra_forward_for_log_prob(self) -> bool:
        return True

    def compute_log_probs_from_trajectory(
        self,
        trajectories: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 3.5,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        """
        Compute log probabilities by replaying the trajectory.

        This performs N extra forward passes (one per timestep) to compute
        the noise_pred needed for log_prob calculation.

        Args:
            trajectories: [B, T+1, C, ...] sampled latents at each step
            timesteps: [T+1] sigma values
            prompt_embeds: [B, seq, hidden] prompt embeddings
            pooled_prompt_embeds: [B, hidden] pooled embeddings (optional)
            guidance_scale: CFG scale

        Returns:
            Dict mapping step index to log_prob tensor [B]
        """
        from .log_prob import compute_sde_log_prob

        log_probs = {}
        num_steps = trajectories.shape[1] - 1
        device = trajectories.device

        with torch.no_grad():
            for t_idx in range(num_steps):
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

    @abstractmethod
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

        This should handle CFG and return the final noise_pred.

        Args:
            latents: [B, C, ...] noisy latents
            sigma: Current sigma value
            prompt_embeds: [B, seq, hidden] prompt embeddings
            pooled_prompt_embeds: [B, hidden] pooled embeddings
            guidance_scale: CFG scale

        Returns:
            noise_pred: [B, C, ...] velocity prediction
        """
        pass
