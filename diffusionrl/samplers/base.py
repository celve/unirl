"""
Base sampler interface for GRPO training.

All samplers must inherit from BaseSampler and implement the sample() method.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Set
import torch

# Import shared data types from canonical types package
from diffusionrl.sde.rules import is_deterministic_sde_type, normalize_sde_type
from diffusionrl.types import RolloutSamples


class BaseSampler(ABC):
    """
    Abstract base class for GRPO samplers.

    All samplers must implement the sample() method which generates
    trajectories and log probabilities for policy gradient training.

    Key Design Principles:
    1. Log probabilities MUST be computed at sampling time
    2. Trajectories are stored as [B, num_steps+1, C, ...] tensors

    Model-Specific Parameter Contracts:
        Different model architectures handle certain parameters differently.
        Subclasses should document which optional kwargs they actually use.

        - ``guidance_scale``: Semantics vary per model. Flux treats it as
          optional (defaults to instance attr), SD3 defaults to 7.0,
          HunyuanVideo uses a fixed internal value and ignores this arg.
        - ``text_ids``: Only used by Flux-family models for positional
          encoding; other models should accept and ignore via **kwargs.
        - Prompt encoding: Some samplers use ``prompt_embeds`` /
          ``pooled_prompt_embeds`` directly, others re-encode from
          ``prompts`` text internally. Check ``requires_pre_encoded``
          on the concrete sampler if available.
    """

    def __init__(
        self,
        eta: float = 1.0,
        sde_type: str = "flow",
        shift: float = 3.0,
    ):
        """
        Initialize sampler.

        Args:
            eta: Noise level for SDE (controls stochasticity)
            sde_type: Transition rule (flow/cps/dance/dpm2)
            shift: Time shift parameter for sigma schedule
        """
        self.eta = eta
        self.sde_type = normalize_sde_type(sde_type)
        self.shift = shift

    @property
    def uses_deterministic_solver(self) -> bool:
        """Whether rollout should bypass stochastic SDE steps."""

        return is_deterministic_sde_type(self.sde_type, eta=self.eta)

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
    ) -> RolloutSamples:
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
            RolloutSamples with trajectories, log_probs, etc.
            - log_probs will only contain entries for sde_indices
            - sde_indices in output reflects which steps used SDE
        """
        pass

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
        from diffusionrl.sde.runtime import get_sigma_schedule
        return get_sigma_schedule(num_steps, self.shift, device)
