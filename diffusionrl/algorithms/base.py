"""
diffusionrl Algorithm Base Class.

Defines the interface for all GRPO algorithm variants.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn


@dataclass
class SamplingRequirements:
    """
    Requirements for the sampling process specified by the algorithm.

    Different algorithms may have different requirements for:
    - Whether to store full trajectories
    - Whether to compute log probabilities
    - Ratio of SDE vs ODE steps (for MixGRPO)
    - Whether to use forward diffusion process (NFT)
    """

    requires_trajectory: bool = True
    """Whether the algorithm needs full denoising trajectories."""

    requires_log_prob: bool = True
    """Whether the algorithm needs log probabilities at each step."""

    requires_embeddings: bool = True
    """Whether the algorithm needs prompt embeddings in the sampled batch."""

    sde_ratio: float = 1.0
    """Ratio of SDE steps (1.0 = all SDE, 0.0 = all ODE). Used by MixGRPO."""

    requires_clean_latents: bool = False
    """Whether the algorithm needs clean latents x0 (NFT forward process)."""

    forward_diffusion_in_loss: bool = False
    """Whether forward diffusion happens in loss computation (NFT style)."""

    @property
    def is_mixed_sampling(self) -> bool:
        """Whether this uses mixed SDE/ODE sampling."""
        return 0.0 < self.sde_ratio < 1.0

    @property
    def is_trajectory_based(self) -> bool:
        """Whether this is a trajectory-based algorithm (GRPO, MixGRPO)."""
        return self.requires_trajectory

    @property
    def is_forward_process(self) -> bool:
        """Whether this is a forward process algorithm (NFT)."""
        return self.requires_clean_latents and not self.requires_trajectory


class BaseAlgorithm(ABC):
    """
    Base class for GRPO algorithm variants.

    Each algorithm variant implements:
    - get_sampling_requirements(): What the sampler needs to provide
    - compute_advantages(): How to compute advantages from rewards
    - compute_loss(): How to compute the training loss

    Subclasses:
    - GRPOAlgorithm: Standard GRPO with group normalization
    - MixGRPOAlgorithm: Mixed SDE/ODE sampling
    - NFTAlgorithm: Noise-Free Training
    """

    def __init__(
        self,
        clip_range: float = 1e-4,
        kl_coef: float = 0.01,
        advantage_type: str = "group",
        epsilon: float = 1e-8,
        clip_max: Optional[float] = 5.0,
        **kwargs,
    ):
        """
        Initialize algorithm.

        Args:
            clip_range: PPO clip range for importance ratio
            kl_coef: KL penalty coefficient
            advantage_type: Type of advantage normalization ("global", "group", "per_prompt")
            epsilon: Small value for numerical stability in advantage normalization
            clip_max: Maximum advantage value for clipping (None to disable)
            **kwargs: Additional algorithm-specific arguments
        """
        self.clip_range = clip_range
        self.kl_coef = kl_coef
        self.advantage_type = advantage_type
        self.epsilon = epsilon
        self.clip_max = clip_max
        self._extra_kwargs = kwargs

    @abstractmethod
    def get_sampling_requirements(self) -> SamplingRequirements:
        """
        Return the sampling requirements for this algorithm.

        Returns:
            SamplingRequirements specifying what the sampler needs to provide
        """
        ...

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        num_samples_per_prompt: int,
        prompts: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Compute advantages from rewards.

        Default implementation provides global and group normalization.
        Subclasses can override for specialized behavior (e.g., per-prompt tracking).

        Args:
            rewards: Reward tensor [batch_size]
            num_samples_per_prompt: Number of samples generated per prompt
            prompts: Optional list of prompt strings (for per_prompt strategies)

        Returns:
            Advantage tensor [batch_size]
        """
        from diffusionrl.advantages.normalizers import (
            normalize_global,
            normalize_grouped,
            build_fixed_size_groups,
            build_prompt_groups,
        )

        batch_size = rewards.shape[0]

        if self.advantage_type == "global":
            advantages = normalize_global(rewards, epsilon=self.epsilon)
        elif self.advantage_type == "group":
            # Build fixed-size groups based on num_samples_per_prompt
            group_indices = build_fixed_size_groups(batch_size, num_samples_per_prompt)
            advantages = normalize_grouped(
                rewards,
                group_indices=group_indices,
                epsilon=self.epsilon,
            )
        elif self.advantage_type == "per_prompt":
            # Build groups by prompt if available, else fall back to fixed groups
            if prompts is not None and len(prompts) == batch_size:
                group_indices = build_prompt_groups(prompts)
            else:
                group_indices = build_fixed_size_groups(batch_size, num_samples_per_prompt)
            advantages = normalize_grouped(
                rewards,
                group_indices=group_indices,
                epsilon=self.epsilon,
            )
        else:
            raise ValueError(f"Unknown advantage_type: {self.advantage_type}")

        if self.clip_max is not None:
            advantages = advantages.clamp(-self.clip_max, self.clip_max)

        return advantages

    @abstractmethod
    def compute_loss(
        self,
        model: nn.Module,
        batch: Dict[str, Any],
        timestep_idx: int,
        advantages: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute the training loss for a single timestep.

        Args:
            model: The model being trained
            batch: Dictionary containing training data
            timestep_idx: Current timestep index
            advantages: Pre-computed advantages
            **kwargs: Additional arguments (prompt_embeds, etc.)

        Returns:
            Tuple of (loss tensor, metrics dictionary)
        """
        ...

    # ========== Algorithm Hooks ==========
    # These hooks allow algorithms to customize behavior without requiring
    # special-case handling in TrainingActor or RolloutManager.

    def post_backward_hook(self, model: nn.Module, batch: Dict[str, Any]) -> None:
        """Hook called after backward pass.

        Override in subclasses to perform post-backward operations.

        Args:
            model: The model being trained
            batch: The training batch
        """
        pass

    def post_optimizer_step_hook(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Hook called after optimizer step.

        Override in subclasses to perform post-step operations (e.g., EMA updates).

        Args:
            model: The model being trained
            optimizer: The optimizer
            batch: The training batch

        Returns:
            Dictionary of metrics from the hook
        """
        return {}

    def requires_ema_update(self) -> bool:
        """Whether this algorithm requires EMA updates after each step.

        Returns:
            True if EMA update is needed (e.g., NFT)
        """
        return False

    def get_ema_decay(self) -> float:
        """Get EMA decay rate for this algorithm.

        Returns:
            EMA decay rate (0.0 if EMA not used)
        """
        return 0.0

    def get_filtered_training_indices(
        self,
        sde_indices: Set[int],
        num_steps: int,
    ) -> Set[int]:
        """
        Get filtered training timestep indices based on algorithm requirements.

        This method handles common filtering operations like:
        - ignore_last: Skip the last timestep (t->0) which has unstable log_prob
        - frozen_init_timesteps: Skip early timesteps with high variance

        Subclasses can override to add algorithm-specific filtering.

        Args:
            sde_indices: Set of SDE timestep indices from scheduler
            num_steps: Total number of timesteps

        Returns:
            Filtered set of timestep indices for training
        """
        result = set(sde_indices)

        # Apply ignore_last if configured
        ignore_last = getattr(self, 'ignore_last', False)
        if ignore_last and result:
            max_idx = max(result)
            result.discard(max_idx)

        # Apply frozen_init_timesteps if configured
        frozen_init = getattr(self, 'frozen_init_timesteps', 0)
        if frozen_init > 0:
            result = {i for i in result if i >= frozen_init}

        return result

    def compute_aggregated_loss(
        self,
        model: nn.Module,
        batch: Dict[str, Any],
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute aggregated loss across all timesteps.

        Args:
            model: The model being trained
            batch: Dictionary containing training data
            **kwargs: Additional arguments

        Returns:
            Tuple of (average loss tensor, aggregated metrics dictionary)
        """
        sde_indices = batch.get("sde_indices", set())
        if not sde_indices:
            num_steps = batch.get("num_steps", 50)
            sde_indices = set(range(num_steps))

        advantages = batch.get("advantages")
        if advantages is None:
            raise ValueError("batch must contain 'advantages'")

        total_loss = torch.tensor(0.0, device=advantages.device)
        all_metrics: Dict[str, Any] = {}
        num_timesteps = 0

        for t_idx in sde_indices:
            loss_t, metrics_t = self.compute_loss(
                model=model,
                batch=batch,
                timestep_idx=t_idx,
                advantages=advantages,
                **kwargs,
            )
            total_loss += loss_t
            num_timesteps += 1

            # Store per-timestep metrics
            for key, value in metrics_t.items():
                all_metrics[f"t{t_idx}_{key}"] = value

        # Compute averages
        if num_timesteps > 0:
            avg_loss = total_loss / num_timesteps
        else:
            avg_loss = total_loss

        all_metrics["num_timesteps"] = num_timesteps
        all_metrics["total_loss"] = total_loss.item()
        all_metrics["avg_loss"] = avg_loss.item()

        return avg_loss, all_metrics

    def get_config(self) -> Dict[str, Any]:
        """Get algorithm configuration as dictionary."""
        return {
            "algorithm_type": self.__class__.__name__,
            "clip_range": self.clip_range,
            "kl_coef": self.kl_coef,
            "advantage_type": self.advantage_type,
            **self._extra_kwargs,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"clip_range={self.clip_range}, "
            f"kl_coef={self.kl_coef}, "
            f"advantage_type={self.advantage_type})"
        )
