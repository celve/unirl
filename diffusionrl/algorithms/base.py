"""
diffusionrl Algorithm Base Class.

Defines algorithm responsibilities in rollout/advantage pipeline.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from .normalizers import normalize_global, normalize_grouped

logger = logging.getLogger(__name__)


@dataclass
class SamplingRequirements:
    """
    Sampling extras specified by algorithm implementations.

    Runtime note:
    - In diffusionRL, ``requires_*`` is resolved from loss
      ``declared_requirements()`` as the single source of truth.
    - The ``requires_*`` fields here are kept for backward compatibility,
      but runtime control-plane logic may ignore them.

    Algorithm-specific extras (e.g. ``sde_ratio`` for MixGRPO,
    ``requires_clean_latents`` for NFT) go into the open ``extras``
    dict.  This lets new algorithms declare their own requirements
    without modifying this shared dataclass.

    Backward-compatible ``@property`` accessors are provided for
    commonly used extras so existing consumers keep working.
    """

    requires_trajectory: bool = True
    """Whether the algorithm needs full denoising trajectories."""

    requires_log_prob: bool = True
    """Whether the algorithm needs log probabilities at each step."""

    requires_embeddings: bool = True
    """Whether the algorithm needs prompt embeddings in the sampled batch."""

    extras: Dict[str, Any] = field(default_factory=dict)
    """Open dict for algorithm-specific sampler extras.

    Known keys (non-exhaustive):
    - ``"sde_ratio"`` (float): fraction of SDE steps, default 1.0 (MixGRPO)
    - ``"requires_clean_latents"`` (bool): need clean x0 (NFT)
    - ``"forward_diffusion_in_loss"`` (bool): forward process in loss (NFT)
    """

    # ------------------------------------------------------------------
    # Backward-compatible property accessors
    # ------------------------------------------------------------------

    @property
    def sde_ratio(self) -> float:
        """Ratio of SDE steps (1.0 = all SDE, 0.0 = all ODE)."""
        return float(self.extras.get("sde_ratio", 1.0))

    @property
    def requires_clean_latents(self) -> bool:
        """Whether the algorithm needs clean latents x0."""
        return bool(self.extras.get("requires_clean_latents", False))

    @property
    def forward_diffusion_in_loss(self) -> bool:
        """Whether forward diffusion happens in loss computation."""
        return bool(self.extras.get("forward_diffusion_in_loss", False))

    # ------------------------------------------------------------------
    # Derived convenience properties
    # ------------------------------------------------------------------

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


@dataclass(frozen=True)
class EMASpec:
    """Algorithm-declared EMA policy.

    The algorithm declares *what* EMA behavior it needs. Runtime owns
    the mechanism that materializes and updates the trackers.
    """

    enable_eval_ema: bool = True
    eval_decay: float = 0.9
    eval_update_interval: int = 1
    reference_mode: str = "none"
    reference_decay: float = 0.0
    reference_decay_type: str = "constant"
    reference_flat_steps: int = 0
    reference_uprate: float = 0.001
    reference_uphold: float = 0.5
    old_adapter_name: str = "old"
    new_adapter_name: str = "default"


class BaseAlgorithm(ABC):
    """
    Base class for algorithm plugins.

    Each algorithm variant implements:
    - declared_requirements(): Static data requirements (delegated from loss)
    - from_config(): Construct from algorithm_config dict (classmethod)
    - get_sampling_requirements(): What the sampler needs to provide
    - compute_advantages(): How to compute advantages from rewards
    - compute_loss(): Single algorithm-owned loss entrypoint
    - optional timestep filtering helpers for backward training assembly

    The Algorithm class is the single source of truth for both rollout-side
    requirements (sampling, advantages) and training-side gradient computation.
    """

    _loss_cls = None

    def __init__(
        self,
        clip_range: float = 1e-4,
        kl_coef: float = 0.01,
        adv_normalization: str = "group",
        samples_per_prompt: int = 1,
        eval_ema_decay: float = 0.9,
        eval_ema_update_interval: int = 1,
        epsilon: float = 1e-8,
        clip_max: Optional[float] = 5.0,
        use_global_std: bool = False,
        trimmed_ratio: float = 0.0,
        **kwargs,
    ):
        """
        Initialize algorithm.

        Args:
            clip_range: PPO clip range for importance ratio
            kl_coef: KL penalty coefficient
            adv_normalization: Type of advantage normalization ("global" or "group")
            samples_per_prompt: Number of rollout samples to generate per prompt
            eval_ema_decay: Eval-time EMA decay
            eval_ema_update_interval: Eval-time EMA update interval in optimizer steps
            epsilon: Small value for numerical stability in advantage normalization
            clip_max: Maximum advantage value for clipping (None to disable)
            use_global_std: Use global std instead of per-group std
            trimmed_ratio: Ratio of outliers trimmed from each side for grouped stats
            **kwargs: Additional algorithm-specific arguments
        """
        self.clip_range = clip_range
        self.kl_coef = kl_coef
        self.adv_normalization = adv_normalization
        self.samples_per_prompt = max(1, int(samples_per_prompt))
        self.eval_ema_decay = float(eval_ema_decay)
        self.eval_ema_update_interval = max(1, int(eval_ema_update_interval))
        self.epsilon = epsilon
        self.clip_max = clip_max
        self.use_global_std = use_global_std
        self.trimmed_ratio = max(0.0, min(float(trimmed_ratio), 0.49))
        self._extra_kwargs = kwargs

        self.loss_fn = self._create_loss_fn()

    @classmethod
    def _resolve_loss_class(cls):
        loss_cls = getattr(cls, "_loss_cls", None)
        if loss_cls is None:
            raise NotImplementedError(
                f"{cls.__name__} must define _loss_cls or override declared_requirements()."
            )
        return loss_cls

    def _create_loss_fn(self):
        loss_cls = getattr(type(self), "_loss_cls", None)
        if loss_cls is None:
            return None
        return loss_cls(self)

    # ------------------------------------------------------------------
    # Class-level contracts (override in subclasses)
    # ------------------------------------------------------------------

    @classmethod
    def declared_requirements(cls) -> Dict[str, bool]:
        """Declare data requirements for contracts / validation pipeline."""
        loss_cls = cls._resolve_loss_class()
        declared = getattr(loss_cls, "declared_requirements", None)
        if not callable(declared):
            raise NotImplementedError(
                f"{cls.__name__} loss class {loss_cls.__name__} must define "
                "declared_requirements()."
            )
        return dict(declared())

    @classmethod
    def from_config(cls, config: dict) -> "BaseAlgorithm":
        """Construct algorithm from an algorithm_config dictionary.

        TrainingActor calls ``algorithm_cls.from_config(algorithm_config)`` to
        create the train-side algorithm instance.  Subclasses should override
        to read their specific parameters from ``config`` and
        ``config['algorithm_kwargs']``.

        Default implementation raises NotImplementedError so custom
        plugins fail loudly if they forget to implement this.
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement from_config() classmethod."
        )

    @classmethod
    def from_args(cls, args: Any) -> "BaseAlgorithm":
        """Legacy wrapper that normalizes args into algorithm_config first."""
        from diffusionrl.config.build_domain_args import build_algorithm_config

        return cls.from_config(build_algorithm_config(args))

    @abstractmethod
    def get_sampling_requirements(self) -> SamplingRequirements:
        """
        Return the sampling requirements for this algorithm.

        Returns:
            SamplingRequirements specifying what the sampler needs to provide
        """
        ...

    def _build_sampling_requirements(
        self,
        *,
        extras: Optional[Dict[str, Any]] = None,
    ) -> SamplingRequirements:
        """Build SamplingRequirements from the algorithm-owned loss contract."""
        declared = type(self).declared_requirements()
        return SamplingRequirements(
            requires_trajectory=bool(declared.get("requires_trajectory", True)),
            requires_log_prob=bool(declared.get("requires_log_prob", True)),
            requires_embeddings=bool(declared.get("requires_embeddings", True)),
            extras=dict(extras or {}),
        )

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        group_ids: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Compute advantages from rewards.

        Dispatches to the appropriate normalization method based on
        ``self.adv_normalization`` ("global" or "group").

        Args:
            rewards: Reward tensor [batch_size]
            group_ids: Optional explicit sample-group identifiers for batch grouping

        Returns:
            Advantage tensor [batch_size]
        """
        if self.adv_normalization == "global":
            return self._normalize_global(rewards)
        elif self.adv_normalization == "group":
            return self._normalize_group(
                rewards,
                group_ids=group_ids,
            )
        else:
            raise ValueError(f"Unknown adv_normalization: {self.adv_normalization}")

    @staticmethod
    def _normalize_group_id(group_id: Any) -> Optional[str]:
        if group_id is None:
            return None
        text = str(group_id).strip()
        return text if text else None

    def _build_groups_from_ids(self, group_ids: List[str]) -> List[List[int]]:
        ordered_groups: Dict[str, List[int]] = {}
        for sample_idx, raw_group_id in enumerate(group_ids):
            group_id = self._normalize_group_id(raw_group_id)
            if group_id is None:
                continue
            ordered_groups.setdefault(group_id, []).append(sample_idx)
        return list(ordered_groups.values())

    def _normalize_global(self, rewards: torch.Tensor) -> torch.Tensor:
        """Global normalization across all samples."""
        return normalize_global(rewards, epsilon=self.epsilon, clip_max=self.clip_max)

    def _normalize_group(
        self,
        rewards: torch.Tensor,
        group_ids: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Group normalization using explicit sample-group identities."""
        batch_size = rewards.shape[0]
        if group_ids is not None and len(group_ids) == batch_size:
            groups = self._build_groups_from_ids(group_ids)
            if groups:
                return normalize_grouped(
                    rewards,
                    groups,
                    epsilon=self.epsilon,
                    clip_max=self.clip_max,
                    trimmed_ratio=self.trimmed_ratio,
                    use_global_std=self.use_global_std,
                )

        raise ValueError(
            "adv_normalization='group' requires explicit group_ids aligned to the reward batch. "
            f"Got batch_size={batch_size}, "
            f"group_ids_len={len(group_ids) if group_ids is not None else None}."
        )

    # ------------------------------------------------------------------
    # Phase 3: Advantage orchestration with reward components
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_advantages_with_components(
        self,
        *,
        rewards: torch.Tensor,
        group_ids: Optional[List[str]] = None,
        component_mix_stage: str = "reward",
        reward_components: Optional[Dict[str, List[float]]] = None,
        reward_component_weights: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """Full advantage computation pipeline including reward component mixing.

        This consolidates the logic that was previously scattered in
        ``rollout_pipeline.compute_advantages`` into the algorithm itself.

        Args:
            rewards: Aggregated reward tensor [batch_size].
            component_mix_stage: ``"reward"`` (default) uses aggregated rewards.
                ``"advantage"`` computes per-component advantages and
                aggregates with weights.
            reward_components: Per-component reward values, keyed by name.
            reward_component_weights: Per-component weights for aggregation.

        Returns:
            Advantage tensor [batch_size].
        """
        ...

    @abstractmethod
    def get_ema_spec(self) -> EMASpec:
        """Declare EMA policy for this algorithm."""
        ...

    # ------------------------------------------------------------------
    # Phase 2: Algorithm-owned training step
    # ------------------------------------------------------------------

    def compute_loss_and_backward(
        self,
        *,
        model: nn.Module,
        batch: Any,
        gradient_accumulation_batch_size: int,
        guidance_scale: float = 3.5,
        **kwargs: Any,
    ) -> tuple:
        """Compute loss and call backward for a single update chunk."""
        del model, batch, gradient_accumulation_batch_size, guidance_scale, kwargs
        raise NotImplementedError(
            f"{type(self).__name__} must implement compute_loss_and_backward()."
        )

    def is_forward_process(self) -> bool:
        """Whether this algorithm trains via forward process (NFT-style)."""
        return bool(self.get_sampling_requirements().is_forward_process)

    @abstractmethod
    def resolve_rollout_sde_indices(
        self,
        *,
        timestep_scheduler: Optional[Any],
        current_step: int,
    ) -> Optional[Set[int]]:
        """Resolve rollout-time SDE indices for the current step.

        Default behavior:
        - Forward-process algorithms do not use rollout SDE indices.
        - Trajectory algorithms read indices from scheduler and receive the
          optional `set_sde_indices` callback when implemented.
        """
        ...

    @abstractmethod
    def get_sampler_validation_config(self, *, args: Any) -> Dict[str, Any]:
        """Get sampler-output validation flags for rollout orchestration."""
        ...

    @abstractmethod
    def assemble_training_batch(
        self,
        *,
        num_inference_steps: int,
        sampler_outputs: List[Any],
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        prompts: List[str],
        sde_indices: Optional[Set[int]] = None,
    ) -> Any:
        """Assemble the typed training batch for this algorithm."""
        ...

    @abstractmethod
    def get_filtered_training_indices(
        self,
        sde_indices: Set[int],
        num_steps: int,
    ) -> Set[int]:
        """
        Get filtered training timestep indices based on algorithm requirements.

        This method handles common filtering operations like:
        - skip_last_timestep: Skip the last timestep (t->0) which has unstable log_prob
        - skip_initial_timesteps: Skip early timesteps with high variance

        Subclasses can override to add algorithm-specific filtering.

        Args:
            sde_indices: Set of SDE timestep indices from scheduler
            num_steps: Total number of timesteps

        Returns:
            Filtered set of timestep indices for training
        """
        ...

    def get_config(self) -> Dict[str, Any]:
        """Get algorithm configuration as dictionary."""
        return {
            "algorithm_type": self.__class__.__name__,
            "clip_range": self.clip_range,
            "kl_coef": self.kl_coef,
            "adv_normalization": self.adv_normalization,
            **self._extra_kwargs,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"clip_range={self.clip_range}, "
            f"kl_coef={self.kl_coef}, "
            f"adv_normalization={self.adv_normalization})"
        )
