"""
diffusionrl Algorithm Base Class.

Defines algorithm responsibilities in rollout/advantage pipeline.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import inspect
from typing import Any, Dict, List, Optional, Set
import warnings

import torch
import torch.nn as nn

from .normalizers import normalize_global, normalize_grouped, build_fixed_size_groups, build_prompt_groups


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


class BaseAlgorithm(ABC):
    """
    Base class for algorithm plugins.

    Each algorithm variant implements:
    - get_sampling_requirements(): What the sampler needs to provide
    - compute_advantages(): How to compute advantages from rewards
    - optional timestep filtering helpers for backward training assembly

    Notes:
    - Loss objects (see diffusionrl.losses) are created independently via
      ``load_function(loss_path) + cls.from_config()``.  Algorithm does
      NOT own loss creation — it only handles advantage computation and
      sampling requirements.
    """

    def __init__(
        self,
        clip_range: float = 1e-4,
        kl_coef: float = 0.01,
        advantage_type: str = "group",
        epsilon: float = 1e-8,
        clip_max: Optional[float] = 5.0,
        use_per_prompt_tracker: bool = False,
        per_prompt_mode: str = "batch",
        per_prompt_buffer_size: int = 16,
        per_prompt_min_count: int = 2,
        use_running_stats: bool = False,
        running_stats_warmup: int = 0,
        use_global_std: bool = False,
        trimmed_ratio: float = 0.0,
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
            use_per_prompt_tracker: Use PerPromptStatTracker for cross-batch stats
            per_prompt_mode: "running" (tracker) or "batch" (per-batch stats)
            per_prompt_buffer_size: Buffer size for per-prompt tracker
            per_prompt_min_count: Min samples before using per-prompt stats
            use_running_stats: Use RunningMeanStd for cross-batch global normalization
            running_stats_warmup: Warmup batches before using running stats
            use_global_std: Use global std instead of per-group std
            trimmed_ratio: Ratio of outliers trimmed from each side for grouped stats
            **kwargs: Additional algorithm-specific arguments
        """
        self.clip_range = clip_range
        self.kl_coef = kl_coef
        self.advantage_type = advantage_type
        self.epsilon = epsilon
        self.clip_max = clip_max
        self.per_prompt_mode = per_prompt_mode
        self.use_global_std = use_global_std
        self.trimmed_ratio = max(0.0, min(float(trimmed_ratio), 0.49))
        self._extra_kwargs = kwargs

        # Per-prompt statistics tracker
        self.per_prompt_tracker = None
        if (use_per_prompt_tracker or advantage_type == "per_prompt") and per_prompt_mode == "running":
            from .per_prompt_tracker import PerPromptStatTracker
            self.per_prompt_tracker = PerPromptStatTracker(
                buffer_size=per_prompt_buffer_size,
                min_count=per_prompt_min_count,
                epsilon=epsilon,
                clip_max=clip_max,
                use_global_std=use_global_std,
            )

        # Running statistics for cross-batch global normalization (DanceGRPO)
        self.running_reward_normalizer = None
        if use_running_stats:
            from .running_stats import RunningRewardNormalizer
            self.running_reward_normalizer = RunningRewardNormalizer(
                epsilon=epsilon,
                clip_max=clip_max,
                warmup_steps=running_stats_warmup,
            )

    @classmethod
    def _base_kwargs_from_args(cls, args: Any) -> Dict[str, Any]:
        """Build shared constructor kwargs from runtime args."""
        return {
            "clip_range": args.algorithm.clip_range,
            "kl_coef": getattr(args.algorithm, "kl_coef", 0.01),
            "advantage_type": getattr(args.algorithm, "advantage_type", "group"),
            "epsilon": getattr(args.algorithm, "advantage_epsilon", 1e-8),
            "clip_max": getattr(args.algorithm, "advantage_clip_max", None),
        }

    @classmethod
    def _algorithm_kwargs_from_args(cls, args: Any) -> Dict[str, Any]:
        """Read normalized algorithm_kwargs dictionary from args."""
        raw = getattr(args.algorithm, "algorithm_kwargs", {})
        if raw is None:
            return {}
        if isinstance(raw, dict):
            payload = dict(raw)
            allowed: Set[str] = set()
            for owner in cls.mro():
                init_fn = owner.__dict__.get("__init__")
                if not callable(init_fn):
                    continue
                try:
                    sig = inspect.signature(init_fn)
                except (TypeError, ValueError):
                    continue
                for name, param in sig.parameters.items():
                    if name == "self":
                        continue
                    if param.kind in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    ):
                        allowed.add(name)
            unknown = sorted(key for key in payload.keys() if key not in allowed)
            if unknown:
                warnings.warn(
                    f"{cls.__name__} received unknown algorithm_kwargs keys: {unknown}. "
                    "These keys are currently not declared in constructor signatures and may be ignored.",
                    stacklevel=3,
                )
            return payload
        raise ValueError(
            "algorithm.algorithm_kwargs must be a dict after parse/validate, "
            f"got: {type(raw).__name__}"
        )

    @classmethod
    def from_args(cls, args: Any) -> "BaseAlgorithm":
        """
        Construct algorithm instance from TrainingArguments.

        Subclasses can override to parse algorithm-specific parameters.
        """
        kwargs = cls._base_kwargs_from_args(args)
        kwargs.update(cls._algorithm_kwargs_from_args(args))
        return cls(**kwargs)

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

        Dispatches to the appropriate normalization method based on
        ``self.advantage_type`` ("global", "group", or "per_prompt").

        Args:
            rewards: Reward tensor [batch_size]
            num_samples_per_prompt: Number of samples generated per prompt
            prompts: Optional list of prompt strings (for per_prompt strategies)

        Returns:
            Advantage tensor [batch_size]
        """
        if self.advantage_type == "global":
            return self._normalize_global(rewards)
        elif self.advantage_type == "group":
            return self._normalize_group(rewards, num_samples_per_prompt)
        elif self.advantage_type == "per_prompt":
            return self._normalize_per_prompt(
                rewards, num_samples_per_prompt, prompts
            )
        else:
            raise ValueError(f"Unknown advantage_type: {self.advantage_type}")

    def _normalize_global(self, rewards: torch.Tensor) -> torch.Tensor:
        """Global normalization across all samples.

        If running_reward_normalizer is enabled (DanceGRPO mode), uses
        cross-batch accumulated statistics for stable normalization.
        Otherwise uses batch-only statistics.
        """
        if self.running_reward_normalizer is not None:
            return self.running_reward_normalizer.normalize(rewards, update_stats=True)
        return normalize_global(rewards, epsilon=self.epsilon, clip_max=self.clip_max)

    def _normalize_group(
        self,
        rewards: torch.Tensor,
        num_samples_per_prompt: int,
    ) -> torch.Tensor:
        """Group normalization within prompt groups."""
        batch_size = rewards.shape[0]
        if num_samples_per_prompt <= 0 or batch_size % num_samples_per_prompt != 0:
            return self._normalize_global(rewards)
        groups = build_fixed_size_groups(batch_size, num_samples_per_prompt)
        return normalize_grouped(
            rewards,
            groups,
            epsilon=self.epsilon,
            clip_max=self.clip_max,
            trimmed_ratio=self.trimmed_ratio,
            use_global_std=self.use_global_std,
        )

    def _normalize_per_prompt(
        self,
        rewards: torch.Tensor,
        num_samples_per_prompt: int,
        prompts: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Per-prompt normalization.

        When a PerPromptStatTracker is active (per_prompt_mode='running'),
        delegates to the tracker for cross-batch statistics.

        Otherwise (per_prompt_mode='batch', the default) normalizes using
        only the current batch, grouping samples by prompt text.
        """
        # Running mode: delegate to cross-batch tracker
        if self.per_prompt_tracker is not None and prompts is not None:
            if len(prompts) * num_samples_per_prompt == len(rewards):
                prompts = [p for p in prompts for _ in range(num_samples_per_prompt)]
            return self.per_prompt_tracker.compute_advantages(
                prompts, rewards, update_stats=True
            )
        # Batch mode: normalize within current batch per prompt text
        if prompts is not None:
            # Expand prompts to per-sample list if needed
            if len(prompts) * num_samples_per_prompt == len(rewards):
                prompts = [p for p in prompts for _ in range(num_samples_per_prompt)]
            if len(prompts) == len(rewards):
                groups = build_prompt_groups(prompts)
                return normalize_grouped(
                    rewards,
                    groups,
                    epsilon=self.epsilon,
                    clip_max=self.clip_max,
                    trimmed_ratio=self.trimmed_ratio,
                    use_global_std=self.use_global_std,
                )
        # Fall back to fixed-size group normalization
        return self._normalize_group(rewards, num_samples_per_prompt)

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

    def is_forward_process(self) -> bool:
        """Whether this algorithm trains via forward process (NFT-style)."""
        return bool(self.get_sampling_requirements().is_forward_process)

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
        if self.is_forward_process() or timestep_scheduler is None:
            return None

        sde_indices = set(int(i) for i in timestep_scheduler.get_sde_indices(current_step))
        if hasattr(self, "set_sde_indices"):
            self.set_sde_indices(sde_indices)
        return sde_indices

    def get_sampler_validation_config(self, *, args: Any) -> Dict[str, Any]:
        """Get sampler-output validation flags for rollout orchestration."""
        is_forward = self.is_forward_process()
        allow_replay = (
            not is_forward
            and bool(getattr(args.sampling, "replay_log_probs", False))
            and getattr(args.algorithm, "loss_type", "grpo") == "grpo"
        )
        return {
            "allow_replay": allow_replay,
            "assert_step_alignment": (not is_forward),
            "mode_label": ("forward" if is_forward else "trajectory"),
        }

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
        """Assemble typed training batch. Subclasses can override strategy."""
        from diffusionrl.runtime.pipeline.rollout_pipeline import (
            assemble_backward_training_batch,
            assemble_forward_training_batch,
        )

        if self.is_forward_process():
            return assemble_forward_training_batch(
                sampler_outputs=sampler_outputs,
                rewards=rewards,
                advantages=advantages,
                prompts=prompts,
            )

        return assemble_backward_training_batch(
            algorithm=self,
            num_inference_steps=num_inference_steps,
            sampler_outputs=sampler_outputs,
            rewards=rewards,
            advantages=advantages,
            prompts=prompts,
            sde_indices=sde_indices,
        )

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
