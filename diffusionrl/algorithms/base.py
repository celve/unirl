"""
diffusionrl Algorithm Base Class.

Defines algorithm responsibilities in rollout/advantage pipeline.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from diffusionrl.types.sampling import RolloutRequest, SamplingRequirements

from .normalizers import normalize_global, normalize_grouped

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EMASpec:
    """Algorithm-declared EMA policy.

    The algorithm declares *what* EMA behavior it needs. Runtime owns
    the mechanism that materializes and updates the trackers.
    """

    enable_eval_ema: bool = True
    eval_ema_decay: float = 0.9
    eval_ema_update_interval: int = 1
    reference_mode: str = "none"
    reference_decay: float = 0.0
    reference_decay_type: str = "constant"
    reference_flat_steps: int = 0
    reference_uprate: float = 0.001
    reference_uphold: float = 0.5
    reference_update_timing: str = "optimizer_step"
    old_adapter_name: str = "old"
    new_adapter_name: str = "default"


class BaseAlgorithm(ABC):
    """
    Base class for algorithm plugins.

    Each algorithm variant implements:
    - from_config(): Construct from algorithm_config dict (classmethod)
    - get_sampling_requirements(): What the sampler needs to provide
    - compute_advantages(): How to compute advantages from rewards
    - compute_loss(): Single algorithm-owned loss entrypoint
    - optional timestep filtering helpers for backward training assembly

    The Algorithm class is the single source of truth for both rollout-side
    requirements (sampling, advantages) and training-side gradient computation.

    Training-side forward dispatch
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The ``ForwardContext`` (pure data container with model-specific fields)
    is carried on ``TrainingBatch.forward_context``.  Subclasses read it
    directly from the batch and pair it with a ``ModelForwardPlugin``
    for execution: ``plugin.forward(model, latents, sigma, **ctx.to_dict())``.
    The base class provides :meth:`_get_forward_plugin` (lazy plugin
    resolution) so that subclasses do not need to duplicate that setup.
    """

    _forward_plugin: object = None  # type: ignore[assignment]
    model_type: str = "default"

    def __init__(
        self,
        kl_coef: float = 0.01,
        component_mix_stage: str = "reward",
        adv_normalization_scope: str = "group",
        samples_per_prompt: int = 1,
        num_inference_steps: int = 0,
        eval_ema_decay: float = 0.9,
        eval_ema_update_interval: int = 1,
        epsilon: float = 1e-8,
        clip_max: Optional[float] = 5.0,
        use_global_std: bool = False,
        trim_outliers_ratio: float = 0.0,
        **kwargs,
    ):
        """
        Initialize algorithm.

        Args:
            kl_coef: KL penalty coefficient
            component_mix_stage: Multi-component reward mixing stage ("reward" or "advantage")
            adv_normalization_scope: Scope of advantage normalization ("global" or "group")
            samples_per_prompt: Number of rollout samples to generate per prompt
            num_inference_steps: Rollout denoising horizon shared by algorithm logic
            eval_ema_decay: Eval-time EMA decay
            eval_ema_update_interval: Eval-time EMA update interval in optimizer steps
            epsilon: Small value for numerical stability in advantage normalization
            clip_max: Maximum advantage value for clipping (None to disable)
            use_global_std: Use global std instead of per-group std
            trim_outliers_ratio: Ratio of outliers trimmed from each side for grouped stats
            **kwargs: Additional algorithm-specific arguments
        """
        self.kl_coef = kl_coef
        self.component_mix_stage = str(component_mix_stage)
        self.adv_normalization_scope = adv_normalization_scope
        self.samples_per_prompt = max(1, int(samples_per_prompt))
        self.num_inference_steps = max(0, int(num_inference_steps))
        self.eval_ema_decay = float(eval_ema_decay)
        self.eval_ema_update_interval = max(1, int(eval_ema_update_interval))
        self.epsilon = epsilon
        self.clip_max = clip_max
        self.use_global_std = use_global_std
        self.trim_outliers_ratio = max(0.0, min(float(trim_outliers_ratio), 0.49))
        self._extra_kwargs = kwargs

    # ------------------------------------------------------------------
    # Class-level contracts (override in subclasses)
    # ------------------------------------------------------------------

    @classmethod
    def resolve_config_kwargs(cls, config: dict) -> Dict[str, Any]:
        """Return normalized raw algorithm_kwargs from algorithm_config."""
        extra = config.get("algorithm_kwargs") or {}
        if not isinstance(extra, dict):
            raise ValueError(
                "algorithm_config.algorithm_kwargs must be a dict. "
                f"Got: {type(extra).__name__}"
            )
        return dict(extra)

    @classmethod
    def from_config(cls, config: dict) -> "BaseAlgorithm":  # [PUBLIC-API → rollout runtime init, training_actor.init()]
        """Construct algorithm from an algorithm_config dictionary.

        TrainingActor calls ``algorithm_cls.from_config(algorithm_config)`` to
        create the train-side algorithm instance.  Subclasses should override
        to read their specific parameters from framework-owned top-level config
        fields plus ``cls.resolve_config_kwargs(config)``.

        Default implementation raises NotImplementedError so custom
        plugins fail loudly if they forget to implement this.
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement from_config() classmethod."
        )

    @abstractmethod
    def get_sampling_requirements(
        self,
    ) -> (
        SamplingRequirements
    ):  # [PUBLIC-API → driver rollout runtime] rollout requirements
        """
        Return the sampling requirements for this algorithm.

        Returns:
            SamplingRequirements specifying what the sampler needs to provide
        """
        ...

    def compute_advantages(  # [PUBLIC-API → rollout pipeline via compute_advantages_with_components()] rollout side
        self,
        rewards: torch.Tensor,
        group_ids: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Compute advantages from rewards.

        Dispatches to the appropriate normalization method based on
        ``self.adv_normalization_scope`` ("global" or "group").

        Args:
            rewards: Reward tensor [batch_size]
            group_ids: Optional explicit sample-group identifiers for batch grouping

        Returns:
            Advantage tensor [batch_size]
        """
        if self.adv_normalization_scope == "global":
            return self._normalize_global(rewards)
        else: 
            return self._normalize_group(
                rewards,
                group_ids=group_ids,
            )

    @staticmethod
    def _normalize_group_id(group_id: Any) -> Optional[str]:  # [HELPER]
        if group_id is None:
            return None
        text = str(group_id).strip()
        return text if text else None

    def _require_valid_group_ids(self, group_ids: List[str]) -> List[str]:  # [HELPER → _normalize_group()]
        normalized: List[str] = []
        for sample_idx, raw_group_id in enumerate(group_ids):
            group_id = self._normalize_group_id(raw_group_id)
            if group_id is None:
                raise ValueError(
                    "adv_normalization_scope='group' requires a non-empty group_id for every sample. "
                    f"Found invalid group_id at sample_idx={sample_idx}."
                )
            normalized.append(group_id)
        return normalized

    def _build_group_index_map(self, group_ids: List[str]) -> Dict[str, List[int]]:  # [HELPER → _normalize_group()]
        ordered_groups: Dict[str, List[int]] = {}
        for sample_idx, raw_group_id in enumerate(group_ids):
            group_id = self._normalize_group_id(raw_group_id)
            if group_id is None:
                continue
            ordered_groups.setdefault(group_id, []).append(sample_idx)
        return ordered_groups

    def _require_expected_group_sizes(
        self,
        group_index_map: Dict[str, List[int]],
    ) -> List[List[int]]:
        expected_group_size = max(1, int(self.samples_per_prompt))
        invalid_groups = [
            (group_id, len(indices))
            for group_id, indices in group_index_map.items()
            if len(indices) != expected_group_size
        ]
        if invalid_groups:
            formatted = ", ".join(
                f"{group_id!r}:{group_size}"
                for group_id, group_size in invalid_groups[:5]
            )
            if len(invalid_groups) > 5:
                formatted = f"{formatted}, ..."
            raise ValueError(
                "adv_normalization_scope='group' requires every sample group to contain exactly "
                "algorithm.samples_per_prompt samples. "
                f"Got algorithm.samples_per_prompt={expected_group_size}; "
                f"invalid group sizes: {formatted}."
            )
        return list(group_index_map.values())

    def _normalize_global(self, rewards: torch.Tensor) -> torch.Tensor:  # [HELPER → compute_advantages()]
        """Global normalization across all samples."""
        return normalize_global(rewards, epsilon=self.epsilon, clip_max=self.clip_max)

    def _normalize_group(  # [HELPER → compute_advantages()]
        self,
        rewards: torch.Tensor,
        group_ids: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Group normalization using explicit sample-group identities."""
        batch_size = rewards.shape[0]
        if group_ids is not None and len(group_ids) == batch_size:
            normalized_group_ids = self._require_valid_group_ids(group_ids)
            group_index_map = self._build_group_index_map(normalized_group_ids)
            groups = self._require_expected_group_sizes(group_index_map)
            if groups:
                return normalize_grouped(
                    rewards,
                    groups,
                    epsilon=self.epsilon,
                    clip_max=self.clip_max,
                    trim_outliers_ratio=self.trim_outliers_ratio,
                    use_global_std=self.use_global_std,
                )

        raise ValueError(
            "adv_normalization_scope='group' requires explicit group_ids aligned to the reward batch. "
            f"Got batch_size={batch_size}, "
            f"group_ids_len={len(group_ids) if group_ids is not None else None}."
        )

    # ------------------------------------------------------------------
    # Phase 3: Advantage orchestration with reward components
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_advantages_with_components(  # [PUBLIC-API → rollout.primitives.compute_advantages()] rollout side
        self,
        *,
        rewards: torch.Tensor,
        group_ids: Optional[List[str]] = None,
        reward_components: Optional[Dict[str, List[float]]] = None,
        reward_component_weights: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """Full advantage computation pipeline including reward component mixing.

        This consolidates the logic that was previously scattered in rollout
        workflow helpers into the algorithm itself.

        Args:
            rewards: Aggregated reward tensor [batch_size].
            reward_components: Per-component reward values, keyed by name.
            reward_component_weights: Per-component weights for aggregation.

        Returns:
            Advantage tensor [batch_size].
        """
        ...

    @abstractmethod
    def get_ema_spec(
        self,
    ) -> EMASpec:  # [PUBLIC-API → training_actor.init()] training side
        """Declare EMA policy for this algorithm."""
        ...

    def prepare_loss_advantages(  # [PUBLIC-API → train_executor] training side advantage transform
        self,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """Transform rollout advantages into the signal consumed by the loss.

        Default behavior is identity. Algorithms such as NFT can override this
        hook to map rollout advantages into loss-side weighting signals without
        mutating the rollout-visible advantage semantics.
        """
        return advantages

    # ------------------------------------------------------------------
    # Phase 2: Algorithm-owned training step
    # ------------------------------------------------------------------

    def _get_forward_plugin(self, model: nn.Module) -> object:
        """Return the model forward plugin, lazy-loading a default if needed.

        The plugin is normally injected by ``TrainingActor`` via
        ``model_bundle.forward_plugin()``.  If it was never set **and**
        ``model_type`` is ``"default"`` we fall back to
        :class:`~diffusionrl.models.forward_plugins.DefaultForwardPlugin`
        with a warning.
        """
        if self._forward_plugin is None:
            if self.model_type != "default":
                raise RuntimeError(
                    f"No forward_plugin set on {type(self).__name__} for "
                    f"model_type={self.model_type!r}. TrainingActor must "
                    "inject plugin via model_bundle.forward_plugin()."
                )
            from diffusionrl.models.forward_plugins import DefaultForwardPlugin
            self._forward_plugin = DefaultForwardPlugin()
            logger.warning(
                "No forward_plugin set on %s (model_type=%s). "
                "Using DefaultForwardPlugin. Set algorithm._forward_plugin "
                "from model_bundle.forward_plugin() for model-specific behavior.",
                type(self).__name__,
                self.model_type,
            )
        return self._forward_plugin

    def compute_loss_and_backward(  # [PUBLIC-API → train_executor] training side: core loss+backward
        self,
        *,
        model: nn.Module,
        batch: Any,
        timesteps: Any = None,
        loss_scale: float = 1.0,
        **kwargs: Any,
    ) -> tuple:
        """Compute loss and call backward for a single micro-batch.

        The micro-batch loop is owned by ``train_executor``; each algorithm
        only needs to handle one already-sliced micro-batch here.

        Args:
            model: The model to compute loss on.
            batch: A single micro-batch (already sliced by train_executor).
            timesteps: Pre-resolved training timesteps from
                ``resolve_training_timesteps()``.
            loss_scale: Gradient accumulation scale (1 / num_mini_batches).

        Returns:
            ``(loss, metrics, num_timesteps, has_backward)``
        """
        del model, batch, timesteps, loss_scale, kwargs
        raise NotImplementedError(
            f"{type(self).__name__} must implement compute_loss_and_backward()."
        )

    def resolve_training_timesteps(
        self,
        *,
        batch: Any,
        current_step: int,
        **kwargs: Any,
    ) -> Any:
        """Resolve algorithm-defined training timesteps once per update chunk.

        The returned object should represent actual training timesteps for the
        update. Algorithms that internally train by logical step index may map
        those timesteps back to indices inside ``compute_loss_and_backward``.
        """
        del batch, current_step, kwargs
        return None

    def is_forward_process(
        self,
    ) -> bool:  # [PUBLIC-API → training_actor] training side algorithm type
        """Whether this algorithm trains via forward process (NFT-style)."""
        return bool(self.get_sampling_requirements().is_forward_process)

    @abstractmethod
    def resolve_rollout_sde_indices(  # [PUBLIC-API → train.py rollout orchestration] rollout side
        self,
        *,
        current_step: int,
    ) -> Set[int]:
        """Resolve rollout-time SDE indices for the current step.

        Default behavior:
        - Forward-process algorithms do not use rollout SDE indices.
        - Trajectory algorithms implement their own current-step-driven policy.
        """
        ...

    @abstractmethod
    def get_sampler_validation_config(
        self, *, allow_replay: bool
    ) -> Dict[str, Any]:  # [PUBLIC-API → driver rollout pipeline] rollout side

        """Get sampler-output validation flags for rollout orchestration."""
        ...

    def assemble_training_batch(  # [PUBLIC-API → driver rollout pipeline] rollout side: assemble TrainingBatch
        self,
        *,
        request: RolloutRequest,
        sampler_outputs: List[Any],
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        sde_indices: Optional[Set[int]] = None,
    ) -> Any:
        """Assemble a unified TrainingBatch from sampler outputs.

        Branches on ``self.get_sampling_requirements()``:

        - ``requires_trajectory=True``: collects ``aux["trajectory_store"]`` and
          ``aux["log_probs"]``, resolves SDE training indices via
          ``resolve_training_indices`` / ``get_filtered_training_indices``,
          and builds ``TrajectoryStore.from_full()``.
        - ``requires_trajectory=False`` (NFT-style): collects
          ``output.latents`` (clean x0) and builds
          ``TrajectoryStore.from_clean_latents()``.

        Subclasses normally do **not** need to override this method.
        Customize training-index selection by overriding
        ``resolve_training_indices`` or ``get_filtered_training_indices``.
        """
        import time as _time
        from diffusionrl.types.sampling import RolloutSamples, LogProbData
        from diffusionrl.types.training_batch import TrainingBatch
        from diffusionrl.types.trajectory_store import TrajectoryStore

        reqs = self.get_sampling_requirements()
        prompts = request.prompts

        # -- shared collection across all outputs --------------------------
        all_forward_contexts = []
        timesteps = None
        trajectory_stores = []
        clean_latents = []
        log_probs_dicts = []
        step_indices = None
        scheduler_was_provided = sde_indices is not None
        raw_scheduler_indices = (
            {int(i) for i in sde_indices} if sde_indices is not None else None
        )
        final_sde_indices: Set[int] = set(raw_scheduler_indices or set())

        for idx, output in enumerate(sampler_outputs):
            if not isinstance(output, RolloutSamples):
                raise TypeError(
                    f"Assemble expects RolloutSamples, got "
                    f"{type(output).__name__} at index={idx}."
                )

            _fwd_ctx = output.aux.get("forward_context")
            if _fwd_ctx is None:
                raise ValueError(
                    f"RolloutSamples at index={idx} missing forward_context."
                )
            all_forward_contexts.append(_fwd_ctx)

            ts = output.timesteps
            if ts is not None and timesteps is None:
                timesteps = ts
            elif (
                ts is not None
                and timesteps is not None
                and not torch.equal(timesteps.to(ts.device), ts)
            ):
                raise ValueError("Mismatched timesteps across sampler outputs")

            if reqs.requires_trajectory:
                _traj_store = output.aux.get("trajectory_store")
                if _traj_store is None:
                    raise ValueError(
                        f"RolloutSamples at index={idx} missing trajectory_store."
                    )
                trajectory_stores.append(_traj_store)

                _log_probs = output.aux.get("log_probs")
                log_probs_dicts.append(
                    _log_probs.to_dict() if _log_probs is not None else {}
                )

                steps = output.aux.get("step_indices")
                if steps is not None:
                    if step_indices is None:
                        step_indices = steps
                    elif not torch.equal(
                        step_indices.to(steps.device), steps
                    ):
                        raise ValueError(
                            "Mismatched step_indices across sampler outputs: "
                            f"expected={step_indices.tolist()} "
                            f"got={steps.tolist()}"
                        )

                if sde_indices is None:
                    final_sde_indices.update(
                        int(i) for i in output.sde_indices
                    )
            else:
                clean_latents.append(output.latents)

        if timesteps is None:
            raise ValueError("No timesteps found in sampler outputs")

        # -- merge forward contexts ----------------------------------------
        merged_forward_context = type(all_forward_contexts[0]).cat(
            all_forward_contexts
        )

        # -- trajectory path (GRPO / MixGRPO) -----------------------------
        if reqs.requires_trajectory:
            if not trajectory_stores:
                raise ValueError("No trajectory stores found in sampler outputs")

            if step_indices is None:
                step_indices = torch.arange(
                    timesteps.shape[0],
                    device=timesteps.device,
                    dtype=torch.long,
                )

            step_labels = [int(v) for v in step_indices[:-1].tolist()]
            step_label_set = set(step_labels)

            def _normalize_to_step_labels(
                indices: Set[int], *, source: str
            ) -> Set[int]:
                if not indices:
                    return set()
                if indices.issubset(step_label_set):
                    return set(indices)
                mapped = {
                    step_labels[i]
                    for i in indices
                    if 0 <= int(i) < len(step_labels)
                }
                if mapped:
                    logger.debug(
                        "%s indices look positional; mapped to step "
                        "labels raw=%s mapped=%s",
                        source,
                        sorted(indices),
                        sorted(mapped),
                    )
                    return mapped
                logger.warning(
                    "%s indices do not match sampled step labels and "
                    "could not be mapped: raw=%s available=%s",
                    source,
                    sorted(indices),
                    sorted(step_labels),
                )
                return set()

            if final_sde_indices:
                final_sde_indices = _normalize_to_step_labels(
                    set(int(i) for i in final_sde_indices),
                    source="Scheduler/Sampler SDE",
                )

            raw_train_indices = self.resolve_training_indices(
                num_steps=len(step_labels),
                sde_indices=(
                    set(final_sde_indices) if scheduler_was_provided else None
                ),
            )
            train_indices = _normalize_to_step_labels(
                set(int(i) for i in raw_train_indices),
                source=f"{type(self).__name__}.resolve_training_indices",
            )
            if not train_indices:
                train_indices = step_label_set
            if not scheduler_was_provided:
                final_sde_indices = train_indices
            else:
                final_sde_indices = (
                    final_sde_indices & train_indices
                    if final_sde_indices
                    else train_indices
                )
            if not final_sde_indices:
                final_sde_indices = (
                    train_indices if train_indices else set(step_labels)
                )

            num_steps = len(step_labels)
            final_sde_indices = self.get_filtered_training_indices(
                final_sde_indices, num_steps
            )
            if not final_sde_indices:
                final_sde_indices = set(step_labels)

            merged_log_probs: Dict[int, torch.Tensor] = {}
            if log_probs_dicts:
                all_lp_indices: Set[int] = set()
                for lpd in log_probs_dicts:
                    all_lp_indices.update(lpd.keys())
                for idx_key in all_lp_indices:
                    values = [
                        lpd[idx_key]
                        for lpd in log_probs_dicts
                        if idx_key in lpd
                    ]
                    if values:
                        merged_log_probs[idx_key] = torch.cat(values, dim=0)
            if final_sde_indices:
                merged_log_probs = {
                    int(k): v
                    for k, v in merged_log_probs.items()
                    if int(k) in set(int(i) for i in final_sde_indices)
                }

            assemble_t0 = _time.perf_counter()

            traj_store = TrajectoryStore.concat(trajectory_stores)

            assemble_t1 = _time.perf_counter()

            # Clamp target_sde_indices to positions actually stored in the
            # trajectory.  Selective stores only keep SDE-boundary positions,
            # so training indices that fall outside (e.g. ODE-only steps)
            # must be dropped to avoid validation failures.
            if traj_store.is_selective and final_sde_indices:
                available = set(
                    idx for idx in final_sde_indices
                    if traj_store.has_position(idx) and traj_store.has_position(idx + 1)
                )
                if available != final_sde_indices:
                    logger.debug(
                        "Clamped target_sde_indices to trajectory store: "
                        "requested=%s available=%s stored=%s",
                        sorted(final_sde_indices),
                        sorted(available),
                        traj_store.stored_positions,
                    )
                    final_sde_indices = available
                if not final_sde_indices:
                    raise ValueError(
                        "No target_sde_indices remain after clamping to "
                        f"stored trajectory positions {traj_store.stored_positions}"
                    )

            batch = TrainingBatch(
                trajectory_store=traj_store,
                log_probs=LogProbData.from_dict(merged_log_probs),
                timesteps=timesteps,
                advantages=advantages,
                forward_context=merged_forward_context,
                rewards=rewards,
                prompts=prompts,
                step_indices=step_indices,
                target_sde_indices=set(
                    int(i) for i in final_sde_indices
                ),
            )
            batch.validate()

            traj_data = traj_store.data
            traj_gb = (
                traj_data.nelement()
                * traj_data.element_size()
                / 1e9
            )
            logger.debug(
                "[TIMING] assemble_training_batch(trajectory): "
                "cat=%.2fs total=%.2fs n=%d traj_gb=%.2f",
                assemble_t1 - assemble_t0,
                _time.perf_counter() - assemble_t0,
                len(trajectory_stores),
                traj_gb,
            )
            return batch

        # -- clean-latents path (NFT) --------------------------------------
        if not clean_latents:
            raise ValueError("No clean latents found in sampler outputs")

        clean_latents_tensor = torch.cat(clean_latents, dim=0)
        total_positions = int(timesteps.shape[0])
        batch = TrainingBatch(
            trajectory_store=TrajectoryStore.from_clean_latents(
                clean_latents_tensor, total_positions=total_positions,
            ),
            timesteps=timesteps,
            advantages=advantages,
            forward_context=merged_forward_context,
            rewards=rewards,
            prompts=prompts,
        )
        batch.validate()
        return batch

    def resolve_training_indices(
        self,
        *,
        num_steps: int,
        sde_indices: Optional[Set[int]] = None,
    ) -> Set[int]:
        """Resolve the timestep indices that should contribute to training.

        This is the explicit counterpart to rollout-time ``sde_indices``.
        """
        if sde_indices is not None:
            return set(int(i) for i in sde_indices)
        return set(range(num_steps))

    def get_filtered_training_indices(  # [PUBLIC-API → assemble_training_batch() internal] rollout side: filter training timesteps
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
        Default returns *sde_indices* unchanged.

        Args:
            sde_indices: Set of SDE timestep indices from scheduler
            num_steps: Total number of timesteps

        Returns:
            Filtered set of timestep indices for training
        """
        return sde_indices

    def get_config(self) -> Dict[str, Any]:  # [PUBLIC-API → serialization/logging]
        """Get algorithm configuration as dictionary."""
        return {
            "algorithm_type": self.__class__.__name__,
            "kl_coef": self.kl_coef,
            "component_mix_stage": self.component_mix_stage,
            "adv_normalization_scope": self.adv_normalization_scope,
            **self._extra_kwargs,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"kl_coef={self.kl_coef}, "
            f"component_mix_stage={self.component_mix_stage}, "
            f"adv_normalization_scope={self.adv_normalization_scope})"
        )
