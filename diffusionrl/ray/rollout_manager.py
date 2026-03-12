"""
diffusionrl Rollout Manager - Coordinates sampling, reward, and data conversion.

Supports:
- GRPO: Trajectory-based sampling with log probabilities
- MixGRPO: Mixed ODE/SDE sampling with timestep scheduler
- NFT: Forward process (only needs clean latents)

Reference: slime/ray/rollout.py
"""
import asyncio
import inspect
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import os
from tqdm import tqdm

import ray
import torch

from diffusionrl.config.arguments import is_training_actor_direct_sampling_mode
from diffusionrl.config.build_domain_args import RewardSchema
from diffusionrl.runtime.contracts import resolve_sampling_requirements
from diffusionrl.runtime.pipeline.rollout_pipeline import compute_advantages as _compute_advantages_stage
from diffusionrl.runtime.pipeline.rollout_pipeline import compute_rewards as _compute_rewards_stage
from diffusionrl.runtime.pipeline.rollout_pipeline import (
    distributed_sample,
    expand_batch_for_sampling,
)
from diffusionrl.runtime.rollout.training_actor_direct_sampling_policy import (
    TrainingActorDirectSamplingPolicy,
)
from diffusionrl.types.sampling import PromptEmbeddings, RolloutOutput
from diffusionrl.types.training_batch import TrainingBatch
from diffusionrl.utils import load_function

logger = logging.getLogger(__name__)

class _PipelineRolloutEngine:
    """Function-style rollout engine facade injected into custom pipelines."""

    def __init__(
        self,
        *,
        manager: Any,
        actor_group: Any,
        batch_template: Dict[str, Any],
        requirements: Any,
        default_sde_indices: Optional[Set[int]],
    ) -> None:
        self._manager = manager
        self._actor_group = actor_group
        self._batch_template = dict(batch_template)
        self._requirements = requirements
        self._default_sde_indices = set(default_sde_indices) if default_sde_indices is not None else None
        self.last_train_prompts: Optional[List[str]] = None
        self.last_base_prompts: Optional[List[str]] = None
        self.last_sde_indices: Optional[Set[int]] = set(default_sde_indices) if default_sde_indices is not None else None

    def generate(
        self,
        prompts: List[str],
        *,
        sde_indices: Optional[Set[int]] = None,
        sampling_overrides: Optional[Dict[str, Any]] = None,
        batch_overrides: Optional[Dict[str, Any]] = None,
    ) -> List[RolloutOutput]:
        if not isinstance(prompts, list) or len(prompts) == 0:
            raise ValueError("engine.generate() requires non-empty prompts list.")

        working_batch = dict(self._batch_template)
        working_batch["prompts"] = list(prompts)
        if isinstance(batch_overrides, dict) and batch_overrides:
            working_batch.update(batch_overrides)

        resolved_sde_indices: Optional[Set[int]]
        if sde_indices is not None:
            resolved_sde_indices = set(int(i) for i in sde_indices)
        elif self._default_sde_indices is not None:
            resolved_sde_indices = set(self._default_sde_indices)
        else:
            resolved_sde_indices = None

        outputs, train_prompts, base_prompts = self._manager._sample(
            actor_group=self._actor_group,
            batch=working_batch,
            sde_indices=resolved_sde_indices,
            requirements=self._requirements,
            sampling_overrides=sampling_overrides,
        )
        if bool(getattr(self._requirements, "requires_embeddings", True)):
            self._manager._attach_missing_embeddings_from_rollout_encoder(
                actor_group=self._actor_group,
                sampler_outputs=outputs,
                prompts=(train_prompts if train_prompts else list(prompts)),
                sampling_overrides=sampling_overrides,
            )
        self.last_train_prompts = train_prompts if train_prompts else list(prompts)
        self.last_base_prompts = base_prompts if base_prompts else list(prompts)
        self.last_sde_indices = resolved_sde_indices
        return outputs


def _resolve_custom_pipeline_result(value: Any) -> Any:
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)
    raise RuntimeError(
        "Custom rollout pipeline returned an awaitable while an event loop is already running. "
        "Please provide a synchronous pipeline or execute async work inside your function."
    )


def _invoke_custom_rollout_pipeline(
    pipeline_fn: Any,
    *,
    manager: Any,
    batch: Dict[str, Any],
    rollout_id: int,
    requirements: Any,
    actor_group: Any,
) -> Any:
    """Invoke custom rollout pipeline using function-style injected interfaces."""
    prompts = batch.get("prompts", []) or []
    if not isinstance(prompts, list) or len(prompts) == 0:
        raise ValueError(
            "Custom rollout pipeline requires non-empty text prompts in batch['prompts']."
        )

    default_sde_indices = manager.algorithm.resolve_rollout_sde_indices(
        timestep_scheduler=manager.timestep_scheduler,
        current_step=manager._current_step,
    )

    engine = _PipelineRolloutEngine(
        manager=manager,
        actor_group=actor_group,
        batch_template=batch,
        requirements=requirements,
        default_sde_indices=default_sde_indices,
    )

    def reward_fn(
        outputs: List[RolloutOutput],
        *,
        prompts_override: Optional[List[str]] = None,
        prompt_metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
        reward_path: Optional[str] = None,
        num_samples_per_prompt: Optional[int] = None,
    ):
        base_prompts = (
            prompts_override
            if prompts_override is not None
            else (engine.last_base_prompts if engine.last_base_prompts is not None else prompts)
        )
        metadata = prompt_metadata if prompt_metadata is not None else batch.get("metadata")
        return manager._compute_rewards_only(
            reward_service=manager.reward_service,
            sampler_outputs=outputs,
            prompts=base_prompts,
            prompt_metadata=metadata,
            reward_path_override=(
                str(reward_path)
                if reward_path is not None
                else None
            ),
            num_samples_per_prompt_override=num_samples_per_prompt,
        )

    def compute_advantages(
        rewards: torch.Tensor,
        *,
        prompts_override: Optional[List[str]] = None,
        num_samples_per_prompt: Optional[int] = None,
    ) -> torch.Tensor:
        adv_prompts = (
            prompts_override
            if prompts_override is not None
            else (engine.last_base_prompts if engine.last_base_prompts is not None else prompts)
        )
        return manager.algorithm.compute_advantages(
            rewards=rewards,
            num_samples_per_prompt=int(
                num_samples_per_prompt
                if num_samples_per_prompt is not None
                else getattr(manager.args.algorithm, "num_samples_per_prompt", 1)
            ),
            prompts=adv_prompts,
        )

    def reward_and_advantage_fn(
        outputs: List[RolloutOutput],
        *,
        prompts_override: Optional[List[str]] = None,
        prompt_metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
    ):
        base_prompts = (
            prompts_override
            if prompts_override is not None
            else (engine.last_base_prompts if engine.last_base_prompts is not None else prompts)
        )
        return manager._compute_reward_and_advantage(
            algorithm=manager.algorithm,
            reward_service=manager.reward_service,
            sampler_outputs=outputs,
            prompts=base_prompts,
            prompt_metadata=prompt_metadata if prompt_metadata is not None else batch.get("metadata"),
        )

    def assemble_batch(
        outputs: List[RolloutOutput],
        *,
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        prompts_override: Optional[List[str]] = None,
        sde_indices: Optional[Set[int]] = None,
    ) -> TrainingBatch:
        train_prompts = (
            prompts_override
            if prompts_override is not None
            else (engine.last_train_prompts if engine.last_train_prompts is not None else prompts)
        )
        resolved_sde = (
            set(int(i) for i in sde_indices)
            if sde_indices is not None
            else engine.last_sde_indices
        )
        return manager._assemble_training_batch(
            algorithm=manager.algorithm,
            sampler_outputs=outputs,
            rewards=rewards,
            advantages=advantages,
            prompts=train_prompts,
            sde_indices=resolved_sde,
        )

    call_kwargs: Dict[str, Any] = {
        "prompts": prompts,
        "engine": engine,
        "reward_fn": reward_fn,
        "compute_advantages": compute_advantages,
        "reward_and_advantage_fn": reward_and_advantage_fn,
        "assemble_batch": assemble_batch,
        "sampling_requirements": requirements,
        "batch": batch,
        "rollout_id": rollout_id,
    }

    return _resolve_custom_pipeline_result(pipeline_fn(**call_kwargs))


@ray.remote
class RolloutManager:
    """
    Rollout Manager - Coordinates the data generation pipeline.

    This manager handles:
    - Loading and managing algorithm, sampler, reward, and model components
    - Orchestrating the sampling -> reward -> advantage computation pipeline
    - Converting data to training format
    - Evaluation generation

    Key design principle (reference slime):
    - All components are dynamically loaded through args paths
    - Support for custom implementations via path overrides
    """

    def __init__(
        self,
        args,
        pg_result: Optional[Tuple] = None,
        reward_pg_result: Optional[Tuple] = None,
    ):
        """
        Initialize RolloutManager.

        Args:
            args: TrainingArguments instance
            pg_result: Optional placement group result for rollout (pg, bundle_indices, gpu_ids)
            reward_pg_result: Optional placement group result for reward (pg, bundle_indices, gpu_ids)
        """
        self.args = args
        self.pg_result = pg_result
        self.reward_pg_result = reward_pg_result
        self._is_initialized = False

        # Components (loaded in init())
        self.algorithm = None
        self.reward_service = None
        self.data_source = None

        # Timestep scheduler for MixGRPO
        self.timestep_scheduler = None
        self.rollout_pipeline_fn = None
        self._warned_ignored_prompt_embeddings = False
        self._warned_missing_prompt_encoder_rpc = False
        self._warned_prompt_encode_fallback_failed = False
        self._reward_schema = RewardSchema.from_args(args)

        # Rollout actor group
        self.rollout_actors = None
        self.external_sampling_actors = None
        self._owns_rollout_actors = False

        # Stats
        self._total_samples_generated = 0
        self._current_step = 0
        self._sampling_requirements = None
        self._last_rollout_metadata: Dict[str, Any] = {}
        self._direct_sampling_request_planner = TrainingActorDirectSamplingPolicy.from_args(args)

    def init(self) -> None:
        """
        Initialize all components via dynamic loading.

        Components are loaded using load_function with paths from args.
        """
        logger.info("Initializing RolloutManager...")
        self._validate_batch_shape()

        # 1. Load algorithm
        algorithm_cls = load_function(self.args.algorithm.algorithm_path)
        if not hasattr(algorithm_cls, "from_args"):
            raise TypeError(
                f"Algorithm {self.args.algorithm.algorithm_path} must implement classmethod from_args(args)."
            )
        self.algorithm = algorithm_cls.from_args(self.args)
        self._sampling_requirements = self._resolve_sampling_requirements()
        logger.info(
            f"Algorithm loaded: {self.args.algorithm.algorithm_path} "
            f"(clip_max={self.args.algorithm.advantage_clip_max}, sde_ratio={getattr(self.args.sampling, 'sde_ratio', 'N/A')})"
        )
        requests_per_rollout = self._direct_sampling_request_planner.requests_per_rollout(
            rollout_total_samples=self._generated_samples_per_rollout()
        )
        if requests_per_rollout > 1:
            logger.info(
                "Training-actor direct sampling sub-batching enabled: "
                "direct_sampling_batch_size=%s sampling_requests_per_rollout=%s "
                "(prompts_per_rollout=%s generated_samples_per_rollout=%s)",
                getattr(self._direct_sampling_request_planner, "direct_sampling_batch_size", None),
                requests_per_rollout,
                self._prompts_per_batch(),
                self._generated_samples_per_rollout(),
            )
        logger.info(
            "Resolved sampling contract: requires_trajectory=%s requires_log_prob=%s "
            "requires_embeddings=%s extras=%s",
            bool(self._sampling_requirements.requires_trajectory),
            bool(self._sampling_requirements.requires_log_prob),
            bool(self._sampling_requirements.requires_embeddings),
            dict(getattr(self._sampling_requirements, "extras", {}) or {}),
        )
        use_per_prompt_tracker = getattr(self.args.algorithm, "use_per_prompt_stat_tracker", False)
        if getattr(self.args.algorithm, "advantage_type", "group") == "per_prompt" and not use_per_prompt_tracker:
            logger.warning("advantage_type=per_prompt but use_per_prompt_stat_tracker=False; advantages will fall back to group normalization.")

        # 2. Load timestep scheduler based on algorithm requirements
        self._init_timestep_scheduler()

        # Optional custom rollout pipeline (slime-style pluggable generate function)
        rollout_pipeline_path = getattr(self.args.rollout, "rollout_pipeline_path", None)
        if rollout_pipeline_path:
            self.rollout_pipeline_fn = load_function(rollout_pipeline_path)
            logger.info("Custom rollout pipeline loaded: %s", rollout_pipeline_path)
            if requests_per_rollout > 1:
                logger.warning(
                    "sampling.direct_sampling_batch_size=%s is not automatically applied inside custom rollout pipelines. "
                    "Custom pipeline code must opt in explicitly if it wants direct-sampling sub-batching.",
                    getattr(self._direct_sampling_request_planner, "direct_sampling_batch_size", None),
                )

        # 3. Initialize reward service
        self._init_reward_service()
        if self.reward_service is None:
            if self._reward_schema.uses_rollout_execution:
                logger.info("Reward service skipped: rollout-local reward execution is active.")
            else:
                raise RuntimeError("RewardService initialization failed.")
        else:
            logger.info(f"Reward service loaded with {len(self.reward_service.workers)} worker(s)")

        # 4. Load data source if available
        try:
            data_source_cls = load_function(self.args.data_source_path)
            self.data_source = data_source_cls(self.args)
            logger.info(f"Data source loaded: {self.args.data_source_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load data source: {e}") from e

        # 5. Create rollout actors if placement group provided
        if self.pg_result is not None and not is_training_actor_direct_sampling_mode(self.args):
            self._create_rollout_actors()

        self._is_initialized = True
        logger.info("RolloutManager initialized")

    def _init_reward_service(self) -> None:
        """Initialize reward service (single reward boundary)."""
        if self._reward_schema.uses_rollout_execution:
            self.reward_service = None
            return
        from diffusionrl.reward.service import RewardService
        self.reward_service = RewardService(
            args=self.args,
            reward_pg_result=self.reward_pg_result,
        )

    def _init_timestep_scheduler(self) -> None:
        """Initialize timestep scheduler based on algorithm requirements and config."""
        from diffusionrl.samplers.schedulers import get_scheduler

        requirements = self._sampling_requirements
        if requirements is None:
            requirements = self._resolve_sampling_requirements()
            self._sampling_requirements = requirements

        # Get scheduler config from args
        scheduler_type = getattr(self.args.algorithm.window, "timestep_strategy", "all")
        num_timesteps = self.args.sampling.num_inference_steps
        timestep_fraction = getattr(self.args.sampling, "timestep_fraction", 1.0)

        # Check if we should use sde_ratio to determine scheduler
        # If sde_ratio < 1.0 and no explicit strategy, infer window scheduler
        if scheduler_type == 'all' and requirements.sde_ratio < 1.0:
            # Auto-configure window scheduler based on sde_ratio
            # sde_ratio = fraction of timesteps that use SDE
            # group_size = num_timesteps * sde_ratio (rounded)
            group_size = max(1, int(num_timesteps * requirements.sde_ratio))
            scheduler_type = 'window'
            logger.info(f"Auto-configured window scheduler from sde_ratio={requirements.sde_ratio}")

        if scheduler_type == 'window':
            # MixGRPO window scheduler
            # Use explicit group_size from args if provided, otherwise from sde_ratio
            explicit_group_size = getattr(self.args.algorithm.window, "window_group_size", None)
            if explicit_group_size is None and requirements.sde_ratio < 1.0:
                group_size = max(1, int(num_timesteps * requirements.sde_ratio))
            else:
                group_size = explicit_group_size or 4

            self.timestep_scheduler = get_scheduler(
                scheduler_type='window',
                num_timesteps=num_timesteps,
                timestep_fraction=timestep_fraction,
                strategy=getattr(self.args.algorithm.window, "window_strategy", "progressive"),
                group_size=group_size,
                iters_per_group=getattr(self.args.algorithm.window, "window_iters_per_group", 25),
                max_iters_per_group=getattr(self.args.algorithm.window, "window_max_iters_per_group", None),
                min_iters_per_group=getattr(self.args.algorithm.window, "window_min_iters_per_group", None),
                overlap=getattr(self.args.algorithm.window, "window_overlap", False),
                overlap_step=getattr(self.args.algorithm.window, "window_overlap_step", 1),
                roll_back=getattr(self.args.algorithm.window, "window_roll_back", False),
            )
            logger.info(f"Window scheduler initialized: group_size={group_size}")
        else:
            # Default: all SDE for sampling (standard GRPO).
            # For DanceGRPO/FlowGRPO, timestep_fraction should affect SDE indices directly.
            self.timestep_scheduler = get_scheduler(
                scheduler_type='all',
                num_timesteps=num_timesteps,
                timestep_fraction=timestep_fraction,
            )
            # Determine whether timestep_fraction restricts the SDE range
            from diffusionrl.samplers.schedulers.timestep_window import _normalize_timestep_fraction
            frac_start, frac_end = _normalize_timestep_fraction(timestep_fraction)
            if frac_start > 0.0 or frac_end < 1.0:
                eff_start = int(num_timesteps * frac_start)
                eff_end = int(num_timesteps * frac_end)
                logger.info(
                    "All SDE scheduler initialized; "
                    f"timestep_fraction={timestep_fraction} "
                    f"(SDE on timesteps [{eff_start}, {eff_end})/{num_timesteps})"
                )
            else:
                logger.info("All SDE scheduler initialized (standard GRPO)")

    def _create_rollout_actors(self) -> None:
        """Create rollout actor group from placement group."""
        from .group_factory import create_rollout_actor_group

        self.rollout_actors = create_rollout_actor_group(self.args, self.pg_result)
        self._owns_rollout_actors = True
        logger.info("Rollout actors created via create_rollout_actor_group")

    def _resolve_sampling_requirements(self):
        """Resolve final sampling contract (loss requires_* + algorithm extras)."""
        algorithm_requirements = self.algorithm.get_sampling_requirements()
        return resolve_sampling_requirements(
            self.args,
            algorithm_requirements=algorithm_requirements,
        )

    def attach_sampling_actors(self, actor_group) -> None:
        """Attach external sampling actors (e.g., TrainingActorGroup)."""
        self.external_sampling_actors = actor_group
        logger.info("External sampling actors attached")

    def _build_training_batch(
        self,
        rollout_id: int,
        *,
        collect_media_preview: bool = False,
        media_max_items: int = 8,
    ) -> TrainingBatch:
        """Run rollout pipeline and return one typed TrainingBatch."""
        if not self._is_initialized:
            raise RuntimeError("RolloutManager not initialized. Call init() first.")

        self._last_rollout_metadata = {}
        logger.info(f"Starting generation for rollout {rollout_id}")

        # 1. Get batch from data source (prompt-only external input contract)
        batch = self._prepare_batch(data_source=self.data_source)
        prompts = batch.get("prompts", [])

        # 2. Get algorithm requirements to determine pipeline
        requirements = self._sampling_requirements
        if requirements is None:
            requirements = self._resolve_sampling_requirements()
            self._sampling_requirements = requirements
        actor_group = self.external_sampling_actors or self.rollout_actors

        train_data: Any
        if self.rollout_pipeline_fn is not None:
            train_data = _invoke_custom_rollout_pipeline(
                self.rollout_pipeline_fn,
                batch=batch,
                manager=self,
                rollout_id=rollout_id,
                requirements=requirements,
                actor_group=actor_group,
            )
        else:
            train_data = self._generate_training_data(
                batch=batch,
                rollout_id=rollout_id,
                requirements=requirements,
                actor_group=actor_group,
                collect_media_preview=collect_media_preview,
                media_max_items=media_max_items,
            )

        if not hasattr(train_data, "slice"):
            raise TypeError(
                "Rollout pipeline must return a TrainingBatch-like object exposing slice(). "
                f"Got type={type(train_data)}"
            )

        self._finalize_rollout_state(train_data=train_data, prompts=prompts)

        return train_data

    def build_training_batch(self, rollout_id: int) -> TrainingBatch:
        """Public rollout entrypoint used by RolloutBufferActor.request_rollout()."""
        return self._build_training_batch(rollout_id)

    def _advance_rollout_state(
        self,
        *,
        sample_count: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Shared rollout-state update used by manager-owned and external workflows."""
        if metadata is not None:
            self._last_rollout_metadata = dict(metadata)
        self._total_samples_generated += max(0, int(sample_count))
        self._current_step += 1
        if self.timestep_scheduler is not None:
            self.timestep_scheduler.update(self._current_step)
        return {
            "current_step": int(self._current_step),
            "total_samples_generated": int(self._total_samples_generated),
            "metadata": dict(self._last_rollout_metadata),
        }

    def _finalize_rollout_state(self, *, train_data: TrainingBatch, prompts: List[str]) -> None:
        """Update counters/scheduler after one rollout payload is produced."""
        try:
            sample_count = None
            if hasattr(train_data, "rewards") and train_data.rewards is not None:
                sample_count = int(train_data.rewards.shape[0])
            elif hasattr(train_data, "batch_size"):
                sample_count = int(train_data.batch_size)
            else:
                sample_count = len(prompts)
            self._advance_rollout_state(sample_count=sample_count)
        except Exception as e:
            logger.warning(f"Failed to compute sample count ({e}), falling back to prompt count")
            self._advance_rollout_state(sample_count=len(prompts))

    def _debug_log_tensor_stats(self, label: str, value: Optional[torch.Tensor]) -> None:
        if not bool(getattr(self.args.debug, "debug_print_tensor_stats", True)):
            return
        if not torch.is_tensor(value):
            return
        if value.numel() == 0:
            logger.info("[debug] %s: shape=%s empty", label, tuple(value.shape))
            return
        flat = value.detach().to(dtype=torch.float32).reshape(-1)
        logger.info(
            "[debug] %s: shape=%s mean=%.6f std=%.6f min=%.6f max=%.6f",
            label,
            tuple(value.shape),
            float(flat.mean().item()),
            float(flat.std(unbiased=False).item()),
            float(flat.min().item()),
            float(flat.max().item()),
        )

    def _build_reward_prompts(
        self,
        *,
        train_prompts: Optional[List[str]],
        base_prompts: Optional[List[str]],
        fallback_prompts: List[str],
        sample_count: int,
    ) -> List[str]:
        if train_prompts:
            return list(train_prompts)[:sample_count]
        candidate = list(base_prompts or fallback_prompts)
        if not candidate:
            return []
        expanded: List[str] = []
        while len(expanded) < sample_count:
            expanded.extend(candidate)
        return expanded[:sample_count]

    def _build_wandb_media_preview(
        self,
        *,
        sampler_outputs: List[Any],
        reward_prompts: List[str],
        rewards: torch.Tensor,
        max_items: int,
    ) -> Optional[Dict[str, Any]]:
        limit = max(1, int(max_items))
        rewards_flat: List[float] = []
        if torch.is_tensor(rewards) and rewards.numel() > 0:
            rewards_flat = [float(v) for v in rewards.detach().cpu().reshape(-1).tolist()]

        images: List[Any] = []
        prompts: List[str] = []
        reward_values: List[float] = []
        global_sample_idx = 0

        for output in sampler_outputs:
            batch_size = int(getattr(output, "batch_size", 0) or 0)
            decoded_images = list(getattr(output, "decoded_images", None) or [])
            for image_idx, image in enumerate(decoded_images):
                if len(images) >= limit:
                    break
                if not hasattr(image, "save"):
                    continue
                sample_idx = global_sample_idx + image_idx
                images.append(image)
                prompt = reward_prompts[sample_idx] if sample_idx < len(reward_prompts) else ""
                prompts.append(str(prompt))
                reward_val = rewards_flat[sample_idx] if sample_idx < len(rewards_flat) else 0.0
                reward_values.append(float(reward_val))
            if len(images) >= limit:
                break
            global_sample_idx += batch_size

        if not images:
            return None

        return {
            "images": images,
            "prompts": prompts,
            "rewards": reward_values,
        }

    def _resolve_rollout_sde_indices(
        self,
        *,
        rollout_id: Optional[int] = None,
        log_debug: bool = False,
    ) -> Optional[Set[int]]:
        sde_indices = self.algorithm.resolve_rollout_sde_indices(
            timestep_scheduler=self.timestep_scheduler,
            current_step=self._current_step,
        )
        if log_debug and rollout_id is not None and sde_indices is not None:
            logger.info(
                "[debug] rollout=%s step=%s sampled_sde_count=%s",
                rollout_id,
                self._current_step,
                len(sde_indices),
            )
        return sde_indices

    def _get_sampler_validation_config(self) -> Dict[str, Any]:
        config = self.algorithm.get_sampler_validation_config(args=self.args)
        if not isinstance(config, dict):
            config = {}
        return {
            "allow_replay": bool(config.get("allow_replay", False)),
            "assert_step_alignment": bool(config.get("assert_step_alignment", True)),
            "mode_label": str(config.get("mode_label", "trajectory")),
        }

    def _generate_training_data_debug(
        self,
        *,
        batch: Dict[str, Any],
        rollout_id: int,
        requirements: Any,
        actor_group: Any,
    ) -> Tuple[TrainingBatch, Dict[str, Any]]:
        """Generate one rollout batch with detailed intermediate payload."""
        prompts = batch.get("prompts", []) or []
        sde_indices = self._resolve_rollout_sde_indices(
            rollout_id=rollout_id,
            log_debug=True,
        )

        logger.info(
            "[debug] rollout=%s stage=sampling prompts=%s num_samples_per_prompt=%s",
            rollout_id,
            len(prompts),
            int(getattr(self.args.algorithm, "num_samples_per_prompt", 1)),
        )
        sampler_outputs, train_prompts, base_prompts, prompt_metadata = (
            self._sample_rollout_batches(
                actor_group=actor_group,
                batch=batch,
                sde_indices=sde_indices,
                requirements=requirements,
                rollout_id=rollout_id,
                sampling_overrides={"_keep_reward_media_for_manager": True},
            )
        )
        logger.info(
            "[debug] rollout=%s stage=sampling done outputs=%s",
            rollout_id,
            len(sampler_outputs),
        )

        logger.info("[debug] rollout=%s stage=reward_advantage", rollout_id)
        rewards, advantages, reward_components = self._compute_reward_and_advantage(
            algorithm=self.algorithm,
            reward_service=self.reward_service,
            sampler_outputs=sampler_outputs,
            prompts=base_prompts if base_prompts else prompts,
            prompt_metadata=prompt_metadata,
        )
        self._debug_log_tensor_stats("rewards", rewards)
        self._debug_log_tensor_stats("advantages", advantages)

        logger.info("[debug] rollout=%s stage=assemble", rollout_id)
        training_batch = self._assemble_training_batch(
            algorithm=self.algorithm,
            sampler_outputs=sampler_outputs,
            rewards=rewards,
            advantages=advantages,
            prompts=train_prompts if train_prompts else prompts,
            sde_indices=sde_indices,
        )
        logger.info(
            "[debug] rollout=%s stage=assemble done batch_type=%s batch_size=%s",
            rollout_id,
            type(training_batch).__name__,
            int(getattr(training_batch, "batch_size", 0)),
        )

        reward_prompts = self._build_reward_prompts(
            train_prompts=train_prompts,
            base_prompts=base_prompts,
            fallback_prompts=prompts,
            sample_count=int(rewards.shape[0]),
        )

        trace_payload = {
            "rollout_id": int(rollout_id),
            "debug_mode": str(getattr(self.args.debug, "debug_mode", "none")),
            "prompts": list(base_prompts if base_prompts else prompts),
            "train_prompts": list(train_prompts if train_prompts else prompts),
            "base_prompts": list(base_prompts if base_prompts else prompts),
            "reward_prompts": reward_prompts,
            "sde_indices": sorted(int(v) for v in (sde_indices or [])),
            "sampler_outputs": sampler_outputs,
            "rewards": rewards,
            "advantages": advantages,
            "reward_components": reward_components,
            "training_batch": training_batch,
        }
        return training_batch, trace_payload

    def _build_training_debug_payload(self, rollout_id: int) -> Dict[str, Any]:
        """Run rollout pipeline and return a debug payload with intermediates."""
        if not self._is_initialized:
            raise RuntimeError("RolloutManager not initialized. Call init() first.")

        batch = self._prepare_batch(data_source=self.data_source)
        prompts = batch.get("prompts", [])
        requirements = self._sampling_requirements
        if requirements is None:
            requirements = self._resolve_sampling_requirements()
            self._sampling_requirements = requirements
        actor_group = self.external_sampling_actors or self.rollout_actors

        if self.rollout_pipeline_fn is not None:
            logger.warning(
                "Custom rollout pipeline detected in debug payload path. "
                "Intermediate sampler/reward breakdown may be incomplete."
            )
            train_data = _invoke_custom_rollout_pipeline(
                self.rollout_pipeline_fn,
                batch=batch,
                manager=self,
                rollout_id=rollout_id,
                requirements=requirements,
                actor_group=actor_group,
            )
            if not hasattr(train_data, "slice"):
                raise TypeError(
                    "Rollout pipeline must return a TrainingBatch-like object exposing slice(). "
                    f"Got type={type(train_data)}"
                )
            payload = {
                "rollout_id": int(rollout_id),
                "debug_mode": str(getattr(self.args.debug, "debug_mode", "none")),
                "prompts": list(prompts),
                "train_prompts": list(prompts),
                "base_prompts": list(prompts),
                "reward_prompts": list(prompts),
                "sde_indices": [],
                "sampler_outputs": [],
                "rewards": getattr(train_data, "rewards", None),
                "advantages": getattr(train_data, "advantages", None),
                "reward_components": {},
                "training_batch": train_data,
                "custom_rollout_pipeline": str(getattr(self.args.rollout, "rollout_pipeline_path", "")),
            }
        else:
            train_data, payload = self._generate_training_data_debug(
                batch=batch,
                rollout_id=rollout_id,
                requirements=requirements,
                actor_group=actor_group,
            )

        self._finalize_rollout_state(train_data=train_data, prompts=prompts)
        return payload

    def build_training_debug_payload(self, rollout_id: int) -> Dict[str, Any]:
        """Public debug rollout entrypoint with full intermediate payload."""
        return self._build_training_debug_payload(rollout_id)

    # ------------------------------------------------------------------
    # Interactive debug stage methods (debug_mode=interactive)
    #
    # These methods execute individual pipeline stages and cache intermediate
    # results inside the actor to avoid serializing large tensors back to the
    # driver between stages.  The driver only receives lightweight summaries.
    # ------------------------------------------------------------------

    def _debug_cache_clear(self) -> None:
        """Reset the interactive-debug intermediate cache."""
        self._debug_cache: Dict[str, Any] = {}

    def debug_sample(self, rollout_id: int) -> Dict[str, Any]:
        """Stage 1: execute sampling, cache results in actor. Returns lightweight summary."""
        if not self._is_initialized:
            raise RuntimeError("RolloutManager not initialized. Call init() first.")

        batch = self._prepare_batch(data_source=self.data_source)
        prompts = batch.get("prompts", [])
        requirements = self._sampling_requirements
        if requirements is None:
            requirements = self._resolve_sampling_requirements()
            self._sampling_requirements = requirements
        actor_group = self.external_sampling_actors or self.rollout_actors

        sde_indices = self._resolve_rollout_sde_indices()

        sampler_outputs, train_prompts, base_prompts, prompt_metadata = self._sample_rollout_batches(
            actor_group=actor_group,
            batch=batch,
            sde_indices=sde_indices,
            requirements=requirements,
            rollout_id=rollout_id,
        )

        self._debug_cache = {
            "rollout_id": rollout_id,
            "batch": batch,
            "prompts": base_prompts if base_prompts else prompts,
            "train_prompts": train_prompts if train_prompts else prompts,
            "base_prompts": base_prompts if base_prompts else prompts,
            "prompt_metadata": prompt_metadata,
            "sampler_outputs": sampler_outputs,
            "sde_indices": sde_indices,
            "requirements": requirements,
        }

        return {
            "rollout_id": rollout_id,
            "num_outputs": len(sampler_outputs),
            "total_samples": sum(int(getattr(o, "batch_size", 1)) for o in sampler_outputs),
            "prompts": base_prompts if base_prompts else prompts,
            "sde_indices": sorted(int(v) for v in (sde_indices or [])),
        }

    def debug_rewards(self) -> Dict[str, Any]:
        """Stage 2: compute rewards from cached sampler_outputs. Returns summary."""
        cache = getattr(self, "_debug_cache", None)
        if not cache or "sampler_outputs" not in cache:
            raise RuntimeError("No cached sampler_outputs. Run debug_sample first.")

        rewards, comps = self._compute_rewards_only(
            reward_service=self.reward_service,
            sampler_outputs=cache["sampler_outputs"],
            prompts=cache["base_prompts"],
            prompt_metadata=cache.get("prompt_metadata"),
        )
        cache["rewards"] = rewards
        cache["reward_components"] = comps
        self._debug_log_tensor_stats("rewards", rewards)

        flat = rewards.detach().to(dtype=torch.float32).reshape(-1)
        return {
            "rewards_mean": float(flat.mean().item()),
            "rewards_std": float(flat.std(unbiased=False).item()),
            "rewards_min": float(flat.min().item()),
            "rewards_max": float(flat.max().item()),
            "num_samples": int(flat.numel()),
            "component_names": sorted(comps.keys()),
        }

    def debug_advantages(self) -> Dict[str, Any]:
        """Stage 3: compute advantages from cached rewards. Returns summary."""
        cache = getattr(self, "_debug_cache", None)
        if not cache or "rewards" not in cache:
            raise RuntimeError("No cached rewards. Run debug_rewards first.")

        advantages = _compute_advantages_stage(
            algorithm=self.algorithm,
            num_samples_per_prompt=int(getattr(self.args.algorithm, "num_samples_per_prompt", 1)),
            reward_mix_mode=str(getattr(self.args.reward, "reward_mix_mode", "reward_aggr")),
            rewards=cache["rewards"],
            prompts=cache["base_prompts"],
            reward_components=cache.get("reward_components", {}),
            reward_workers=getattr(self.reward_service, "workers", None) if self.reward_service is not None else None,
            reward_component_weights=self._reward_schema.component_weights(),
        )
        cache["advantages"] = advantages
        self._debug_log_tensor_stats("advantages", advantages)

        flat = advantages.detach().to(dtype=torch.float32).reshape(-1)
        return {
            "advantages_mean": float(flat.mean().item()),
            "advantages_std": float(flat.std(unbiased=False).item()),
            "advantages_min": float(flat.min().item()),
            "advantages_max": float(flat.max().item()),
            "num_samples": int(flat.numel()),
        }

    def debug_assemble(self) -> Dict[str, Any]:
        """Stage 4: assemble training batch from cached intermediates. Returns summary."""
        cache = getattr(self, "_debug_cache", None)
        if not cache or "advantages" not in cache:
            raise RuntimeError("No cached advantages. Run debug_advantages first.")

        training_batch = self._assemble_training_batch(
            algorithm=self.algorithm,
            sampler_outputs=cache["sampler_outputs"],
            rewards=cache["rewards"],
            advantages=cache["advantages"],
            prompts=cache["train_prompts"],
            sde_indices=cache["sde_indices"],
        )
        cache["training_batch"] = training_batch

        return {
            "batch_type": type(training_batch).__name__,
            "batch_size": int(getattr(training_batch, "batch_size", 0)),
        }

    def debug_fetch_payload(self) -> Dict[str, Any]:
        """Transfer cached intermediates back to driver (large; call only when saving)."""
        cache = getattr(self, "_debug_cache", None)
        if not cache:
            raise RuntimeError("No debug cache available. Run debug_sample first.")

        sample_count = 0
        if cache.get("rewards") is not None and torch.is_tensor(cache["rewards"]):
            sample_count = int(cache["rewards"].shape[0])
        reward_prompts = self._build_reward_prompts(
            train_prompts=cache.get("train_prompts"),
            base_prompts=cache.get("base_prompts"),
            fallback_prompts=cache.get("prompts", []),
            sample_count=sample_count,
        )
        return {
            "rollout_id": cache.get("rollout_id", 0),
            "prompts": cache.get("prompts", []),
            "train_prompts": cache.get("train_prompts", []),
            "base_prompts": cache.get("base_prompts", []),
            "reward_prompts": reward_prompts,
            "sde_indices": sorted(int(v) for v in (cache.get("sde_indices") or [])),
            "sampler_outputs": cache.get("sampler_outputs", []),
            "rewards": cache.get("rewards"),
            "advantages": cache.get("advantages"),
            "reward_components": cache.get("reward_components", {}),
            "training_batch": cache.get("training_batch"),
            "debug_mode": "interactive",
        }

    def _validate_batch_shape(self) -> None:
        """Validate rollout-generated sample count against training batch/world size."""
        try:
            gen_batch = self._generated_samples_per_rollout()

            train_world_size = self.args.ray.training_num_nodes * self.args.ray.training_num_gpus_per_node
            if train_world_size <= 0:
                return

            if gen_batch % train_world_size != 0:
                logger.warning(
                    "Rollout-generated sample count (%d) is not divisible by train_world_size (%d). "
                    "This will produce uneven local training batches. Consider adjusting prompts_per_batch, "
                    "num_samples_per_prompt, or the training actor topology.",
                    gen_batch,
                    train_world_size,
                )
                return

            local_batch_size = gen_batch // train_world_size
            gradient_accumulation_batch_size = int(self.args.training.gradient_accumulation_batch_size)
            update_mode = str(self.args.training.update_mode or "single_update").strip().lower()

            if update_mode == "multi_update":
                multi_update_batch_size = int(self.args.training.multi_update_batch_size)
                if local_batch_size % multi_update_batch_size != 0:
                    logger.warning(
                        "Local rollout batch size (%d) is not divisible by multi_update_batch_size (%d). "
                        "Consider adjusting training.multi_update_batch_size or rollout batch geometry.",
                        local_batch_size,
                        multi_update_batch_size,
                    )
                if multi_update_batch_size % gradient_accumulation_batch_size != 0:
                    logger.warning(
                        "multi_update_batch_size (%d) is not divisible by gradient_accumulation_batch_size (%d). "
                        "Consider adjusting training.gradient_accumulation_batch_size or "
                        "training.multi_update_batch_size.",
                        multi_update_batch_size,
                        gradient_accumulation_batch_size,
                    )
            elif local_batch_size % gradient_accumulation_batch_size != 0:
                logger.warning(
                    "Local rollout batch size (%d) is not divisible by gradient_accumulation_batch_size (%d). "
                    "Consider adjusting training.gradient_accumulation_batch_size or rollout batch geometry.",
                    local_batch_size,
                    gradient_accumulation_batch_size,
                )
        except Exception as e:
            logger.warning(f"Batch shape validation skipped: {e}")

    def _prompts_per_batch(self) -> int:
        prompts_per_batch = getattr(self.args.algorithm, "prompts_per_batch", None)
        if prompts_per_batch is None:
            raise ValueError("algorithm.prompts_per_batch must be set explicitly.")
        return max(1, int(prompts_per_batch))

    def _generated_samples_per_rollout(self) -> int:
        num_samples_per_prompt = int(getattr(self.args.algorithm, "num_samples_per_prompt", 1))
        return max(1, self._prompts_per_batch() * max(1, num_samples_per_prompt))

    def _generate_training_data(
        self,
        batch: Dict[str, Any],
        rollout_id: int,
        requirements: Any,
        actor_group: Any,
        *,
        collect_media_preview: bool = False,
        media_max_items: int = 8,
    ) -> TrainingBatch:
        """Unified training data generation for all algorithm types.

        Algorithm-specific behavior (forward vs trajectory, replay/alignment policy)
        is delegated to algorithm strategy hooks.

        Args:
            batch: Dict containing prompts (and optionally embeddings/metadata)
            rollout_id: Current rollout iteration number
            requirements: Algorithm sampling requirements
            actor_group: Actor group for distributed sampling
        """
        prompts = batch.get("prompts", []) or []
        sde_indices = self._resolve_rollout_sde_indices()
        if sde_indices is not None:
            logger.debug(f"SDE indices for step {self._current_step}: {sorted(sde_indices)[:5]}...")

        sampling_overrides: Dict[str, Any] = {
            "_keep_reward_media_for_manager": bool(collect_media_preview),
        }

        sampler_outputs, train_prompts, base_prompts, prompt_metadata = self._sample_rollout_batches(
            actor_group=actor_group,
            batch=batch,
            sde_indices=sde_indices,
            requirements=requirements,
            rollout_id=rollout_id,
            sampling_overrides=sampling_overrides,
        )

        # Reward + advantage
        rewards, advantages, _ = self._compute_reward_and_advantage(
            algorithm=self.algorithm,
            reward_service=self.reward_service,
            sampler_outputs=sampler_outputs,
            prompts=base_prompts if base_prompts else prompts,
            prompt_metadata=prompt_metadata,
        )

        if collect_media_preview:
            reward_prompts = self._build_reward_prompts(
                train_prompts=train_prompts,
                base_prompts=base_prompts,
                fallback_prompts=prompts,
                sample_count=int(rewards.shape[0]),
            )
            media_preview = self._build_wandb_media_preview(
                sampler_outputs=sampler_outputs,
                reward_prompts=reward_prompts,
                rewards=rewards,
                max_items=media_max_items,
            )
            if media_preview is not None:
                self._last_rollout_metadata["wandb_media_preview"] = media_preview

        # Assemble
        return self._assemble_training_batch(
            algorithm=self.algorithm,
            sampler_outputs=sampler_outputs,
            rewards=rewards,
            advantages=advantages,
            prompts=train_prompts if train_prompts else prompts,
            sde_indices=sde_indices,
        )

    # TODO(refactor): Move this post-sampling embedding-fallback cluster into
    # runtime/rollout/sampler_output_contract.py.
    # Planned move as one group:
    # - _build_prompt_embeddings_from_payload
    # - _attach_missing_embeddings_from_payload
    # - _sampler_outputs_need_prompt_encode_fallback
    # - _sampler_outputs_batch_size
    # - _resolve_prompt_encode_kwargs
    # - _attach_missing_embeddings_from_rollout_encoder
    # RolloutManager should keep only the panel-level wiring and call a thin
    # postprocess/attach entrypoint after _sample().
    @staticmethod
    def _build_prompt_embeddings_from_payload(payload: Dict[str, Any]) -> Optional[PromptEmbeddings]:
        """Build PromptEmbeddings from an internal runtime payload.

        This helper is intentionally limited to internal payloads such as
        ``encode_prompt()`` results. User-facing data batches are prompt-only
        and should never be treated as an embedding fallback source.
        """
        def _optional_tensor(name: str) -> Optional[torch.Tensor]:
            value = payload.get(name)
            return value if torch.is_tensor(value) else None

        prompt_embeds = payload.get("prompt_embeds")
        if not torch.is_tensor(prompt_embeds):
            return None
        return PromptEmbeddings(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=_optional_tensor("pooled_prompt_embeds"),
            encoder_attention_mask=_optional_tensor("encoder_attention_mask"),
            negative_prompt_embeds=_optional_tensor("negative_prompt_embeds"),
            negative_pooled_prompt_embeds=_optional_tensor("negative_pooled_prompt_embeds"),
            text_ids=_optional_tensor("text_ids"),
            image_ids=_optional_tensor("image_ids"),
        )

    def _attach_missing_embeddings_from_payload(
        self,
        *,
        sampler_outputs: List[RolloutOutput],
        payload: Dict[str, Any],
        source_label: str,
    ) -> None:
        fallback_embeddings = self._build_prompt_embeddings_from_payload(payload)
        if fallback_embeddings is None:
            return

        offset = 0
        for idx, output in enumerate(sampler_outputs):
            if not isinstance(output, RolloutOutput):
                continue
            sample_count = int(output.batch_size)
            end = offset + sample_count

            needs_fallback_slice = output.embeddings is None
            if not needs_fallback_slice and output.embeddings is not None:
                emb = output.embeddings
                needs_fallback_slice = any(
                    (
                        emb.pooled_prompt_embeds is None and fallback_embeddings.pooled_prompt_embeds is not None,
                        emb.encoder_attention_mask is None
                        and fallback_embeddings.encoder_attention_mask is not None,
                        emb.negative_prompt_embeds is None
                        and fallback_embeddings.negative_prompt_embeds is not None,
                        emb.negative_pooled_prompt_embeds is None
                        and fallback_embeddings.negative_pooled_prompt_embeds is not None,
                        emb.text_ids is None and fallback_embeddings.text_ids is not None,
                        emb.image_ids is None and fallback_embeddings.image_ids is not None,
                    )
                )

            if needs_fallback_slice:
                if end > fallback_embeddings.prompt_embeds.shape[0]:
                    raise ValueError(
                        f"Cannot attach fallback embeddings from {source_label}: insufficient embeddings "
                        f"(need end={end}, have={fallback_embeddings.prompt_embeds.shape[0]})."
                    )
                sliced = fallback_embeddings.slice(offset, end)
                if output.embeddings is None:
                    output.embeddings = sliced
                    logger.debug(
                        "Attached fallback prompt embeddings from %s to sampler output idx=%s (range=%s:%s)",
                        source_label,
                        idx,
                        offset,
                        end,
                    )
                else:
                    emb = output.embeddings
                    patched_fields: List[str] = []
                    if emb.pooled_prompt_embeds is None and sliced.pooled_prompt_embeds is not None:
                        emb.pooled_prompt_embeds = sliced.pooled_prompt_embeds
                        patched_fields.append("pooled_prompt_embeds")
                    if emb.encoder_attention_mask is None and sliced.encoder_attention_mask is not None:
                        emb.encoder_attention_mask = sliced.encoder_attention_mask
                        patched_fields.append("encoder_attention_mask")
                    if emb.negative_prompt_embeds is None and sliced.negative_prompt_embeds is not None:
                        emb.negative_prompt_embeds = sliced.negative_prompt_embeds
                        patched_fields.append("negative_prompt_embeds")
                    if (
                        emb.negative_pooled_prompt_embeds is None
                        and sliced.negative_pooled_prompt_embeds is not None
                    ):
                        emb.negative_pooled_prompt_embeds = sliced.negative_pooled_prompt_embeds
                        patched_fields.append("negative_pooled_prompt_embeds")
                    if emb.text_ids is None and sliced.text_ids is not None:
                        emb.text_ids = sliced.text_ids
                        patched_fields.append("text_ids")
                    if emb.image_ids is None and sliced.image_ids is not None:
                        emb.image_ids = sliced.image_ids
                        patched_fields.append("image_ids")
                    if patched_fields:
                        logger.debug(
                            "Patched missing embedding fields from %s at idx=%s: %s",
                            source_label,
                            idx,
                            patched_fields,
                        )

            offset = end

    @staticmethod
    def _sampler_outputs_need_prompt_encode_fallback(
        sampler_outputs: List[Any],
    ) -> bool:
        for output in sampler_outputs:
            if not isinstance(output, RolloutOutput):
                continue
            emb = output.embeddings
            if emb is None or emb.prompt_embeds is None:
                return True
            if any(
                value is None
                for value in (
                    emb.pooled_prompt_embeds,
                    emb.encoder_attention_mask,
                    emb.negative_prompt_embeds,
                    emb.negative_pooled_prompt_embeds,
                    emb.text_ids,
                    emb.image_ids,
                )
            ):
                return True
        return False

    @staticmethod
    def _sampler_outputs_batch_size(
        sampler_outputs: List[Any],
    ) -> int:
        total = 0
        for output in sampler_outputs:
            if isinstance(output, RolloutOutput):
                total += int(output.batch_size)
        return total

    def _resolve_prompt_encode_kwargs(
        self,
        *,
        sampling_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        overrides = dict(sampling_overrides or {})
        resolved: Dict[str, Any] = {}
        defaults = {
            "height": getattr(self.args, "height", None),
            "width": getattr(self.args, "width", None),
            "num_frames": getattr(self.args, "num_frames", None),
        }
        for key, default in defaults.items():
            raw_value = overrides.get(key, default)
            if raw_value is None:
                continue
            try:
                resolved[key] = int(raw_value)
            except (TypeError, ValueError):
                logger.warning(
                    "Ignoring invalid %s=%r while building prompt-encode kwargs fallback.",
                    key,
                    raw_value,
                )
        return resolved

    def _attach_missing_embeddings_from_rollout_encoder(
        self,
        *,
        actor_group: Any,
        sampler_outputs: List[Any],
        prompts: List[str],
        sampling_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._sampler_outputs_need_prompt_encode_fallback(sampler_outputs):
            return
        if not isinstance(prompts, list) or len(prompts) == 0:
            if not self._warned_prompt_encode_fallback_failed:
                logger.warning(
                    "Prompt-encoding fallback skipped because prompts are missing."
                )
                self._warned_prompt_encode_fallback_failed = True
            return
        if actor_group is None:
            if not self._warned_prompt_encode_fallback_failed:
                logger.warning(
                    "Prompt-encoding fallback skipped because actor_group is missing."
                )
                self._warned_prompt_encode_fallback_failed = True
            return

        encode_fn = getattr(actor_group, "encode_prompt", None)
        if not callable(encode_fn):
            if not self._warned_missing_prompt_encoder_rpc:
                logger.warning(
                    "Rollout actor group does not expose encode_prompt(); "
                    "cannot attach fallback embeddings when sampler outputs omit them."
                )
                self._warned_missing_prompt_encoder_rpc = True
            return

        total_samples = self._sampler_outputs_batch_size(sampler_outputs)
        if total_samples > 0 and len(prompts) != total_samples:
            logger.warning(
                "Prompt-encoding fallback prompt count mismatch (prompts=%s, sampled=%s).",
                len(prompts),
                total_samples,
            )

        encode_kwargs = self._resolve_prompt_encode_kwargs(
            sampling_overrides=sampling_overrides
        )
        try:
            encoded_payload = encode_fn(prompts=list(prompts), **encode_kwargs)
            if not isinstance(encoded_payload, dict):
                raise TypeError(
                    "encode_prompt fallback payload must be dict, "
                    f"got {type(encoded_payload).__name__}"
                )
            self._attach_missing_embeddings_from_payload(
                sampler_outputs=sampler_outputs,
                payload=encoded_payload,
                source_label="rollout encode_prompt fallback",
            )
        except Exception as exc:
            if not self._warned_prompt_encode_fallback_failed:
                logger.warning(
                    "Prompt-encoding fallback via rollout actor failed: %s",
                    exc,
                )
                self._warned_prompt_encode_fallback_failed = True

    # --- Pipeline executor methods ---

    def _prepare_batch(self, *, data_source: Any) -> Dict[str, Any]:
        """Fetch one prompt batch from data source."""
        if data_source is not None:
            batch_size = self._prompts_per_batch()
            samples = data_source.get_samples(batch_size)
            if isinstance(samples, dict):
                return samples
            raise TypeError(
                "DataSource.get_samples() must return Dict[str, Any] with at least 'prompts'. "
                f"Got {type(samples).__name__}."
            )

        default_prompts = [
            "A beautiful sunset over the ocean",
            "A cat playing with a ball of yarn",
            "A mountain landscape with snow",
            "A futuristic city at night",
        ]
        batch_size = self._prompts_per_batch()
        return {"prompts": default_prompts[:batch_size]}

    def _sample(
        self,
        *,
        actor_group: Any,
        batch: Dict[str, Any],
        sde_indices: Optional[Set[int]],
        requirements: Optional[Any] = None,
        sampling_overrides: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Any], List[str], List[str]]:
        """Run distributed sampling with prompt-major K expansion."""
        prompts = batch.get("prompts", []) or []
        if not isinstance(prompts, list) or len(prompts) == 0:
            raise ValueError(
                "Rollout sampling requires non-empty text prompts in batch['prompts']. "
                "Prompt-embedding-only batches are no longer supported."
            )

        embedding_keys = (
            "prompt_embeds",
            "pooled_prompt_embeds",
            "negative_prompt_embeds",
            "negative_pooled_prompt_embeds",
            "text_ids",
            "image_ids",
            "encoder_attention_mask",
        )
        if (
            not self._warned_ignored_prompt_embeddings
            and any(key in batch for key in embedding_keys)
        ):
            logger.warning(
                "Rollout sampling now uses prompt-only input; batch embedding fields are ignored "
                "and are no longer used as fallback training embeddings."
            )
            self._warned_ignored_prompt_embeddings = True

        overrides = dict(sampling_overrides or {})
        overrides.setdefault("_base_prompts", list(prompts))
        batch_metadata = batch.get("metadata")
        if isinstance(batch_metadata, list):
            overrides.setdefault("_prompt_metadata", list(batch_metadata))
        num_samples_per_prompt = int(
            overrides.pop(
                "num_samples_per_prompt",
                getattr(self.args.algorithm, "num_samples_per_prompt", 1),
            )
        )
        init_same_noise = bool(
            overrides.pop(
                "init_same_noise",
                getattr(self.args.sampling, "init_same_noise", False),
            )
        )
        num_inference_steps = int(
            overrides.pop(
                "num_inference_steps",
                getattr(self.args.sampling, "num_inference_steps", 50),
            )
        )
        guidance_scale = float(
            overrides.pop(
                "guidance_scale",
                getattr(self.args.sampling, "guidance_scale", 7.5),
            )
        )
        height = int(overrides.pop("height", getattr(self.args, "height", 256)))
        width = int(overrides.pop("width", getattr(self.args, "width", 256)))
        num_frames = int(
            overrides.pop(
                "num_frames",
                getattr(self.args, "num_frames", 16),
            )
        )
        requires_trajectory = True
        requires_log_prob = True
        if requirements is not None:
            requires_trajectory = bool(getattr(requirements, "requires_trajectory", True))
            requires_log_prob = bool(getattr(requirements, "requires_log_prob", True))

        sampling_batch, train_prompts = expand_batch_for_sampling(
            {"prompts": prompts, "metadata": batch.get("metadata"), "latents": batch.get("latents")},
            num_samples_per_prompt=num_samples_per_prompt,
        )
        sampler_outputs = distributed_sample(
            actor_group=actor_group,
            batch=sampling_batch,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_frames=num_frames,
            init_same_noise=init_same_noise,
            num_samples_per_prompt=num_samples_per_prompt,
            sde_indices=sde_indices,
            return_trajectories=requires_trajectory,
            return_log_probs=requires_log_prob,
            extra_generate_kwargs=overrides,
        )
        return sampler_outputs, (train_prompts if train_prompts is not None else prompts), prompts

    def _sample_rollout_batches(
        self,
        *,
        actor_group: Any,
        batch: Dict[str, Any],
        sde_indices: Optional[Set[int]],
        requirements: Optional[Any] = None,
        sampling_overrides: Optional[Dict[str, Any]] = None,
        rollout_id: Optional[int] = None,
    ) -> Tuple[
        List[Any],
        List[str],
        List[str],
        Optional[List[Optional[Dict[str, Any]]]],
    ]:
        """Sample one rollout batch through the configured request policy."""
        validation_config = self._get_sampler_validation_config()

        def sample_request(current_batch: Dict[str, Any]) -> Tuple[List[Any], List[str], List[str]]:
            return self._sample(
                actor_group=actor_group,
                batch=current_batch,
                sde_indices=sde_indices,
                requirements=requirements,
                sampling_overrides=sampling_overrides,
            )

        def attach_embeddings(sampler_outputs: List[Any], prompts: List[str]) -> None:
            self._attach_missing_embeddings_from_rollout_encoder(
                actor_group=actor_group,
                sampler_outputs=sampler_outputs,
                prompts=prompts,
                sampling_overrides=sampling_overrides,
            )

        def validate_sampler_outputs(sampler_outputs: List[Any]) -> None:
            self._validate_sampler_outputs(
                sampler_outputs=sampler_outputs,
                requirements=requirements,
                allow_replay=validation_config["allow_replay"],
                assert_step_alignment=validation_config["assert_step_alignment"],
                mode_label=validation_config["mode_label"],
            )

        sampled_rollout = self._direct_sampling_request_planner.sample_rollout(
            batch=batch,
            rollout_id=rollout_id,
            sample_request=sample_request,
            attach_embeddings=attach_embeddings if bool(getattr(requirements, "requires_embeddings", True)) else None,
            validate_sampler_outputs=validate_sampler_outputs,
        )
        return (
            sampled_rollout.sampler_outputs,
            sampled_rollout.train_prompts,
            sampled_rollout.base_prompts,
            sampled_rollout.prompt_metadata,
        )

    # TODO(refactor): Move sampler-output contract validation into
    # runtime/rollout/sampler_output_contract.py next to the fallback helpers
    # above. This is post-sampling contract handling, not panel orchestration.
    # RolloutManager should pass requirements/validation config in and let the
    # runtime helper raise contract errors or return validated outputs.
    def _validate_sampler_outputs(
        self,
        *,
        sampler_outputs: List[Any],
        requirements: Any,
        allow_replay: bool,
        assert_step_alignment: bool,
        mode_label: str,
    ) -> None:
        """Validate sampler outputs against algorithm requirements."""
        replay_notice_emitted = False
        for idx, out in enumerate(sampler_outputs):
            try:
                meta = getattr(out, "metadata", {}) or {}
                generator_type = meta.get("generator_type") if isinstance(meta, dict) else None
                allow_missing_log_probs = bool(allow_replay)
                if allow_missing_log_probs and not replay_notice_emitted:
                    logger.warning(
                        "Replay path enabled: allowing missing rollout log_probs; "
                        "training actors will replay old log_probs before backward."
                    )
                    replay_notice_emitted = True

                out.validate_contract(
                    requires_log_probs=bool(requirements.requires_log_prob) and not allow_missing_log_probs,
                    requires_trajectory=bool(requirements.requires_trajectory),
                    requires_embeddings=bool(getattr(requirements, "requires_embeddings", True)),
                )

                if assert_step_alignment:
                    resolved_steps = out.resolved_step_indices
                    if int(resolved_steps.shape[0]) != int(out.timesteps.shape[0]):
                        raise ValueError(
                            f"step/timestep length mismatch: step_indices={resolved_steps.shape[0]}, "
                            f"timesteps={out.timesteps.shape[0]}"
                        )
            except Exception as e:
                meta = getattr(out, "metadata", {}) or {}
                generator_type = meta.get("generator_type") if isinstance(meta, dict) else None
                capabilities = meta.get("engine_capabilities") if isinstance(meta, dict) else None
                traj_shape = tuple(out.trajectories.shape) if getattr(out, "trajectories", None) is not None else None
                latents_shape = tuple(out.latents.shape) if getattr(out, "latents", None) is not None else None
                steps_shape = tuple(out.resolved_step_indices.shape) if hasattr(out, "resolved_step_indices") else None
                hint = ""
                if generator_type in {"fastvideo", "sglang"}:
                    hint = (
                        f" {generator_type} currently may omit rollout log_probs; "
                        "enable replay_log_probs and ensure prompt text inputs are present."
                    )
                raise RuntimeError(
                    f"Sampler output contract validation failed in {mode_label} path at index={idx}: {e}.{hint} "
                    f"capabilities={capabilities}, latents_shape={latents_shape}, "
                    f"trajectories_shape={traj_shape}, step_indices_shape={steps_shape}"
                ) from e

    def _compute_reward_and_advantage(
        self,
        *,
        algorithm: Any,
        reward_service: Any,
        sampler_outputs: List[Any],
        prompts: List[str],
        prompt_metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, List[float]]]:
        """Compute rewards and advantages from sampler outputs."""
        rewards, reward_components = self._compute_rewards_only(
            reward_service=reward_service,
            sampler_outputs=sampler_outputs,
            prompts=prompts,
            prompt_metadata=prompt_metadata,
        )

        advantages = _compute_advantages_stage(
            algorithm=algorithm,
            num_samples_per_prompt=int(getattr(self.args.algorithm, "num_samples_per_prompt", 1)),
            reward_mix_mode=str(getattr(self.args.reward, "reward_mix_mode", "reward_aggr")),
            rewards=rewards,
            prompts=prompts,
            reward_components=reward_components,
            reward_workers=getattr(reward_service, "workers", None) if reward_service is not None else None,
            reward_component_weights=self._reward_schema.component_weights(),
        )
        return rewards, advantages, reward_components

    def _compute_rewards_only(
        self,
        *,
        reward_service: Any,
        sampler_outputs: List[Any],
        prompts: List[str],
        prompt_metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
        reward_path_override: Optional[str] = None,
        num_samples_per_prompt_override: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """Compute rewards only (shared by default path and custom pipelines)."""
        num_samples_per_prompt = int(
            num_samples_per_prompt_override
            if num_samples_per_prompt_override is not None
            else getattr(self.args.algorithm, "num_samples_per_prompt", 1)
        )
        reward_path = str(
            reward_path_override
            if reward_path_override is not None
            else getattr(self.args.reward, "reward_path", "")
        )

        return _compute_rewards_stage(
            reward_service=reward_service,
            reward_path=reward_path,
            num_samples_per_prompt=num_samples_per_prompt,
            sampler_outputs=sampler_outputs,
            base_prompts=prompts,
            prompt_metadata=prompt_metadata,
        )

    def _assemble_training_batch(
        self,
        *,
        algorithm: Any,
        sampler_outputs: List[Any],
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        prompts: List[str],
        sde_indices: Optional[Set[int]],
    ) -> TrainingBatch:
        """Delegate training-batch assembly to algorithm strategy."""
        return algorithm.assemble_training_batch(
            num_inference_steps=int(self.args.sampling.num_inference_steps),
            sampler_outputs=sampler_outputs,
            rewards=rewards,
            advantages=advantages,
            prompts=prompts,
            sde_indices=sde_indices,
        )

    def _eval_batch(
        self,
        *,
        rollout_id: int,
        data_source: Any,
        actor_group: Any,
        reward_service: Any,
    ) -> Dict[str, Any]:
        """Run evaluation sampling and reward aggregation."""
        eval_batch: Dict[str, Any]
        if data_source is not None and hasattr(data_source, "get_eval_samples"):
            eval_samples = data_source.get_eval_samples(self.args.rollout.eval_batch_size)
            if isinstance(eval_samples, dict):
                eval_batch = dict(eval_samples)
            elif isinstance(eval_samples, list):
                eval_batch = {"prompts": list(eval_samples)}
            else:
                raise TypeError(
                    "DataSource.get_eval_samples() must return List[str] or Dict[str, Any] "
                    f"with at least 'prompts'. Got {type(eval_samples).__name__}."
                )
        else:
            eval_batch = self._prepare_batch(data_source=data_source)

        prompts = list(eval_batch.get("prompts", [])[: self.args.rollout.eval_batch_size])
        prompt_metadata = eval_batch.get("metadata")
        if isinstance(prompt_metadata, list):
            prompt_metadata = prompt_metadata[: len(prompts)]
        else:
            prompt_metadata = None

        outputs = distributed_sample(
            actor_group=actor_group,
            batch={"prompts": prompts},
            num_inference_steps=int(self.args.sampling.num_inference_steps),
            guidance_scale=float(self.args.sampling.guidance_scale),
            height=int(self.args.height),
            width=int(self.args.width),
            num_frames=int(self.args.num_frames),
            init_same_noise=bool(getattr(self.args.sampling, "init_same_noise", False)),
            num_samples_per_prompt=int(getattr(self.args.algorithm, "num_samples_per_prompt", 1)),
            sde_indices=None,
        )
        rewards, _ = self._compute_rewards_only(
            reward_service=reward_service,
            sampler_outputs=outputs,
            prompts=prompts,
            prompt_metadata=prompt_metadata,
        )

        return {
            "rollout_id": rollout_id,
            "num_samples": len(prompts),
            "mean_reward": rewards.mean().item(),
            "std_reward": rewards.std().item(),
            "prompts": prompts,
        }

    def generate_and_push(
        self,
        rollout_id: int,
        buffer: Any,
        *,
        collect_media_preview: bool = False,
        media_max_items: int = 8,
    ) -> Dict[str, Any]:
        """Generate training data and push directly to buffer.

        This is the preferred flow: Manager generates, then pushes result.
        Avoids the old pattern where Buffer pulls from Manager (which blocked
        the Buffer actor for the entire generation cycle).

        Args:
            rollout_id: Current rollout iteration number
            buffer: RolloutBufferActor handle (Ray actor)

        Returns:
            Push result dict from buffer.push()
        """
        train_data = self._build_training_batch(
            rollout_id,
            collect_media_preview=collect_media_preview,
            media_max_items=media_max_items,
        )
        metadata = dict(self._last_rollout_metadata) if self._last_rollout_metadata else None
        push_result = ray.get(
            buffer.push.remote(
                rollout_id=rollout_id,
                train_data=train_data,
                metadata=metadata,
            )
        )
        if not push_result.get("accepted", False):
            raise RuntimeError(
                f"Rollout buffer rejected rollout_id={rollout_id}: {push_result.get('error')}"
            )
        return push_result

    def eval(self, rollout_id: int) -> Dict[str, Any]:
        """
        Run evaluation generation.

        Args:
            rollout_id: Current rollout iteration

        Returns:
            Dictionary of evaluation metrics and samples
        """
        logger.info(f"Running evaluation for rollout {rollout_id}")

        actor_group = self.external_sampling_actors or self.rollout_actors
        return self._eval_batch(
            rollout_id=rollout_id,
            data_source=self.data_source,
            actor_group=actor_group,
            reward_service=self.reward_service,
        )

    def update_weights(
        self,
        state_dict_ref: ray.ObjectRef,
    ) -> None:
        """Update rollout actor weights from training.

        .. deprecated::
            Kept for backward compatibility. New code should use
            WeightSyncProtocol which delegates to rollout_actor_group directly.

        Args:
            state_dict_ref: ObjectRef containing new state dict
        """
        if self.rollout_actors is not None:
            self.rollout_actors.update_weights(state_dict_ref)

    def update_weights_from_path(
        self,
        checkpoint_path: str,
    ) -> None:
        """Update rollout actor weights from a shared checkpoint path.

        .. deprecated:: Proxy retained for TrainingActor remote calls.
        """
        if self.rollout_actors is not None and hasattr(self.rollout_actors, "update_weights_from_path"):
            self.rollout_actors.update_weights_from_path(checkpoint_path)

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        target_modules: Optional[List[str]] = None,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ) -> None:
        """Update rollout actor weights from serialized tensor payload.

        .. deprecated:: Proxy retained for TrainingActor remote calls.
        """
        if self.rollout_actors is not None and hasattr(self.rollout_actors, "update_weights_from_tensor"):
            self.rollout_actors.update_weights_from_tensor(
                serialized_named_tensors=serialized_named_tensors,
                target_modules=target_modules,
                load_format=load_format,
                flush_cache=flush_cache,
            )

    def init_weights_update_group(
        self,
        *,
        master_address: str,
        master_port: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
    ) -> None:
        """Initialize rollout-side distributed group for weight updates.

        .. deprecated:: Proxy retained for WeightSyncProtocol remote calls.
        """
        if self.rollout_actors is not None and hasattr(self.rollout_actors, "init_weights_update_group"):
            self.rollout_actors.init_weights_update_group(
                master_address=master_address,
                master_port=master_port,
                world_size=world_size,
                group_name=group_name,
                backend=backend,
            )

    def destroy_weights_update_group(
        self,
        *,
        group_name: str,
    ) -> None:
        """Destroy rollout-side distributed group for weight updates.

        .. deprecated:: Proxy retained for WeightSyncProtocol remote calls.
        """
        if self.rollout_actors is not None and hasattr(self.rollout_actors, "destroy_weights_update_group"):
            self.rollout_actors.destroy_weights_update_group(group_name=group_name)

    def update_weights_from_distributed(
        self,
        *,
        names: List[str],
        dtypes: List[str],
        shapes: List[List[int]],
        group_name: str,
        target_modules: Optional[List[str]] = None,
        flush_cache: bool = True,
    ) -> None:
        """Update rollout actor weights from distributed broadcast metadata.

        .. deprecated:: Proxy retained for TrainingActor remote calls.
        """
        if self.rollout_actors is not None and hasattr(self.rollout_actors, "update_weights_from_distributed"):
            self.rollout_actors.update_weights_from_distributed(
                names=names,
                dtypes=dtypes,
                shapes=shapes,
                group_name=group_name,
                target_modules=target_modules,
                flush_cache=flush_cache,
            )

    def get_weight_sync_topology(self) -> Dict[str, int]:
        """Return rollout-side topology for weight-sync strategy selection."""
        if self.rollout_actors is None or not hasattr(self.rollout_actors, "get_weight_sync_topology"):
            return {"num_actors": 0, "num_gpus_per_actor": 0, "total_gpus": 0}
        payload = self.rollout_actors.get_weight_sync_topology()
        if not isinstance(payload, dict):
            return {"num_actors": 0, "num_gpus_per_actor": 0, "total_gpus": 0}
        return {
            "num_actors": int(payload.get("num_actors", 0)),
            "num_gpus_per_actor": int(payload.get("num_gpus_per_actor", 0)),
            "total_gpus": int(payload.get("total_gpus", 0)),
        }

    def get_dataset_step_info(self) -> Dict[str, Any]:
        """Compute rollout-step progress information for the current dataset."""
        prompts_per_batch = self._prompts_per_batch()
        drop_last = bool(getattr(self.data_source, "drop_last", False))
        info: Dict[str, Any] = {
            "num_samples": 0,
            "prompts_per_batch": prompts_per_batch,
            "estimated_steps_per_dataset_pass": 0,
            "steps_before_reset": 0,
            "remainder_samples": 0,
            "drop_last": drop_last,
            "exact_dataset_pass_per_cycle": False,
        }

        if self.data_source is None or not hasattr(self.data_source, "num_samples"):
            return info

        num_samples = int(self.data_source.num_samples)
        info["num_samples"] = num_samples
        if num_samples <= 0:
            return info

        estimated_steps = (num_samples + prompts_per_batch - 1) // prompts_per_batch
        remainder = num_samples % prompts_per_batch
        if drop_last:
            steps_before_reset = num_samples // prompts_per_batch
        else:
            steps_before_reset = estimated_steps

        info.update(
            {
                "estimated_steps_per_dataset_pass": int(estimated_steps),
                "steps_before_reset": int(steps_before_reset),
                "remainder_samples": int(remainder),
                "exact_dataset_pass_per_cycle": bool(
                    remainder == 0 and steps_before_reset == estimated_steps
                ),
            }
        )
        return info

    def get_stats(self) -> Dict[str, Any]:
        """Get rollout statistics."""
        return {
            "total_samples_generated": self._total_samples_generated,
        }

    def sleep(self) -> None:
        """Put rollout actors into sleep mode."""
        if self.rollout_actors is not None:
            self.rollout_actors.sleep()

    def wake_up(self) -> None:
        """Wake rollout actors up for generation/weight update."""
        if self.rollout_actors is not None:
            self.rollout_actors.wake_up()

    def dispose(self) -> None:
        """Clean up resources."""
        if self._owns_rollout_actors and self.rollout_actors is not None:
            self.rollout_actors.dispose()
        logger.info("RolloutManager disposed")


def create_rollout_manager(
    args,
    pg_result: Optional[Tuple] = None,
    reward_pg_result: Optional[Tuple] = None,
) -> Tuple[ray.ObjectRef, Dict[str, Any]]:
    """
    Factory function to create RolloutManager.

    Args:
        args: TrainingArguments instance
        pg_result: Placement group result for rollout actors
        reward_pg_result: Placement group result for reward actors

    Returns:
        Tuple of (RolloutManager actor handle, dataset step info)
    """
    rollout_manager = RolloutManager.options(
        num_cpus=1,
        num_gpus=0,
    ).remote(args, pg_result, reward_pg_result)

    # Initialize
    ray.get(rollout_manager.init.remote())

    dataset_step_info = ray.get(rollout_manager.get_dataset_step_info.remote())

    return rollout_manager, dataset_step_info
