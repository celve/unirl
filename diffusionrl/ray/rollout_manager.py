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

import ray
import torch

from diffusionrl.config.arguments import is_training_actor_direct_sampling_mode
from diffusionrl.runtime.pipeline.advantage_stage import compute_advantages as _compute_advantages_stage
from diffusionrl.runtime.pipeline.reward_stage import compute_rewards as _compute_rewards_stage
from diffusionrl.runtime.pipeline.sampling_stage import (
    distributed_sample,
    expand_batch_for_sampling,
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
            sampling_overrides=sampling_overrides,
        )
        self._manager._attach_missing_embeddings_from_batch(
            sampler_outputs=outputs,
            batch=working_batch,
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
                else getattr(manager.args, "num_samples_per_prompt", 1)
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

        # Rollout actor group
        self.rollout_actors = None
        self.external_sampling_actors = None
        self._owns_rollout_actors = False

        # Stats
        self._total_samples_generated = 0
        self._current_step = 0

    def init(self) -> None:
        """
        Initialize all components via dynamic loading.

        Components are loaded using load_function with paths from args.
        """
        logger.info("Initializing RolloutManager...")
        self._validate_batch_shape()

        # 1. Load algorithm
        algorithm_cls = load_function(self.args.algorithm_path)
        if not hasattr(algorithm_cls, "from_args"):
            raise TypeError(
                f"Algorithm {self.args.algorithm_path} must implement classmethod from_args(args)."
            )
        self.algorithm = algorithm_cls.from_args(self.args)
        logger.info(f"Algorithm loaded: {self.args.algorithm_path} (clip_max={self.args.advantage_clip_max}, sde_ratio={getattr(self.args, 'sde_ratio', 'N/A')})")
        use_per_prompt_tracker = getattr(self.args, 'use_per_prompt_stat_tracker', False)
        if getattr(self.args, "advantage_type", "group") == "per_prompt" and not use_per_prompt_tracker:
            logger.warning("advantage_type=per_prompt but use_per_prompt_stat_tracker=False; advantages will fall back to group normalization.")

        # 2. Load timestep scheduler based on algorithm requirements
        self._init_timestep_scheduler()

        # Optional custom rollout pipeline (slime-style pluggable generate function)
        rollout_pipeline_path = getattr(self.args, "rollout_pipeline_path", None)
        if rollout_pipeline_path:
            self.rollout_pipeline_fn = load_function(rollout_pipeline_path)
            logger.info("Custom rollout pipeline loaded: %s", rollout_pipeline_path)

        # 3. Initialize reward service
        self._init_reward_service()
        if self.reward_service is None:
            raise RuntimeError("RewardService initialization failed.")
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
        from diffusionrl.reward.service import RewardService
        self.reward_service = RewardService(
            args=self.args,
            reward_pg_result=self.reward_pg_result,
        )

    def _init_timestep_scheduler(self) -> None:
        """Initialize timestep scheduler based on algorithm requirements and config."""
        from diffusionrl.samplers.schedulers import get_scheduler

        # Get algorithm's sampling requirements
        requirements = self.algorithm.get_sampling_requirements()

        # Get scheduler config from args
        scheduler_type = getattr(self.args, 'timestep_strategy', 'all')
        num_timesteps = self.args.num_inference_steps
        timestep_fraction = getattr(self.args, 'timestep_fraction', 1.0)

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
            explicit_group_size = getattr(self.args, 'window_group_size', None)
            if explicit_group_size is None and requirements.sde_ratio < 1.0:
                group_size = max(1, int(num_timesteps * requirements.sde_ratio))
            else:
                group_size = explicit_group_size or 4

            self.timestep_scheduler = get_scheduler(
                scheduler_type='window',
                num_timesteps=num_timesteps,
                timestep_fraction=timestep_fraction,
                strategy=getattr(self.args, 'window_strategy', 'progressive'),
                group_size=group_size,
                iters_per_group=getattr(self.args, 'window_iters_per_group', 25),
                max_iters_per_group=getattr(self.args, 'window_max_iters_per_group', None),
                min_iters_per_group=getattr(self.args, 'window_min_iters_per_group', None),
                overlap=getattr(self.args, 'window_overlap', False),
                overlap_step=getattr(self.args, 'window_overlap_step', 1),
                roll_back=getattr(self.args, 'window_roll_back', False),
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
            if timestep_fraction < 1.0:
                effective_steps = int(num_timesteps * timestep_fraction)
                logger.info(
                    "All SDE scheduler initialized; "
                    f"timestep_fraction={timestep_fraction} (SDE on first {effective_steps}/{num_timesteps} timesteps)"
                )
            else:
                logger.info("All SDE scheduler initialized (standard GRPO)")

    def _create_rollout_actors(self) -> None:
        """Create rollout actor group from placement group."""
        from .groups.factory import create_rollout_actor_group

        self.rollout_actors = create_rollout_actor_group(self.args, self.pg_result)
        self._owns_rollout_actors = True
        logger.info("Rollout actors created via create_rollout_actor_group")

    def attach_sampling_actors(self, actor_group) -> None:
        """Attach external sampling actors (e.g., TrainingActorGroup)."""
        self.external_sampling_actors = actor_group
        logger.info("External sampling actors attached")

    def _build_training_batch(self, rollout_id: int) -> TrainingBatch:
        """Run rollout pipeline and return one typed TrainingBatch."""
        if not self._is_initialized:
            raise RuntimeError("RolloutManager not initialized. Call init() first.")

        logger.info(f"Starting generation for rollout {rollout_id}")

        # 1. Get batch from data source (supports both prompts and embeddings)
        batch = self._prepare_batch(data_source=self.data_source)
        prompts = batch.get("prompts", [])

        # 2. Get algorithm requirements to determine pipeline
        requirements = self.algorithm.get_sampling_requirements()
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
            self._total_samples_generated += sample_count
        except Exception as e:
            logger.warning(f"Failed to compute sample count ({e}), falling back to prompt count")
            self._total_samples_generated += len(prompts)

        self._current_step += 1
        if self.timestep_scheduler is not None:
            self.timestep_scheduler.update(self._current_step)

    def _debug_log_tensor_stats(self, label: str, value: Optional[torch.Tensor]) -> None:
        if not bool(getattr(self.args, "debug_print_tensor_stats", True)):
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
            int(getattr(self.args, "num_samples_per_prompt", 1)),
        )
        sampler_outputs, train_prompts, base_prompts = self._sample(
            actor_group=actor_group,
            batch=batch,
            sde_indices=sde_indices,
        )
        self._attach_missing_embeddings_from_batch(
            sampler_outputs=sampler_outputs,
            batch=batch,
        )
        if bool(getattr(requirements, "requires_embeddings", True)):
            self._attach_missing_embeddings_from_rollout_encoder(
                actor_group=actor_group,
                sampler_outputs=sampler_outputs,
                prompts=(train_prompts if train_prompts else prompts),
            )
        logger.info(
            "[debug] rollout=%s stage=sampling done outputs=%s",
            rollout_id,
            len(sampler_outputs),
        )

        validation_config = self._get_sampler_validation_config()
        self._validate_sampler_outputs(
            sampler_outputs=sampler_outputs,
            requirements=requirements,
            allow_replay=validation_config["allow_replay"],
            assert_step_alignment=validation_config["assert_step_alignment"],
            mode_label=validation_config["mode_label"],
        )

        logger.info("[debug] rollout=%s stage=reward_advantage", rollout_id)
        rewards, advantages, reward_components = self._compute_reward_and_advantage(
            algorithm=self.algorithm,
            reward_service=self.reward_service,
            sampler_outputs=sampler_outputs,
            prompts=base_prompts if base_prompts else prompts,
            prompt_metadata=batch.get("metadata"),
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
            "debug_mode": str(getattr(self.args, "debug_mode", "none")),
            "prompts": list(prompts),
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
        requirements = self.algorithm.get_sampling_requirements()
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
                "debug_mode": str(getattr(self.args, "debug_mode", "none")),
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
                "custom_rollout_pipeline": str(getattr(self.args, "rollout_pipeline_path", "")),
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
        requirements = self.algorithm.get_sampling_requirements()
        actor_group = self.external_sampling_actors or self.rollout_actors

        sde_indices = self._resolve_rollout_sde_indices()

        sampler_outputs, train_prompts, base_prompts = self._sample(
            actor_group=actor_group,
            batch=batch,
            sde_indices=sde_indices,
        )
        self._attach_missing_embeddings_from_batch(
            sampler_outputs=sampler_outputs, batch=batch,
        )
        if bool(getattr(requirements, "requires_embeddings", True)):
            self._attach_missing_embeddings_from_rollout_encoder(
                actor_group=actor_group,
                sampler_outputs=sampler_outputs,
                prompts=(train_prompts if train_prompts else prompts),
            )

        validation_config = self._get_sampler_validation_config()
        self._validate_sampler_outputs(
            sampler_outputs=sampler_outputs,
            requirements=requirements,
            allow_replay=validation_config["allow_replay"],
            assert_step_alignment=validation_config["assert_step_alignment"],
            mode_label=validation_config["mode_label"],
        )

        self._debug_cache = {
            "rollout_id": rollout_id,
            "batch": batch,
            "prompts": prompts,
            "train_prompts": train_prompts if train_prompts else prompts,
            "base_prompts": base_prompts if base_prompts else prompts,
            "sampler_outputs": sampler_outputs,
            "sde_indices": sde_indices,
            "requirements": requirements,
        }

        return {
            "rollout_id": rollout_id,
            "num_outputs": len(sampler_outputs),
            "total_samples": sum(int(getattr(o, "batch_size", 1)) for o in sampler_outputs),
            "prompts": prompts,
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
            prompt_metadata=cache["batch"].get("metadata"),
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
            num_samples_per_prompt=int(getattr(self.args, "num_samples_per_prompt", 1)),
            reward_mix_mode=str(getattr(self.args, "reward_mix_mode", "reward_aggr")),
            rewards=cache["rewards"],
            prompts=cache["base_prompts"],
            reward_components=cache.get("reward_components", {}),
            reward_workers=getattr(self.reward_service, "workers", None) if self.reward_service is not None else None,
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
        """Validate prompts_per_batch * k against training batch/world_size."""
        try:
            prompts_per_batch = getattr(self.args, "prompts_per_batch", 1)
            k = getattr(self.args, "num_samples_per_prompt", 1)
            gen_batch = prompts_per_batch * k

            train_world_size = self.args.training_num_nodes * self.args.training_num_gpus_per_node
            train_bsz = self.args.batch_size
            denom = train_world_size * train_bsz if train_world_size > 0 else train_bsz

            if denom > 0 and gen_batch % denom != 0:
                logger.warning(
                    "Generation batch (%d) is not divisible by train_world_size*batch_size (%d). "
                    "This may lead to uneven gradient accumulation. Consider adjusting prompts_per_batch or batch_size.",
                    gen_batch,
                    denom,
                )
        except Exception as e:
            logger.warning(f"Batch shape validation skipped: {e}")

    def _generate_training_data(
        self,
        batch: Dict[str, Any],
        rollout_id: int,
        requirements: Any,
        actor_group: Any,
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

        # Sample
        sampler_outputs, train_prompts, base_prompts = self._sample(
            actor_group=actor_group,
            batch=batch,
            sde_indices=sde_indices,
        )
        self._attach_missing_embeddings_from_batch(
            sampler_outputs=sampler_outputs,
            batch=batch,
        )
        if bool(getattr(requirements, "requires_embeddings", True)):
            self._attach_missing_embeddings_from_rollout_encoder(
                actor_group=actor_group,
                sampler_outputs=sampler_outputs,
                prompts=(train_prompts if train_prompts else prompts),
            )

        # Validate
        validation_config = self._get_sampler_validation_config()
        self._validate_sampler_outputs(
            sampler_outputs=sampler_outputs,
            requirements=requirements,
            allow_replay=validation_config["allow_replay"],
            assert_step_alignment=validation_config["assert_step_alignment"],
            mode_label=validation_config["mode_label"],
        )

        # Reward + advantage
        rewards, advantages, _ = self._compute_reward_and_advantage(
            algorithm=self.algorithm,
            reward_service=self.reward_service,
            sampler_outputs=sampler_outputs,
            prompts=base_prompts if base_prompts else prompts,
            prompt_metadata=batch.get("metadata"),
        )

        # Assemble
        return self._assemble_training_batch(
            algorithm=self.algorithm,
            sampler_outputs=sampler_outputs,
            rewards=rewards,
            advantages=advantages,
            prompts=train_prompts if train_prompts else prompts,
            sde_indices=sde_indices,
        )

    @staticmethod
    def _build_prompt_embeddings_from_batch(batch: Dict[str, Any]) -> Optional[PromptEmbeddings]:
        def _optional_tensor(name: str) -> Optional[torch.Tensor]:
            value = batch.get(name)
            return value if torch.is_tensor(value) else None

        prompt_embeds = batch.get("prompt_embeds")
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

    def _attach_missing_embeddings_from_batch(
        self,
        *,
        sampler_outputs: List[RolloutOutput],
        batch: Dict[str, Any],
    ) -> None:
        fallback_embeddings = self._build_prompt_embeddings_from_batch(batch)
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
                        "Cannot attach fallback embeddings: insufficient batch embeddings "
                        f"(need end={end}, have={fallback_embeddings.prompt_embeds.shape[0]})."
                    )
                sliced = fallback_embeddings.slice(offset, end)
                if output.embeddings is None:
                    output.embeddings = sliced
                    logger.debug(
                        "Attached fallback prompt embeddings to sampler output idx=%s (range=%s:%s)",
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
                            "Patched missing embedding fields from batch fallback at idx=%s: %s",
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
            self._attach_missing_embeddings_from_batch(
                sampler_outputs=sampler_outputs,
                batch=encoded_payload,
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
            batch_size = getattr(self.args, "prompts_per_batch", None) or self.args.batch_size
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
        batch_size = getattr(self.args, "prompts_per_batch", None) or self.args.batch_size
        return {"prompts": default_prompts[:batch_size]}

    def _sample(
        self,
        *,
        actor_group: Any,
        batch: Dict[str, Any],
        sde_indices: Optional[Set[int]],
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
                "Rollout sampling now uses prompt-only input; batch embedding fields are ignored."
            )
            self._warned_ignored_prompt_embeddings = True

        overrides = dict(sampling_overrides or {})
        num_samples_per_prompt = int(
            overrides.pop(
                "num_samples_per_prompt",
                getattr(self.args, "num_samples_per_prompt", 1),
            )
        )
        init_same_noise = bool(
            overrides.pop(
                "init_same_noise",
                getattr(self.args, "init_same_noise", False),
            )
        )
        num_inference_steps = int(
            overrides.pop(
                "num_inference_steps",
                getattr(self.args, "num_inference_steps", 50),
            )
        )
        guidance_scale = float(
            overrides.pop(
                "guidance_scale",
                getattr(self.args, "guidance_scale", 7.5),
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
            extra_generate_kwargs=overrides,
        )
        return sampler_outputs, (train_prompts if train_prompts is not None else prompts), prompts

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
            num_samples_per_prompt=int(getattr(self.args, "num_samples_per_prompt", 1)),
            reward_mix_mode=str(getattr(self.args, "reward_mix_mode", "reward_aggr")),
            rewards=rewards,
            prompts=prompts,
            reward_components=reward_components,
            reward_workers=getattr(reward_service, "workers", None) if reward_service is not None else None,
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
            else getattr(self.args, "num_samples_per_prompt", 1)
        )
        reward_path = str(
            reward_path_override
            if reward_path_override is not None
            else getattr(self.args, "reward_path", "")
        )

        if reward_service is None:
            raise RuntimeError("RewardService is not initialized.")
        return _compute_rewards_stage(
            reward_service=reward_service,
            reward_path=reward_path,
            num_samples_per_prompt=num_samples_per_prompt,
            sampler_outputs=sampler_outputs,
            prompts=prompts,
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
            num_inference_steps=int(self.args.num_inference_steps),
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
        if data_source is not None and hasattr(data_source, "get_eval_samples"):
            prompts = data_source.get_eval_samples(self.args.eval_batch_size)
        else:
            prompts = self._prepare_batch(data_source=data_source).get("prompts", [])[: self.args.eval_batch_size]

        outputs = distributed_sample(
            actor_group=actor_group,
            batch={"prompts": prompts},
            num_inference_steps=int(self.args.num_inference_steps),
            guidance_scale=float(self.args.guidance_scale),
            height=int(self.args.height),
            width=int(self.args.width),
            num_frames=int(self.args.num_frames),
            init_same_noise=bool(getattr(self.args, "init_same_noise", False)),
            num_samples_per_prompt=int(getattr(self.args, "num_samples_per_prompt", 1)),
            sde_indices=None,
        )
        rewards, _ = self._compute_rewards_only(
            reward_service=reward_service,
            sampler_outputs=outputs,
            prompts=prompts,
        )

        return {
            "rollout_id": rollout_id,
            "num_samples": len(prompts),
            "mean_reward": rewards.mean().item(),
            "std_reward": rewards.std().item(),
            "prompts": prompts,
        }

    def generate_and_push(self, rollout_id: int, buffer: Any) -> Dict[str, Any]:
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
        train_data = self._build_training_batch(rollout_id)
        push_result = ray.get(buffer.push.remote(rollout_id=rollout_id, train_data=train_data))
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

    def get_num_rollout_per_epoch(self) -> int:
        """Get number of rollouts per epoch."""
        if self.args.rollouts_per_epoch is not None:
            return self.args.rollouts_per_epoch

        if self.data_source is not None and hasattr(self.data_source, "num_samples"):
            num_samples = int(self.data_source.num_samples)
            if num_samples > 0:
                prompts_per_batch = int(
                    getattr(self.args, "prompts_per_batch", 0) or self.args.batch_size
                )
                prompts_per_batch = max(1, prompts_per_batch)
                return max(1, num_samples // prompts_per_batch)

        return 100

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
) -> Tuple[ray.ObjectRef, int]:
    """
    Factory function to create RolloutManager.

    Args:
        args: TrainingArguments instance
        pg_result: Placement group result for rollout actors
        reward_pg_result: Placement group result for reward actors

    Returns:
        Tuple of (RolloutManager actor handle, num_rollout_per_epoch)
    """
    rollout_manager = RolloutManager.options(
        num_cpus=1,
        num_gpus=0,
    ).remote(args, pg_result, reward_pg_result)

    # Initialize
    ray.get(rollout_manager.init.remote())

    # Get rollouts per epoch
    num_rollout_per_epoch = ray.get(rollout_manager.get_num_rollout_per_epoch.remote())

    return rollout_manager, num_rollout_per_epoch
