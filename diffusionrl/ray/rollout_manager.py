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
from diffusionrl.runtime.rollout import RolloutPipelineExecutor
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
        if self._requirements.is_forward_process:
            resolved_sde_indices = None
        elif sde_indices is not None:
            resolved_sde_indices = set(int(i) for i in sde_indices)
        elif self._default_sde_indices is not None:
            resolved_sde_indices = set(self._default_sde_indices)
        else:
            resolved_sde_indices = None

        outputs, train_prompts, base_prompts = self._manager.rollout_executor.sample(
            actor_group=self._actor_group,
            batch=working_batch,
            sde_indices=resolved_sde_indices,
            sampling_overrides=sampling_overrides,
        )
        self._manager._attach_missing_embeddings_from_batch(
            sampler_outputs=outputs,
            batch=working_batch,
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

    default_sde_indices: Optional[Set[int]] = None
    if not requirements.is_forward_process:
        default_sde_indices = set(
            manager.timestep_scheduler.get_sde_indices(manager._current_step)
        )
        if hasattr(manager.algorithm, "set_sde_indices"):
            manager.algorithm.set_sde_indices(default_sde_indices)

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
        return manager.rollout_executor.compute_rewards_only(
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
        return manager.rollout_executor.compute_reward_and_advantage(
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
        return manager.rollout_executor.assemble_training_batch(
            algorithm=manager.algorithm,
            requirements=requirements,
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
        self.rollout_executor = RolloutPipelineExecutor(args)
        self.rollout_pipeline_fn = None

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
        batch = self.rollout_executor.prepare_batch(data_source=self.data_source)
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
            if requirements.is_forward_process:
                # NFT: Forward process algorithm
                # Only need clean latents, not trajectories
                train_data = self._generate_nft_data(
                    batch=batch,
                    rollout_id=rollout_id,
                    requirements=requirements,
                    actor_group=actor_group,
                )
            else:
                # GRPO/MixGRPO: Trajectory-based algorithm
                train_data = self._generate_trajectory_data(
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

        # Update stats (count actual generated samples)
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

        # Update scheduler (for MixGRPO)
        if self.timestep_scheduler is not None:
            self.timestep_scheduler.update(self._current_step)

        return train_data

    def build_training_batch(self, rollout_id: int) -> TrainingBatch:
        """Public rollout entrypoint used by RolloutBufferActor.request_rollout()."""
        return self._build_training_batch(rollout_id)

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

    def _generate_trajectory_data(
        self,
        batch: Dict[str, Any],
        rollout_id: int,
        requirements: Any,
        actor_group: Any,
    ) -> TrainingBatch:
        """Generate training data for trajectory-based algorithms (GRPO/MixGRPO).

        Args:
            batch: Dict containing either:
                - Embedding mode: prompt_embeds, pooled_prompt_embeds, text_ids (optional), prompts
                - Text mode: prompts only
            rollout_id: Current rollout iteration number
        """
        prompts = batch.get("prompts", []) or []

        # Get SDE indices from scheduler
        sde_indices = self.timestep_scheduler.get_sde_indices(self._current_step)
        logger.debug(f"SDE indices for step {self._current_step}: {sorted(sde_indices)[:5]}...")

        # Update algorithm's knowledge of current SDE indices (for MixGRPO state tracking)
        if hasattr(self.algorithm, 'set_sde_indices'):
            self.algorithm.set_sde_indices(sde_indices)

        sampler_outputs, train_prompts, base_prompts = self.rollout_executor.sample(
            actor_group=actor_group,
            batch=batch,
            sde_indices=sde_indices,
        )
        self._attach_missing_embeddings_from_batch(
            sampler_outputs=sampler_outputs,
            batch=batch,
        )

        allow_replay = (
            bool(getattr(self.args, "replay_log_probs", False))
            and getattr(self.args, "loss_type", "grpo") == "grpo"
        )
        self.rollout_executor.validate_sampler_outputs(
            sampler_outputs=sampler_outputs,
            requirements=requirements,
            allow_replay=allow_replay,
            assert_step_alignment=True,
            mode_label="trajectory",
        )

        rewards, advantages, _ = self.rollout_executor.compute_reward_and_advantage(
            algorithm=self.algorithm,
            reward_service=self.reward_service,
            sampler_outputs=sampler_outputs,
            prompts=base_prompts if base_prompts else prompts,
            prompt_metadata=batch.get("metadata"),
        )

        return self.rollout_executor.assemble_training_batch(
            algorithm=self.algorithm,
            requirements=requirements,
            sampler_outputs=sampler_outputs,
            rewards=rewards,
            advantages=advantages,
            prompts=train_prompts if train_prompts else prompts,
            sde_indices=sde_indices,
        )

    def _generate_nft_data(
        self,
        batch: Dict[str, Any],
        rollout_id: int,
        requirements: Any,
        actor_group: Any,
    ) -> TrainingBatch:
        """Generate training data for NFT (forward process algorithm).

        NFT doesn't need trajectories or log_probs - only clean latents x0.
        The forward diffusion happens in the loss function.

        Args:
            batch: Dict containing either:
                - Embedding mode: prompt_embeds, pooled_prompt_embeds, text_ids (optional), prompts
                - Text mode: prompts only
            rollout_id: Current rollout iteration number
        """
        prompts = batch.get("prompts", []) or []

        sampler_outputs, train_prompts, base_prompts = self.rollout_executor.sample(
            actor_group=actor_group,
            batch=batch,
            sde_indices=None,
        )
        self._attach_missing_embeddings_from_batch(
            sampler_outputs=sampler_outputs,
            batch=batch,
        )

        self.rollout_executor.validate_sampler_outputs(
            sampler_outputs=sampler_outputs,
            requirements=requirements,
            allow_replay=False,
            assert_step_alignment=False,
            mode_label="nft",
        )

        rewards, advantages, _ = self.rollout_executor.compute_reward_and_advantage(
            algorithm=self.algorithm,
            reward_service=self.reward_service,
            sampler_outputs=sampler_outputs,
            prompts=base_prompts if base_prompts else prompts,
            prompt_metadata=batch.get("metadata"),
        )

        return self.rollout_executor.assemble_training_batch(
            algorithm=self.algorithm,
            requirements=requirements,
            sampler_outputs=sampler_outputs,
            rewards=rewards,
            advantages=advantages,
            prompts=train_prompts if train_prompts else prompts,
            sde_indices=None,
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
        return self.rollout_executor.eval_batch(
            rollout_id=rollout_id,
            data_source=self.data_source,
            actor_group=actor_group,
            reward_service=self.reward_service,
        )

    def update_weights(
        self,
        state_dict_ref: ray.ObjectRef,
    ) -> None:
        """
        Update rollout actor weights from training.

        Args:
            state_dict_ref: ObjectRef containing new state dict
        """
        if self.rollout_actors is not None:
            self.rollout_actors.update_weights(state_dict_ref)

    def update_weights_from_path(
        self,
        checkpoint_path: str,
    ) -> None:
        """Update rollout actor weights from a shared checkpoint path."""
        if self.rollout_actors is not None and hasattr(self.rollout_actors, "update_weights_from_path"):
            self.rollout_actors.update_weights_from_path(checkpoint_path)

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
