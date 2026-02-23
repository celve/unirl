"""
diffusionrl Rollout Manager - Coordinates sampling, reward, and data conversion.

Supports:
- GRPO: Trajectory-based sampling with log probabilities
- MixGRPO: Mixed ODE/SDE sampling with timestep scheduler
- NFT: Forward process (only needs clean latents)

Reference: slime/ray/rollout.py
"""
import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple

import ray
import torch

from diffusionrl.ray.data_buffer import RolloutDataBuffer
from diffusionrl.ray.rollout_buffer import create_rollout_buffer_actor
from diffusionrl.runtime.rollout import RolloutPipelineExecutor
from diffusionrl.types.sampling import PromptEmbeddings, SamplerOutput
from diffusionrl.types.training_batch import TrainingBatch
from diffusionrl.utils import load_function

logger = logging.getLogger(__name__)


def _build_legacy_algorithm_kwargs(algorithm_cls: Any, args: Any) -> Dict[str, Any]:
    """Best-effort kwargs mapping for legacy plugins without from_args()."""
    kwargs: Dict[str, Any] = {}
    try:
        params = inspect.signature(algorithm_cls.__init__).parameters
    except (TypeError, ValueError):
        return kwargs

    for name, param in params.items():
        if name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if name == "clip_max" and hasattr(args, "advantage_clip_max"):
            kwargs[name] = getattr(args, "advantage_clip_max")
            continue
        if hasattr(args, name):
            kwargs[name] = getattr(args, name)
    return kwargs


def _invoke_custom_rollout_pipeline(
    pipeline_fn: Any,
    *,
    manager: Any,
    batch: Dict[str, Any],
    rollout_id: int,
    requirements: Any,
    actor_group: Any,
) -> Any:
    """Invoke custom rollout pipeline with flexible signature matching."""
    call_kwargs: Dict[str, Any] = {
        "manager": manager,
        "batch": batch,
        "rollout_id": rollout_id,
        "requirements": requirements,
        "actor_group": actor_group,
        "algorithm": manager.algorithm,
        "reward_service": manager.reward_service,
        "rollout_executor": manager.rollout_executor,
        "timestep_scheduler": manager.timestep_scheduler,
    }

    signature: Optional[inspect.Signature]
    try:
        signature = inspect.signature(pipeline_fn)
    except (TypeError, ValueError):
        signature = None

    if signature is not None:
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )
        try:
            if accepts_kwargs:
                return pipeline_fn(**call_kwargs)

            accepted_kwargs = {
                key: value
                for key, value in call_kwargs.items()
                if key in signature.parameters
            }
            if accepted_kwargs:
                return pipeline_fn(**accepted_kwargs)
        except TypeError:
            # Fall back to positional invocation below.
            pass

    return pipeline_fn(manager, batch, rollout_id, requirements, actor_group)


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
            pg_result: Optional placement group result for inference (pg, bundle_indices, gpu_ids)
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
        self.data_buffer = RolloutDataBuffer(
            partition_train_data=bool(getattr(args, "partition_train_data", True))
        )
        self.rollout_buffer = None
        self.rollout_buffer_enabled = bool(getattr(args, "rollout_buffer_enabled", True))
        self.rollout_pipeline_fn = None

        # Inference actor group
        self.inference_actors = None
        self.external_sampling_actors = None
        self._owns_inference_actors = False

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
        if hasattr(algorithm_cls, "from_args"):
            self.algorithm = algorithm_cls.from_args(self.args)
        else:
            legacy_kwargs = _build_legacy_algorithm_kwargs(algorithm_cls, self.args)
            logger.warning(
                "Algorithm %s does not implement from_args(); falling back to legacy kwargs matching.",
                self.args.algorithm_path,
            )
            self.algorithm = algorithm_cls(**legacy_kwargs)
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

        # 5. Create inference actors if placement group provided
        if self.pg_result is not None and getattr(self.args, "sampling_backend", "inference") != "training":
            self._create_inference_actors()

        if self.rollout_buffer_enabled:
            self.rollout_buffer = create_rollout_buffer_actor(self.args)
            logger.info("Rollout buffer actor created")
        else:
            logger.warning("Rollout buffer actor disabled; using legacy data_buffer.put handoff path.")

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

    def _create_inference_actors(self) -> None:
        """Create inference actor group from placement group."""
        from .groups.factory import create_inference_actor_group

        self.inference_actors = create_inference_actor_group(self.args, self.pg_result)
        self._owns_inference_actors = True
        logger.info("Inference actors created via create_inference_actor_group")

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
        actor_group = self.external_sampling_actors or self.inference_actors

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

    def generate(self, rollout_id: int, world_size: Optional[int] = None) -> Any:
        """
        Compatibility path: rollout + immediate handoff for training.

        Pipeline depends on algorithm type:
        - GRPO/MixGRPO: sampling -> reward -> advantages -> BackwardTrainingBatch
        - NFT: inference -> reward -> advantages -> ForwardTrainingBatch

        Args:
            rollout_id: Current rollout iteration number

        Returns:
            - ObjectRef containing typed TrainingBatch (BackwardTrainingBatch or ForwardTrainingBatch), or
            - List[ObjectRef] when partitioning is enabled and world_size is provided.
            Ray can serialize dataclasses directly via pickle.
        """
        train_data = self._build_training_batch(rollout_id)

        # Serialize and partition via rollout buffer actor (default),
        # with legacy data_buffer.put() fallback for compatibility.
        if self.rollout_buffer is not None:
            return ray.get(
                self.rollout_buffer.push_and_pop.remote(
                    rollout_id=rollout_id,
                    train_data=train_data,
                    world_size=world_size,
                    metadata={"step": int(self._current_step)},
                )
            )

        return self.data_buffer.put(train_data=train_data, world_size=world_size)

    def generate_and_buffer(self, rollout_id: int) -> Dict[str, Any]:
        """
        Generate one rollout and enqueue it into RolloutBufferActor.

        This is the data-centric path used by sync/async train loops to decouple
        rollout production from training consumption.
        """
        if self.rollout_buffer is None:
            raise RuntimeError(
                "generate_and_buffer requires rollout_buffer_enabled=true."
            )

        train_data = self._build_training_batch(rollout_id)
        push_result = ray.get(
            self.rollout_buffer.push.remote(
                rollout_id=rollout_id,
                train_data=train_data,
                metadata={"step": int(self._current_step)},
            )
        )
        if not push_result.get("accepted", False):
            raise RuntimeError(
                f"Rollout buffer rejected rollout_id={rollout_id}: {push_result.get('error')}"
            )
        return push_result

    def pop_training_data(self, world_size: Optional[int] = None) -> Any:
        """Pop next ready training batch from RolloutBufferActor."""
        if self.rollout_buffer is None:
            raise RuntimeError("pop_training_data requires rollout_buffer_enabled=true.")

        payload = ray.get(self.rollout_buffer.pop.remote(world_size=world_size))
        if payload is None:
            raise RuntimeError("Rollout buffer is empty; no training data available.")
        return payload

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
            bool(
                getattr(self.args, "replay_log_probs", False)
                or getattr(self.args, "fastvideo_replay_log_probs", False)
            )
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
        sampler_outputs: List[SamplerOutput],
        batch: Dict[str, Any],
    ) -> None:
        fallback_embeddings = self._build_prompt_embeddings_from_batch(batch)
        if fallback_embeddings is None:
            return

        offset = 0
        for idx, output in enumerate(sampler_outputs):
            if not isinstance(output, SamplerOutput):
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

        actor_group = self.external_sampling_actors or self.inference_actors
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
        Update inference actor weights from training.

        Args:
            state_dict_ref: ObjectRef containing new state dict
        """
        if self.inference_actors is not None:
            self.inference_actors.update_weights(state_dict_ref)

    def update_weights_from_path(
        self,
        checkpoint_path: str,
    ) -> None:
        """Update inference actor weights from a shared checkpoint path."""
        if self.inference_actors is not None and hasattr(self.inference_actors, "update_weights_from_path"):
            self.inference_actors.update_weights_from_path(checkpoint_path)

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
        stats = {
            "total_samples_generated": self._total_samples_generated,
        }
        if self.rollout_buffer is not None:
            try:
                stats["rollout_buffer"] = ray.get(self.rollout_buffer.get_stats.remote())
            except Exception as e:
                logger.warning("Failed to fetch rollout buffer stats: %s", e)
        return stats

    def offload(self) -> None:
        """Offload inference actors to CPU."""
        if self.inference_actors is not None:
            self.inference_actors.offload()

    def onload(self) -> None:
        """Load inference actors back to GPU."""
        if self.inference_actors is not None:
            self.inference_actors.onload()

    def load_weights(self) -> None:
        """Stage 1: load modules needed for weight update."""
        if self.inference_actors is not None and hasattr(self.inference_actors, "onload_weights"):
            self.inference_actors.onload_weights()
        else:
            self.onload()

    def after_weight_update(self) -> None:
        """Stage 2: run post-update restore hooks."""
        if self.inference_actors is not None and hasattr(self.inference_actors, "onload_post_update"):
            self.inference_actors.onload_post_update()

    def reload_runtime_cache(self) -> None:
        """Stage 3: restore runtime cache (KV/CUDA graph)."""
        if self.inference_actors is not None and hasattr(self.inference_actors, "onload_runtime_cache"):
            self.inference_actors.onload_runtime_cache()

    # Backward-compatible aliases (pre-cleanup naming).
    def onload_weights(self) -> None:
        self.load_weights()

    def onload_post_update(self) -> None:
        self.after_weight_update()

    def onload_runtime_cache(self) -> None:
        self.reload_runtime_cache()

    def dispose(self) -> None:
        """Clean up resources."""
        if self.rollout_buffer is not None:
            try:
                ray.get(self.rollout_buffer.clear.remote())
            except Exception:
                pass
            try:
                ray.kill(self.rollout_buffer)
            except Exception:
                pass
            self.rollout_buffer = None

        if self._owns_inference_actors and self.inference_actors is not None:
            self.inference_actors.dispose()
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
        pg_result: Placement group result for inference actors
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
