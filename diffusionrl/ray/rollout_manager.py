"""
diffusionrl Rollout Manager - Coordinates sampling, reward, and data conversion.

Supports:
- GRPO: Trajectory-based sampling with log probabilities
- MixGRPO: Mixed ODE/SDE sampling with timestep scheduler
- NFT: Forward process (only needs clean latents)

Reference: slime/ray/rollout.py
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Set, Union

import ray
import torch

from diffusionrl.types import (
    LogProbData,
    PromptEmbeddings,
    GRPOTrainingBatch,
    NFTTrainingBatch,
    TrainingBatch,
)
from diffusionrl.utils import load_function

logger = logging.getLogger(__name__)


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
            args: GRPOArguments instance
            pg_result: Optional placement group result for inference (pg, bundle_indices, gpu_ids)
            reward_pg_result: Optional placement group result for reward (pg, bundle_indices, gpu_ids)
        """
        self.args = args
        self.pg_result = pg_result
        self.reward_pg_result = reward_pg_result
        self._is_initialized = False

        # Components (loaded in init())
        self.algorithm = None
        self.sampler = None
        self.reward_service = None  # Changed from reward_worker to reward_service
        self.model_bundle = None
        self.data_source = None

        # Timestep scheduler for MixGRPO
        self.timestep_scheduler = None

        # Inference actor group
        self.inference_actors = None
        self.external_sampling_actors = None
        self._owns_inference_actors = False

        # Stats
        self._total_samples_generated = 0
        self._current_step = 0
        self._inference_weight_version = 0
        self._last_generated_weight_version: Optional[int] = None

    def init(self) -> None:
        """
        Initialize all components via dynamic loading.

        Components are loaded using load_function with paths from args.
        """
        logger.info("Initializing RolloutManager...")
        self._validate_batch_shape()

        # 1. Load algorithm with all required parameters
        algorithm_cls = load_function(self.args.algorithm_path)

        # Build algorithm kwargs
        algorithm_kwargs = {
            "clip_range": self.args.clip_range,
            "kl_coef": getattr(self.args, 'kl_coef', 0.01),
            "advantage_type": getattr(self.args, 'advantage_type', 'group'),
            "eta": getattr(self.args, 'eta', 1.0),
            "sde_type": getattr(self.args, 'sde_type', 'sde'),
            "clip_max": getattr(self.args, 'advantage_clip_max', None),  # Advantage clipping
        }

        # MixGRPOAlgorithm-specific: sde_ratio for mixed ODE/SDE sampling
        if hasattr(self.args, 'sde_ratio'):
            algorithm_kwargs["sde_ratio"] = self.args.sde_ratio

        # MixGRPO window_training: only train on SDE window timesteps
        window_training = getattr(self.args, 'window_training', False)
        algorithm_kwargs["window_training"] = window_training

        # Per-prompt tracker configuration (Flow-GRPO)
        use_per_prompt_tracker = getattr(self.args, 'use_per_prompt_stat_tracker', False)
        per_prompt_mode = getattr(self.args, 'per_prompt_mode', 'running')
        per_prompt_buffer_size = getattr(self.args, 'per_prompt_buffer_size', 16)
        per_prompt_min_count = getattr(self.args, 'per_prompt_min_count', 2)
        algorithm_kwargs["use_per_prompt_tracker"] = use_per_prompt_tracker
        algorithm_kwargs["per_prompt_mode"] = per_prompt_mode
        algorithm_kwargs["per_prompt_buffer_size"] = per_prompt_buffer_size
        algorithm_kwargs["per_prompt_min_count"] = per_prompt_min_count

        # Running statistics configuration (DanceGRPO cross-batch global normalization)
        use_running_stats = getattr(self.args, 'use_running_stats', False)
        running_stats_warmup = getattr(self.args, 'running_stats_warmup', 0)
        algorithm_kwargs["use_running_stats"] = use_running_stats
        algorithm_kwargs["running_stats_warmup"] = running_stats_warmup
        algorithm_kwargs["use_global_std"] = getattr(self.args, 'use_global_std', False)

        # MixGRPO stability controls (passed to algorithm for get_filtered_training_indices)
        algorithm_kwargs["ignore_last"] = getattr(self.args, 'ignore_last', False)
        algorithm_kwargs["frozen_init_timesteps"] = getattr(self.args, 'frozen_init_timesteps', 0)

        self.algorithm = algorithm_cls(**algorithm_kwargs)
        logger.info(f"Algorithm loaded: {self.args.algorithm_path} (clip_max={self.args.advantage_clip_max}, sde_ratio={getattr(self.args, 'sde_ratio', 'N/A')})")
        if getattr(self.args, "advantage_type", "group") == "per_prompt" and not use_per_prompt_tracker:
            logger.warning("advantage_type=per_prompt but use_per_prompt_stat_tracker=False; advantages will fall back to group normalization.")

        # 2. Load timestep scheduler based on algorithm requirements
        self._init_timestep_scheduler()

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

        self._is_initialized = True
        logger.info("RolloutManager initialized")

    def _init_reward_service(self) -> None:
        """Initialize reward service (single reward boundary)."""
        from diffusionrl.workers.reward.service import RewardService
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
        from .actor_group import create_inference_actor_group

        self.inference_actors = create_inference_actor_group(self.args, self.pg_result)
        self._owns_inference_actors = True
        logger.info("Inference actors created via create_inference_actor_group")

    def attach_sampling_actors(self, actor_group) -> None:
        """Attach external sampling actors (e.g., TrainingActorGroup)."""
        self.external_sampling_actors = actor_group
        logger.info("External sampling actors attached")

    def generate(self, rollout_id: int, world_size: Optional[int] = None) -> Any:
        """
        Complete data generation pipeline.

        Pipeline depends on algorithm type:
        - GRPO/MixGRPO: sampling -> reward -> advantages -> GRPOTrainingBatch
        - NFT: inference -> reward -> advantages -> NFTTrainingBatch

        Args:
            rollout_id: Current rollout iteration number

        Returns:
            - ObjectRef containing typed TrainingBatch (GRPOTrainingBatch or NFTTrainingBatch), or
            - List[ObjectRef] when partitioning is enabled and world_size is provided.
            Ray can serialize dataclasses directly via pickle.
        """
        if not self._is_initialized:
            raise RuntimeError("RolloutManager not initialized. Call init() first.")

        logger.info(f"Starting generation for rollout {rollout_id}")

        # 1. Get batch from data source (supports both prompts and embeddings)
        batch = self._get_batch()
        prompts = batch.get("prompts", [])

        # 2. Get algorithm requirements to determine pipeline
        requirements = self.algorithm.get_sampling_requirements()

        train_data: TrainingBatch
        if requirements.is_forward_process:
            # NFT: Forward process algorithm
            # Only need clean latents, not trajectories
            train_data = self._generate_nft_data(batch, rollout_id, requirements)
        else:
            # GRPO/MixGRPO: Trajectory-based algorithm
            train_data = self._generate_trajectory_data(batch, rollout_id, requirements)

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

        # Optional: partition training data by world_size to reduce object store pressure
        if getattr(self.args, "partition_train_data", True) and world_size:
            batch_size = getattr(train_data, "batch_size", None)
            if batch_size is None:
                logger.warning("Training batch does not expose batch_size; skipping partition.")
            else:
                per_rank = batch_size // world_size
                remainder = batch_size % world_size

                if per_rank == 0:
                    logger.warning(
                        "Batch size %d too small for world_size %d; skipping partition.",
                        batch_size,
                        world_size,
                    )
                else:
                    if remainder != 0:
                        logger.warning(
                            "Batch size %d not divisible by world_size %d; dropping %d samples for even partition.",
                            batch_size,
                            world_size,
                            remainder,
                        )
                    refs = []
                    for rank in range(world_size):
                        start = rank * per_rank
                        end = start + per_rank
                        part = train_data.slice(start, end)
                        refs.append(ray.put(part))
                    return refs

        # Put typed dataclass directly in object store
        # Ray can serialize dataclasses via pickle
        return ray.put(train_data)

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
    ) -> GRPOTrainingBatch:
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

        # Expand batch for K-repeat sampling (prompt-major)
        sampling_batch, train_prompts = self._expand_batch_for_sampling(batch)

        # Sample (distributed across inference actors)
        sampler_outputs, rewards = self._distributed_sample(sampling_batch, sde_indices=sde_indices)
        self._capture_sampling_weight_version(sampler_outputs)

        # Validate sampler outputs against algorithm requirements
        allow_fastvideo_replay = (
            bool(getattr(self.args, "fastvideo_replay_log_probs", False))
            and getattr(self.args, "sampler_engine_type", None) == "fastvideo"
            and getattr(self.args, "loss_type", "grpo") == "grpo"
        )
        replay_notice_emitted = False
        for idx, out in enumerate(sampler_outputs):
            try:
                meta = getattr(out, "metadata", {}) or {}
                generator_type = meta.get("generator_type") if isinstance(meta, dict) else None
                allow_missing_log_probs = bool(
                    allow_fastvideo_replay and generator_type == "fastvideo"
                )
                if allow_missing_log_probs and not replay_notice_emitted:
                    logger.warning(
                        "FastVideo replay path enabled: allowing missing rollout log_probs; "
                        "training actors will replay old log_probs before backward."
                    )
                    replay_notice_emitted = True
                out.validate_contract(
                    requires_log_probs=bool(requirements.requires_log_prob) and not allow_missing_log_probs,
                    requires_trajectory=bool(requirements.requires_trajectory),
                    requires_embeddings=bool(getattr(requirements, "requires_embeddings", True)),
                )
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
                if generator_type == "fastvideo":
                    hint = (
                        " FastVideo currently lacks full GRPO contract "
                        "(log_probs/prompt embeddings). Use FSDP sampling for training."
                    )
                raise RuntimeError(
                    f"Sampler output contract validation failed at index={idx}: {e}.{hint} "
                    f"capabilities={capabilities}, latents_shape={latents_shape}, "
                    f"trajectories_shape={traj_shape}, step_indices_shape={steps_shape}"
                ) from e

        # Compute rewards
        reward_components: Dict[str, List[float]] = {}
        if rewards is None:
            rewards, reward_components = self._compute_rewards(
                sampler_outputs,
                prompts,
                prompt_metadata=batch.get("metadata"),
            )

        # Compute advantages (pass prompts for per_prompt tracker support).
        # Optional component-wise aggregation is controlled by reward_mix_mode.
        advantages = self._compute_advantages(
            rewards=rewards,
            prompts=prompts,
            reward_components=reward_components,
        )
        # Convert to training data format
        return self._convert_to_train_data(
            sampler_outputs=sampler_outputs,
            rewards=rewards,
            advantages=advantages,
            prompts=train_prompts if train_prompts is not None else prompts,
            sde_indices=sde_indices,
        )

    def _generate_nft_data(
        self,
        batch: Dict[str, Any],
        rollout_id: int,
        requirements: Any,
    ) -> NFTTrainingBatch:
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

        # For NFT, we run inference to get clean images/videos
        # Then the loss function applies forward diffusion internally
        sampling_batch, train_prompts = self._expand_batch_for_sampling(batch)
        sampler_outputs, rewards = self._distributed_sample(sampling_batch, sde_indices=None)
        self._capture_sampling_weight_version(sampler_outputs)

        # Validate sampler outputs for NFT (needs clean latents + embeddings)
        for idx, out in enumerate(sampler_outputs):
            try:
                out.validate_contract(
                    requires_log_probs=bool(requirements.requires_log_prob),
                    requires_trajectory=bool(requirements.requires_trajectory),
                    requires_embeddings=bool(getattr(requirements, "requires_embeddings", True)),
                )
            except Exception as e:
                raise RuntimeError(
                    f"Sampler output contract validation failed in NFT path at index={idx}: {e}"
                ) from e

        # Compute rewards on clean outputs
        reward_components: Dict[str, List[float]] = {}
        if rewards is None:
            rewards, reward_components = self._compute_rewards(
                sampler_outputs,
                prompts,
                prompt_metadata=batch.get("metadata"),
            )

        # Compute advantages (pass prompts for per_prompt tracker support).
        # Optional component-wise aggregation is controlled by reward_mix_mode.
        advantages = self._compute_advantages(
            rewards=rewards,
            prompts=prompts,
            reward_components=reward_components,
        )

        # Convert to NFT training data format (clean_latents instead of trajectories)
        return self._convert_to_nft_train_data(
            sampler_outputs=sampler_outputs,
            rewards=rewards,
            advantages=advantages,
            prompts=train_prompts if train_prompts is not None else prompts,
        )

    def _expand_batch_for_sampling(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Optional[List[str]]]:
        """
        Expand batch for K-repeat sampling using prompt-major order.

        This repeats prompts/embeddings along the batch dimension so that
        sampling generates num_samples_per_prompt outputs per unique prompt.

        Returns:
            (expanded_batch, train_prompts)
        """
        k = getattr(self.args, "num_samples_per_prompt", 1)
        if k <= 1:
            return batch, batch.get("prompts")

        prompts = batch.get("prompts")
        base_size = None

        if prompts is not None:
            base_size = len(prompts)
        else:
            for key in ("prompt_embeds", "pooled_prompt_embeds", "text_ids"):
                val = batch.get(key)
                if torch.is_tensor(val):
                    base_size = val.shape[0]
                    break

        if base_size is None or base_size == 0:
            return batch, prompts

        train_prompts: Optional[List[str]] = None
        if prompts is not None:
            train_prompts = [p for p in prompts for _ in range(k)]

        expanded: Dict[str, Any] = dict(batch)
        if prompts is not None:
            expanded["prompts"] = train_prompts
        if "metadata" in expanded and isinstance(expanded["metadata"], list):
            metadata = expanded["metadata"]
            if len(metadata) == base_size:
                expanded["metadata"] = [m for m in metadata for _ in range(k)]

        def _repeat(value: Any) -> Any:
            if torch.is_tensor(value) and value.shape[0] == base_size:
                return value.repeat_interleave(k, dim=0)
            return value

        for key in (
            "prompt_embeds",
            "pooled_prompt_embeds",
            "negative_prompt_embeds",
            "negative_pooled_prompt_embeds",
            "text_ids",
            "image_ids",
            "encoder_attention_mask",
            "latents",
        ):
            if key in expanded:
                expanded[key] = _repeat(expanded[key])

        return expanded, train_prompts

    def _convert_to_nft_train_data(
        self,
        sampler_outputs: List[Dict[str, Any]],
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        prompts: List[str],
    ) -> NFTTrainingBatch:
        """Convert pipeline outputs to NFT training data format.

        NFT needs clean_latents (x0) instead of trajectories.

        Returns:
            NFTTrainingBatch: Typed training batch for NFT algorithm
        """
        # Extract clean latents from sampler outputs
        # Each output contains latents from one InferenceActor shard
        clean_latents = []
        all_prompt_embeds = []
        all_pooled_prompt_embeds = []
        all_encoder_attention_mask = []
        all_negative_prompt_embeds = []
        all_negative_pooled_prompt_embeds = []
        all_text_ids = []
        all_image_ids = []
        all_timesteps = []

        for output in sampler_outputs:
            if isinstance(output, dict):
                latents = output.get("latents")
                pe = output.get("prompt_embeds")
                ppe = output.get("pooled_prompt_embeds")
                eam = output.get("encoder_attention_mask")
                npe = output.get("negative_prompt_embeds")
                nppe = output.get("negative_pooled_prompt_embeds")
                tid = output.get("text_ids")
                iid = output.get("image_ids")
                ts = output.get("timesteps")
            elif hasattr(output, "latents"):
                latents = output.latents
                if hasattr(output, "embeddings") and output.embeddings is not None:
                    pe = output.embeddings.prompt_embeds
                    ppe = output.embeddings.pooled_prompt_embeds
                    eam = output.embeddings.encoder_attention_mask
                    npe = output.embeddings.negative_prompt_embeds
                    nppe = output.embeddings.negative_pooled_prompt_embeds
                    tid = output.embeddings.text_ids
                    iid = output.embeddings.image_ids
                else:
                    pe = getattr(output, "prompt_embeds", None)
                    ppe = getattr(output, "pooled_prompt_embeds", None)
                    eam = getattr(output, "encoder_attention_mask", None)
                    npe = getattr(output, "negative_prompt_embeds", None)
                    nppe = getattr(output, "negative_pooled_prompt_embeds", None)
                    tid = getattr(output, "text_ids", None)
                    iid = getattr(output, "image_ids", None)
                ts = getattr(output, "timesteps", None)
            else:
                latents = output
                pe = ppe = eam = npe = nppe = tid = iid = None
                ts = None

            if latents is not None:
                clean_latents.append(latents)
            if pe is not None:
                all_prompt_embeds.append(pe)
            if ppe is not None:
                all_pooled_prompt_embeds.append(ppe)
            if eam is not None:
                all_encoder_attention_mask.append(eam)
            if npe is not None:
                all_negative_prompt_embeds.append(npe)
            if nppe is not None:
                all_negative_pooled_prompt_embeds.append(nppe)
            if tid is not None:
                all_text_ids.append(tid)
            if iid is not None:
                all_image_ids.append(iid)
            if ts is not None:
                all_timesteps.append(ts)

        # Concatenate clean latents along batch dimension (not stack!)
        # Each actor returns [shard_batch, C, H, W], we need [total_batch, C, H, W]
        if clean_latents:
            clean_latents_tensor = torch.cat(clean_latents, dim=0)
        else:
            raise ValueError("No clean latents found in sampler outputs")

        # Concatenate embeddings from all outputs
        prompt_embeds = torch.cat(all_prompt_embeds, dim=0) if all_prompt_embeds else None
        pooled_prompt_embeds = torch.cat(all_pooled_prompt_embeds, dim=0) if all_pooled_prompt_embeds else None
        encoder_attention_mask = (
            torch.cat(all_encoder_attention_mask, dim=0)
            if all_encoder_attention_mask else None
        )
        negative_prompt_embeds = (
            torch.cat(all_negative_prompt_embeds, dim=0)
            if all_negative_prompt_embeds else None
        )
        negative_pooled_prompt_embeds = (
            torch.cat(all_negative_pooled_prompt_embeds, dim=0)
            if all_negative_pooled_prompt_embeds else None
        )
        # text_ids: For FLUX, these are [B, seq, 3] per actor, so concatenate along batch dim
        text_ids = torch.cat(all_text_ids, dim=0) if all_text_ids else None
        # image_ids: For FLUX, these are [num_patches, 3] and SHARED across all samples
        # Do NOT concatenate - just take the first one (they're all identical position encodings)
        image_ids = all_image_ids[0] if all_image_ids else None
        timesteps = all_timesteps[0] if all_timesteps else None

        if prompt_embeds is None:
            raise ValueError("No prompt embeddings found in sampler outputs")

        # Create typed embeddings
        embeddings = PromptEmbeddings(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            encoder_attention_mask=encoder_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            text_ids=text_ids,
            image_ids=image_ids,
        )

        # Create typed batch
        batch = NFTTrainingBatch(
            clean_latents=clean_latents_tensor,
            advantages=advantages,
            embeddings=embeddings,
            rewards=rewards,
            prompts=prompts,
            timesteps=timesteps,
            sampling_weight_version=self._last_generated_weight_version,
        )

        # Validate before returning
        batch.validate()

        return batch

    def _get_batch(self) -> Dict[str, Any]:
        """
        Get training batch from data source.

        Returns:
            Dict containing either:
            - Embedding mode: {"prompt_embeds": Tensor, "pooled_prompt_embeds": Tensor,
                             "text_ids": Tensor (optional), "prompts": List[str]}
            - Text mode: {"prompts": List[str]}
        """
        if self.data_source is not None:
            batch_size = getattr(self.args, "prompts_per_batch", None) or self.args.batch_size
            samples = self.data_source.get_samples(batch_size)
            # If data source returns a dict, use it directly
            if isinstance(samples, dict):
                return samples
            # If data source returns a list of strings, wrap it
            return {"prompts": samples}

        # Default prompts for testing
        default_prompts = [
            "A beautiful sunset over the ocean",
            "A cat playing with a ball of yarn",
            "A mountain landscape with snow",
            "A futuristic city at night",
        ]
        batch_size = getattr(self.args, "prompts_per_batch", None) or self.args.batch_size
        return {"prompts": default_prompts[:batch_size]}

    def _get_prompts(self) -> List[str]:
        """
        Get prompts for this batch.

        DEPRECATED: Use _get_batch() for full embedding support.
        Kept for backward compatibility.
        """
        batch = self._get_batch()
        return batch.get("prompts", [])

    def _distributed_sample(
        self,
        batch: Union[List[str], Dict[str, Any]],
        sde_indices: Optional[Set[int]] = None,
    ) -> Tuple[List[Any], Optional[torch.Tensor]]:
        """
        Sample across distributed inference actors.

        Args:
            batch: Either:
                - List[str]: List of text prompts (legacy)
                - Dict: Batch containing prompts and/or pre-computed embeddings
            sde_indices: Set of timestep indices for SDE sampling (MixGRPO).
                If None, all timesteps use SDE (standard GRPO).

        Returns:
            Tuple of (sampler_outputs, rewards_tensor_or_None)
        """
        actor_group = self.external_sampling_actors or self.inference_actors
        if actor_group is None:
            raise RuntimeError("No sampling actors available")

        # Handle legacy list format
        if isinstance(batch, list):
            batch = {"prompts": batch}

        # Check if we have pre-computed embeddings
        has_embeddings = "prompt_embeds" in batch

        # Get init_same_noise configuration (DanceGRPO/MixGRPO)
        init_same_noise = getattr(self.args, 'init_same_noise', False)
        num_samples_per_prompt = getattr(self.args, 'num_samples_per_prompt', 1)

        # Generate across actors with sde_indices for mixed sampling
        if has_embeddings:
            # Embedding mode: pass pre-computed embeddings
            gen_kwargs = dict(
                prompts=batch.get("prompts"),
                prompt_embeds=batch.get("prompt_embeds"),
                pooled_prompt_embeds=batch.get("pooled_prompt_embeds"),
                text_ids=batch.get("text_ids"),
                num_inference_steps=self.args.num_inference_steps,
                guidance_scale=self.args.guidance_scale,
                height=self.args.height,
                width=self.args.width,
                num_frames=self.args.num_frames,
                sde_indices=sde_indices,
                decode_for_reward=True,  # Request decoded images for reward computation
                init_same_noise=init_same_noise,
                num_samples_per_prompt=num_samples_per_prompt,
            )
        else:
            # Text mode: pass prompts for on-the-fly encoding
            gen_kwargs = dict(
                prompts=batch.get("prompts", []),
                num_inference_steps=self.args.num_inference_steps,
                guidance_scale=self.args.guidance_scale,
                height=self.args.height,
                width=self.args.width,
                num_frames=self.args.num_frames,
                sde_indices=sde_indices,
                decode_for_reward=True,  # Request decoded images for reward computation
                init_same_noise=init_same_noise,
                num_samples_per_prompt=num_samples_per_prompt,
            )

        rewards_tensor: Optional[torch.Tensor] = None
        if hasattr(actor_group, "sample_batch"):
            outputs = actor_group.sample_batch(**gen_kwargs)
        else:
            outputs = actor_group.generate(**gen_kwargs)

        # Merge outputs from all actors
        merged_outputs = []
        for output in outputs:
            if hasattr(output, '__iter__') and not isinstance(output, dict):
                merged_outputs.extend(output)
            else:
                merged_outputs.append(output)

        return merged_outputs, rewards_tensor

    def _compute_rewards(
        self,
        sampler_outputs: List[Dict[str, Any]],
        prompts: List[str],
        prompt_metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """
        Compute rewards for generated samples using batch processing.
        Uses RewardService to compute rewards in a batched manner.

        Args:
            sampler_outputs: List of sampler outputs
            prompts: List of text prompts

        Returns:
            Tuple of:
                - Tensor of rewards [batch_size]
                - Reward components by worker/model name
        """
        # Use RewardService
        from diffusionrl.workers.reward.base import RewardRequest

        # Batch media and prompts together
        all_images = []
        all_videos = []
        all_prompts = []
        all_metadata: List[Optional[Dict[str, Any]]] = []

        # Get num_samples_per_prompt for correct prompt-sample alignment
        num_samples_per_prompt = getattr(self.args, 'num_samples_per_prompt', 1)

        # Maintain a running sample index so that even when one sampler_output
        # contains multiple outputs, we still map to prompts in
        # prompt-major order: [p0 xK][p1 xK]...
        sample_idx = 0

        def _append_media(items: List[Any], target: List[Any]) -> None:
            nonlocal sample_idx
            for item in items:
                prompt_idx = sample_idx // num_samples_per_prompt
                prompt = prompts[prompt_idx % len(prompts)] if prompts else ""
                target.append(item)
                all_prompts.append(prompt)
                if prompt_metadata and len(prompt_metadata) > 0:
                    all_metadata.append(prompt_metadata[prompt_idx % len(prompt_metadata)])
                else:
                    all_metadata.append(None)
                sample_idx += 1

        prefer_video_inputs = self._reward_prefers_video_inputs()
        if prefer_video_inputs:
            all_videos = []
            for output in sampler_outputs:
                videos = self._extract_videos_from_output(output)
                if not videos:
                    # Fallback to image mode when video payload is unavailable.
                    prefer_video_inputs = False
                    all_videos = []
                    all_prompts = []
                    all_metadata = []
                    sample_idx = 0
                    break
                _append_media(videos, all_videos)

        if not prefer_video_inputs:
            for output in sampler_outputs:
                images = self._extract_images_from_output(output)
                _append_media(images, all_images)

        # Handle case where no media were extracted
        if not all_images and not all_videos:
            logger.warning("No media extracted from sampler outputs")
            return torch.zeros(len(sampler_outputs), dtype=torch.float32), {}

        request_kwargs: Dict[str, Any] = {
            "prompts": all_prompts,
            "metadata": all_metadata if any(m is not None for m in all_metadata) else None,
        }
        if prefer_video_inputs and all_videos:
            request_kwargs["videos"] = all_videos
        else:
            request_kwargs["images"] = all_images
        request = RewardRequest(**request_kwargs)

        # Compute rewards in single batch call
        response = self.reward_service.compute_rewards(request)

        return torch.tensor(response.rewards, dtype=torch.float32), response.reward_components

    def _capture_sampling_weight_version(self, sampler_outputs: List[Any]) -> Optional[int]:
        """
        Capture and validate per-rollout sampling weight version from actor metadata.
        """
        versions: Set[int] = set()
        for output in sampler_outputs:
            metadata = None
            if isinstance(output, dict):
                metadata = output.get("metadata")
            else:
                metadata = getattr(output, "metadata", None)
            if not isinstance(metadata, dict):
                continue
            if "weight_version" not in metadata:
                continue
            try:
                versions.add(int(metadata["weight_version"]))
            except Exception:
                logger.warning("Invalid weight_version in sampler metadata: %s", metadata.get("weight_version"))

        if not versions:
            return None
        if len(versions) != 1:
            raise RuntimeError(
                f"Sampler outputs have inconsistent weight versions: {sorted(versions)}"
            )

        version = int(next(iter(versions)))
        self._last_generated_weight_version = version
        if int(self._inference_weight_version) != int(version):
            logger.warning(
                "Sampling used weight_version=%s while manager expects=%s",
                version,
                self._inference_weight_version,
            )
        return version

    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        prompts: List[str],
        reward_components: Optional[Dict[str, List[float]]] = None,
    ) -> torch.Tensor:
        """
        Compute advantages from reward tensor.

        Default path (`reward_mix_mode=reward_aggr`) uses aggregated rewards directly.
        Optional path (`reward_mix_mode=advantage_aggr`) computes advantages per reward
        component and aggregates them with reward worker weights.
        """
        reward_mix_mode = getattr(self.args, "reward_mix_mode", "reward_aggr")
        if reward_mix_mode != "advantage_aggr" or not reward_components:
            return self.algorithm.compute_advantages(
                rewards=rewards,
                num_samples_per_prompt=self.args.num_samples_per_prompt,
                prompts=prompts,
            )

        weights = self._get_reward_component_weights(reward_components)
        weighted_advantages = torch.zeros_like(rewards)
        total_weight = 0.0

        for component_name, component_rewards in reward_components.items():
            component_tensor = torch.tensor(
                component_rewards,
                dtype=rewards.dtype,
                device=rewards.device,
            )
            if component_tensor.shape != rewards.shape:
                logger.warning(
                    "Skipping reward component %s due to shape mismatch: expected %s, got %s",
                    component_name,
                    tuple(rewards.shape),
                    tuple(component_tensor.shape),
                )
                continue

            component_advantages = self.algorithm.compute_advantages(
                rewards=component_tensor,
                num_samples_per_prompt=self.args.num_samples_per_prompt,
                prompts=prompts,
            )
            weight = float(weights.get(component_name, 1.0))
            weighted_advantages += component_advantages * weight
            total_weight += weight

        if total_weight <= 0:
            logger.warning(
                "reward_mix_mode=advantage_aggr but no valid reward components; "
                "falling back to aggregated reward advantages."
            )
            return self.algorithm.compute_advantages(
                rewards=rewards,
                num_samples_per_prompt=self.args.num_samples_per_prompt,
                prompts=prompts,
            )

        return weighted_advantages / total_weight

    def _get_reward_component_weights(
        self,
        reward_components: Dict[str, List[float]],
    ) -> Dict[str, float]:
        """Map reward component name to configured worker weight."""
        default_weights = {name: 1.0 for name in reward_components.keys()}
        if self.reward_service is None or not hasattr(self.reward_service, "workers"):
            return default_weights

        for worker in self.reward_service.workers:
            model_name = worker.get_model_name()
            if model_name in default_weights:
                default_weights[model_name] = float(worker.get_weight())
        return default_weights

    def _extract_images_from_output(self, output: Any) -> List[Any]:
        """
        Extract images from a sampler output.

        Args:
            output: Sampler output (dict or object)

        Returns:
            List of images (PIL.Image or tensors)
        """
        if isinstance(output, dict):
            # Dictionary output
            decoded = output.get("decoded_images")
            if decoded is not None and len(decoded) > 0:
                return decoded if isinstance(decoded, list) else [decoded]

            latents = output.get("latents")
            if latents is not None:
                # Split batch latents into individual images if batched
                if isinstance(latents, torch.Tensor) and latents.dim() >= 3:
                    return [lat for lat in latents]
                return [latents]

            return []

        elif hasattr(output, "decoded_images") and output.decoded_images is not None:
            # SamplerOutput with decoded images
            return output.decoded_images if isinstance(output.decoded_images, list) else [output.decoded_images]

        elif hasattr(output, "latents") and output.latents is not None:
            # SamplerOutput with latents only
            # Split batch latents into individual images if batched
            latents = output.latents
            if isinstance(latents, torch.Tensor) and latents.dim() >= 3:
                return [lat for lat in latents]
            return [latents]

        else:
            # Raw output (tensor or image)
            return [output]

    def _extract_videos_from_output(self, output: Any) -> List[torch.Tensor]:
        """Extract decoded videos (preferred) or 5D tensors as video payload."""
        decoded_videos = None
        latents = None

        if isinstance(output, dict):
            decoded_videos = output.get("decoded_videos")
            metadata = output.get("metadata")
            if decoded_videos is None and isinstance(metadata, dict):
                decoded_videos = metadata.get("decoded_videos")
            latents = output.get("latents")
        else:
            metadata = getattr(output, "metadata", None)
            if isinstance(metadata, dict):
                decoded_videos = metadata.get("decoded_videos")
            latents = getattr(output, "latents", None)

        if torch.is_tensor(decoded_videos):
            if decoded_videos.dim() >= 5:
                return [video for video in decoded_videos]
            if decoded_videos.dim() == 4:
                return [decoded_videos]

        if torch.is_tensor(latents):
            if latents.dim() >= 5:
                return [video for video in latents]
            if latents.dim() == 4:
                return [latents]

        return []

    def _reward_prefers_video_inputs(self) -> bool:
        """Best-effort switch for video-native reward workers."""
        reward_path = str(getattr(self.args, "reward_path", "") or "")
        return "VideoRewardWorker" in reward_path

    def _convert_to_train_data(
        self,
        sampler_outputs: List[Dict[str, Any]],
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        prompts: List[str],
        sde_indices: Optional[Set[int]] = None,
    ) -> GRPOTrainingBatch:
        """
        Convert pipeline outputs to training data format.

        Args:
            sampler_outputs: List of sampler outputs
            rewards: Reward tensor [B]
            advantages: Advantage tensor [B]
            prompts: List of prompts
            sde_indices: Set of SDE timestep indices (from scheduler)

        Returns:
            GRPOTrainingBatch: Typed training batch for GRPO/MixGRPO
        """
        # Extract trajectories, log_probs, embeddings from all actor outputs
        # Each actor returns results for a shard of the batch
        trajectories = []
        log_probs_dicts = []
        timesteps = None
        step_indices = None
        final_sde_indices = sde_indices if sde_indices is not None else set()
        all_prompt_embeds = []
        all_pooled_prompt_embeds = []
        all_encoder_attention_mask = []
        all_negative_prompt_embeds = []
        all_negative_pooled_prompt_embeds = []
        all_text_ids = []
        all_image_ids = []

        for output in sampler_outputs:
            if isinstance(output, dict):
                traj = output.get("trajectories")
                log_probs = output.get("log_probs_dict", {})
                ts = output.get("timesteps")
                steps = output.get("step_indices")
                if steps is not None and not torch.is_tensor(steps):
                    steps = torch.tensor(steps, dtype=torch.long)
                sde_idx = output.get("sde_indices", set())
                pe = output.get("prompt_embeds")
                ppe = output.get("pooled_prompt_embeds")
                eam = output.get("encoder_attention_mask")
                npe = output.get("negative_prompt_embeds")
                nppe = output.get("negative_pooled_prompt_embeds")
                tid = output.get("text_ids")
                iid = output.get("image_ids")
            elif hasattr(output, "trajectories"):
                traj = output.trajectories
                # Handle typed LogProbData or legacy dict
                if hasattr(output, "log_probs") and output.log_probs is not None:
                    if hasattr(output.log_probs, "to_dict"):
                        log_probs = output.log_probs.to_dict()
                    else:
                        log_probs = output.log_probs
                else:
                    log_probs = {}
                ts = output.timesteps
                steps = getattr(output, "step_indices", None)
                if steps is not None and not torch.is_tensor(steps):
                    steps = torch.tensor(steps, dtype=torch.long)
                sde_idx = output.sde_indices if hasattr(output, "sde_indices") else set()
                if hasattr(output, "embeddings") and output.embeddings is not None:
                    pe = output.embeddings.prompt_embeds
                    ppe = output.embeddings.pooled_prompt_embeds
                    eam = output.embeddings.encoder_attention_mask
                    npe = output.embeddings.negative_prompt_embeds
                    nppe = output.embeddings.negative_pooled_prompt_embeds
                    tid = output.embeddings.text_ids
                    iid = output.embeddings.image_ids
                else:
                    pe = getattr(output, "prompt_embeds", None)
                    ppe = getattr(output, "pooled_prompt_embeds", None)
                    eam = getattr(output, "encoder_attention_mask", None)
                    npe = getattr(output, "negative_prompt_embeds", None)
                    nppe = getattr(output, "negative_pooled_prompt_embeds", None)
                    tid = getattr(output, "text_ids", None)
                    iid = getattr(output, "image_ids", None)
            else:
                traj = None
                log_probs = {}
                ts = None
                steps = None
                sde_idx = set()
                pe = ppe = eam = npe = nppe = tid = iid = None

            if traj is not None:
                trajectories.append(traj)
            log_probs_dicts.append(log_probs)
            if ts is not None and timesteps is None:
                timesteps = ts
            elif ts is not None and timesteps is not None and not torch.equal(timesteps.to(ts.device), ts):
                raise ValueError("Mismatched timesteps across sampler outputs")
            if steps is not None:
                if step_indices is None:
                    step_indices = steps
                elif not torch.equal(step_indices.to(steps.device), steps):
                    raise ValueError(
                        "Mismatched step_indices across sampler outputs: "
                        f"expected={step_indices.tolist()} got={steps.tolist()}"
                    )
            # Only use sampler's sde_indices if not provided by scheduler
            if sde_indices is None:
                final_sde_indices.update(sde_idx)
            # Collect embeddings from all outputs
            if pe is not None:
                all_prompt_embeds.append(pe)
            if ppe is not None:
                all_pooled_prompt_embeds.append(ppe)
            if eam is not None:
                all_encoder_attention_mask.append(eam)
            if npe is not None:
                all_negative_prompt_embeds.append(npe)
            if nppe is not None:
                all_negative_pooled_prompt_embeds.append(nppe)
            if tid is not None:
                all_text_ids.append(tid)
            if iid is not None:
                all_image_ids.append(iid)

        # Concatenate trajectories along batch dimension (not stack!)
        # Each actor returns [shard_batch, num_steps, C, H, W]
        if trajectories:
            trajectories_tensor = torch.cat(trajectories, dim=0)
        else:
            raise ValueError("No trajectories found in sampler outputs")

        if timesteps is None:
            raise ValueError("No timesteps found in sampler outputs")
        if step_indices is None:
            step_indices = torch.arange(
                timesteps.shape[0],
                device=timesteps.device,
                dtype=torch.long,
            )

        # Apply algorithm-defined training indices (e.g., MixGRPO window training)
        if hasattr(self.algorithm, "get_training_indices"):
            train_indices = self.algorithm.get_training_indices(len(timesteps) - 1)
            if sde_indices is None:
                final_sde_indices = train_indices
            else:
                final_sde_indices = final_sde_indices & train_indices
            # Guard against empty intersection (would skip training)
            if len(final_sde_indices) == 0:
                logger.warning(
                    "Training timestep set is empty after intersecting scheduler and algorithm indices. "
                    "Falling back to algorithm-provided indices."
                )
                final_sde_indices = train_indices if len(train_indices) > 0 else set(range(len(timesteps) - 1))

        # Apply algorithm-defined timestep filtering (ignore_last, frozen_init_timesteps, etc.)
        # This delegates to algorithm.get_filtered_training_indices() which handles:
        # - ignore_last: Skip the last timestep (t->0) which has unstable log_prob
        # - frozen_init_timesteps: Skip early timesteps with high variance
        # - Any other algorithm-specific filtering logic
        num_steps = len(timesteps) - 1
        final_sde_indices = self.algorithm.get_filtered_training_indices(
            final_sde_indices, num_steps
        )

        if len(final_sde_indices) == 0:
            logger.warning(
                "Training timestep set is empty after algorithm filtering; "
                "falling back to all timesteps."
            )
            final_sde_indices = set(int(i) for i in step_indices[:-1].tolist())

        # Merge log_probs_dicts by concatenating along batch dimension
        merged_log_probs: Dict[int, torch.Tensor] = {}
        if log_probs_dicts:
            # Get all timestep indices
            all_indices: Set[int] = set()
            for lpd in log_probs_dicts:
                all_indices.update(lpd.keys())

            for idx in all_indices:
                values = []
                for lpd in log_probs_dicts:
                    if idx in lpd:
                        values.append(lpd[idx])
                if values:
                    # Concatenate along batch dimension
                    merged_log_probs[idx] = torch.cat(values, dim=0)

        if final_sde_indices:
            merged_log_probs = {
                int(idx): value
                for idx, value in merged_log_probs.items()
                if int(idx) in set(int(i) for i in final_sde_indices)
            }

        # Concatenate embeddings from all outputs
        prompt_embeds = torch.cat(all_prompt_embeds, dim=0) if all_prompt_embeds else None
        pooled_prompt_embeds = torch.cat(all_pooled_prompt_embeds, dim=0) if all_pooled_prompt_embeds else None
        encoder_attention_mask = (
            torch.cat(all_encoder_attention_mask, dim=0)
            if all_encoder_attention_mask else None
        )
        negative_prompt_embeds = (
            torch.cat(all_negative_prompt_embeds, dim=0)
            if all_negative_prompt_embeds else None
        )
        negative_pooled_prompt_embeds = (
            torch.cat(all_negative_pooled_prompt_embeds, dim=0)
            if all_negative_pooled_prompt_embeds else None
        )
        # text_ids: For FLUX, these are [B, seq, 3] per actor, so concatenate along batch dim
        text_ids = torch.cat(all_text_ids, dim=0) if all_text_ids else None
        # image_ids: For FLUX, these are [num_patches, 3] and SHARED across all samples
        # Do NOT concatenate - just take the first one (they're all identical position encodings)
        image_ids = all_image_ids[0] if all_image_ids else None

        if prompt_embeds is None:
            raise ValueError("No prompt embeddings found in sampler outputs")

        # Create typed embeddings
        embeddings = PromptEmbeddings(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            encoder_attention_mask=encoder_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            text_ids=text_ids,
            image_ids=image_ids,
        )

        # Create typed batch
        batch = GRPOTrainingBatch(
            trajectories=trajectories_tensor,
            log_probs=LogProbData.from_dict(merged_log_probs),
            timesteps=timesteps,
            advantages=advantages,
            embeddings=embeddings,
            rewards=rewards,
            prompts=prompts,
            num_steps=self.args.num_inference_steps,
            step_indices=step_indices,
            target_sde_indices=set(int(i) for i in final_sde_indices),
            sampling_weight_version=self._last_generated_weight_version,
        )

        # Validate before returning
        batch.validate()

        return batch

    def eval(self, rollout_id: int) -> Dict[str, Any]:
        """
        Run evaluation generation.

        Args:
            rollout_id: Current rollout iteration

        Returns:
            Dictionary of evaluation metrics and samples
        """
        logger.info(f"Running evaluation for rollout {rollout_id}")

        # Get eval prompts
        if self.data_source is not None and hasattr(self.data_source, "get_eval_samples"):
            prompts = self.data_source.get_eval_samples(self.args.eval_batch_size)
        else:
            prompts = self._get_prompts()[:self.args.eval_batch_size]

        # Generate samples
        outputs, rewards = self._distributed_sample(prompts)

        # Compute rewards (unless colocate reward already returned them)
        if rewards is None:
            rewards, _ = self._compute_rewards(outputs, prompts)

        return {
            "rollout_id": rollout_id,
            "num_samples": len(prompts),
            "mean_reward": rewards.mean().item(),
            "std_reward": rewards.std().item(),
            "prompts": prompts,
        }

    def update_weights(
        self,
        state_dict_ref: ray.ObjectRef,
        weight_version: Optional[int] = None,
    ) -> List[int]:
        """
        Update inference actor weights from training.

        Args:
            state_dict_ref: ObjectRef containing new state dict
        """
        if self.inference_actors is not None:
            versions = self.inference_actors.update_weights(
                state_dict_ref,
                weight_version=weight_version,
            )
            if versions:
                self._inference_weight_version = int(versions[0])
            return [int(v) for v in versions]
        return []

    def update_weights_from_path(
        self,
        checkpoint_path: str,
        weight_version: Optional[int] = None,
    ) -> List[int]:
        """Update inference actor weights from a shared checkpoint path."""
        if self.inference_actors is not None and hasattr(self.inference_actors, "update_weights_from_path"):
            versions = self.inference_actors.update_weights_from_path(
                checkpoint_path,
                weight_version=weight_version,
            )
            if versions:
                self._inference_weight_version = int(versions[0])
            return [int(v) for v in versions]
        return []

    def get_inference_weight_version(self) -> int:
        """Get last known synchronized inference weight version."""
        return int(self._inference_weight_version)

    def get_last_generated_weight_version(self) -> Optional[int]:
        """Get sampling weight version from most recent rollout generation."""
        return self._last_generated_weight_version

    def assert_inference_weight_version(self, expected_version: int, strict: bool = True) -> bool:
        """
        Validate inference actor weight version consistency.
        """
        if self.inference_actors is None:
            return True
        if not hasattr(self.inference_actors, "get_weight_versions"):
            return True

        versions = self.inference_actors.get_weight_versions()
        if not versions:
            return True

        unique = sorted(set(int(v) for v in versions))
        if len(unique) != 1:
            raise RuntimeError(f"Inference actors have inconsistent weight versions: {unique}")
        current = int(unique[0])
        self._inference_weight_version = current

        if current != int(expected_version):
            msg = (
                f"Inference weight version mismatch: expected={int(expected_version)}, "
                f"actual={current}"
            )
            if strict:
                raise RuntimeError(msg)
            logger.warning(msg)
            return False
        return True

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

    def offload(self) -> None:
        """Offload inference actors to CPU."""
        if self.inference_actors is not None:
            self.inference_actors.offload()

    def onload(self) -> None:
        """Load inference actors back to GPU."""
        if self.inference_actors is not None:
            self.inference_actors.onload()

    def onload_weights(self) -> None:
        """Stage 1 onload: modules needed for weight update."""
        if self.inference_actors is not None and hasattr(self.inference_actors, "onload_weights"):
            self.inference_actors.onload_weights()
        else:
            self.onload()

    def onload_post_update(self) -> None:
        """Stage 2 onload: post-update restore hooks."""
        if self.inference_actors is not None and hasattr(self.inference_actors, "onload_post_update"):
            self.inference_actors.onload_post_update()

    def onload_runtime_cache(self) -> None:
        """Stage 3 onload: runtime cache restore (KV/CUDA graph)."""
        if self.inference_actors is not None and hasattr(self.inference_actors, "onload_runtime_cache"):
            self.inference_actors.onload_runtime_cache()

    def dispose(self) -> None:
        """Clean up resources."""
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
        args: GRPOArguments instance
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
