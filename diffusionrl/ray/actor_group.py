"""
diffusionrl Actor Group Management.

Reference: slime/ray/actor_group.py
"""
import logging
from typing import Any, Dict, List, Optional, Type

import ray
import torch
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

logger = logging.getLogger(__name__)


class BaseActorGroup:
    """
    Base class for managing groups of Ray actors.

    Handles actor creation, initialization, and lifecycle management.
    """

    def __init__(
        self,
        actor_class: Type,
        num_actors: int,
        pg: PlacementGroup,
        bundle_indices: List[int],
        gpu_ids: Optional[List[int]] = None,
        num_gpus_per_actor: float = 1.0,
        num_cpus_per_actor: float = 1.0,
        num_gpus_per_engine: int = 1,
        capture_child_tasks: bool = False,
        runtime_env: Optional[dict] = None,
        **actor_kwargs,
    ):
        """
        Initialize actor group.

        Args:
            actor_class: Ray actor class to instantiate
            num_actors: Number of actors to create
            pg: Placement group to schedule actors on
            bundle_indices: Bundle indices for each actor
            gpu_ids: Physical GPU IDs for each bundle (for Slime multi-GPU pattern)
            num_gpus_per_actor: GPUs per actor (Ray resource claim)
            num_cpus_per_actor: CPUs per actor
            num_gpus_per_engine: Actual GPUs each engine needs (>1 for Slime pattern)
            capture_child_tasks: Whether child processes inherit PG scheduling
            runtime_env: Runtime environment variables (e.g. NOSET env vars)
            **actor_kwargs: Additional kwargs passed to actor constructor
        """
        self.actor_class = actor_class
        self.num_actors = num_actors
        self.pg = pg
        self.bundle_indices = bundle_indices
        self._actor_handles: List[ray.actor.ActorHandle] = []

        # Create actors
        for i in range(num_actors):
            if num_gpus_per_engine > 1 and gpu_ids is not None:
                # Slime pattern: multi-GPU actor uses fractional GPU claim + base_gpu_id
                bi = bundle_indices[i * num_gpus_per_engine]
                base_gpu_id = gpu_ids[i * num_gpus_per_engine]
                actor_kwargs_i = {**actor_kwargs, "base_gpu_id": base_gpu_id}
            else:
                # Standard: single GPU actor
                bi = bundle_indices[i] if i < len(bundle_indices) else i
                actor_kwargs_i = actor_kwargs

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=bi,
                placement_group_capture_child_tasks=capture_child_tasks,
            )

            options = {
                "num_gpus": num_gpus_per_actor,
                "num_cpus": num_cpus_per_actor,
                "scheduling_strategy": scheduling_strategy,
            }
            if runtime_env:
                options["runtime_env"] = runtime_env

            actor = actor_class.options(**options).remote(
                rank=i,
                world_size=num_actors,
                **actor_kwargs_i,
            )
            self._actor_handles.append(actor)

        logger.info(f"Created {num_actors} actors of type {actor_class}")

    def async_init(self, config: dict) -> List[ray.ObjectRef]:
        """
        Asynchronously initialize all actors.

        Args:
            config: Configuration dictionary passed to each actor's init()

        Returns:
            List of ObjectRefs for init completion
        """
        return [actor.init.remote(config) for actor in self._actor_handles]

    def init(self, config: dict) -> List[Any]:
        """
        Synchronously initialize all actors.

        Args:
            config: Configuration dictionary

        Returns:
            List of init results
        """
        refs = self.async_init(config)
        return ray.get(refs)

    def get_actors(self) -> List[ray.actor.ActorHandle]:
        """Get all actor handles."""
        return self._actor_handles

    def get_actor(self, index: int) -> ray.actor.ActorHandle:
        """Get actor at specific index."""
        return self._actor_handles[index]

    def health_check(self) -> List[bool]:
        """Check health of all actors."""
        refs = [actor.health_check.remote() for actor in self._actor_handles]
        return ray.get(refs)

    def dispose(self) -> None:
        """Kill all actors and clean up."""
        for actor in self._actor_handles:
            try:
                ray.kill(actor)
            except Exception as e:
                logger.warning(f"Error killing actor: {e}")
        self._actor_handles.clear()
        logger.info("Actor group disposed")


class InferenceActorGroup(BaseActorGroup):
    """Actor group for inference/generation."""

    def __init__(
        self,
        num_actors: int,
        pg: PlacementGroup,
        bundle_indices: List[int],
        gpu_ids: Optional[List[int]] = None,
        num_gpus_per_actor: float = 1.0,
        num_gpus_per_engine: int = 1,
        capture_child_tasks: bool = False,
        runtime_env: Optional[dict] = None,
        **kwargs,
    ):
        """
        Initialize inference actor group.

        Args:
            num_actors: Number of inference actors
            pg: Placement group
            bundle_indices: Bundle indices for scheduling
            gpu_ids: Physical GPU IDs (for Slime multi-GPU pattern)
            num_gpus_per_actor: GPUs per actor (Ray resource claim, default 1.0)
            num_gpus_per_engine: Actual GPUs each engine needs (>1 for Slime pattern)
            capture_child_tasks: Whether child processes inherit PG scheduling
            runtime_env: Runtime environment variables
            **kwargs: Additional actor kwargs including num_gpus_allocated
        """
        from .actors import InferenceActor

        # Pass num_gpus_allocated to actor constructor
        if "num_gpus_allocated" not in kwargs:
            if num_gpus_per_engine > 1:
                # Slime pattern: actual GPU count is num_gpus_per_engine
                kwargs["num_gpus_allocated"] = num_gpus_per_engine
            elif num_gpus_per_actor < 1:
                # Colocate mode: fractional GPU claim, but 1 physical GPU
                kwargs["num_gpus_allocated"] = 1
            else:
                kwargs["num_gpus_allocated"] = int(num_gpus_per_actor)

        super().__init__(
            actor_class=InferenceActor,
            num_actors=num_actors,
            pg=pg,
            bundle_indices=bundle_indices,
            gpu_ids=gpu_ids,
            num_gpus_per_actor=num_gpus_per_actor,
            num_gpus_per_engine=num_gpus_per_engine,
            capture_child_tasks=capture_child_tasks,
            runtime_env=runtime_env,
            **kwargs,
        )

    def async_generate(
        self,
        prompts: Optional[List[str]] = None,
        **kwargs,
    ) -> List[ray.ObjectRef]:
        """
        Asynchronously generate samples across all actors.

        Args:
            prompts: Optional list of text prompts to generate from
            **kwargs: Additional generation arguments

        Returns:
            List of ObjectRefs for generation results
        """
        # Determine batch size from prompts or batched tensors
        batch_size = None
        if prompts is not None:
            batch_size = len(prompts)
        else:
            for key in (
                "prompt_embeds",
                "pooled_prompt_embeds",
                "encoder_attention_mask",
                "text_ids",
                "latents",
                "image_ids",
            ):
                val = kwargs.get(key)
                if val is None:
                    continue
                if hasattr(val, "shape") and len(getattr(val, "shape", ())) > 0:
                    batch_size = val.shape[0]
                    break
                if isinstance(val, list):
                    batch_size = len(val)
                    break

        if batch_size is None:
            raise ValueError("InferenceActorGroup.generate requires prompts or batched tensors")

        # Distribute batch across actors
        prompts_per_actor = batch_size // self.num_actors
        remainder = batch_size % self.num_actors

        refs = []
        start = 0
        batched_keys = (
            "prompt_embeds",
            "pooled_prompt_embeds",
            "encoder_attention_mask",
            "text_ids",
            "latents",
            "image_ids",
        )
        for i, actor in enumerate(self._actor_handles):
            # Give extra prompts to first few actors
            count = prompts_per_actor + (1 if i < remainder else 0)
            end = start + count

            if count > 0:
                actor_prompts = prompts[start:end] if prompts is not None else None
                actor_kwargs = dict(kwargs)
                for key in batched_keys:
                    if key not in actor_kwargs:
                        continue
                    val = actor_kwargs[key]
                    if val is None:
                        continue
                    if hasattr(val, "shape") and len(getattr(val, "shape", ())) > 0 and val.shape[0] == batch_size:
                        actor_kwargs[key] = val[start:end]
                    elif isinstance(val, list) and len(val) == batch_size:
                        actor_kwargs[key] = val[start:end]
                refs.append(actor.generate.remote(prompts=actor_prompts, **actor_kwargs))

            start = end

        return refs

    def generate(
        self,
        prompts: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Any]:
        """
        Synchronously generate samples.

        Args:
            prompts: Optional list of text prompts
            **kwargs: Additional generation arguments

        Returns:
            List of generation outputs
        """
        refs = self.async_generate(prompts, **kwargs)
        return ray.get(refs)

    def async_sample_batch(
        self,
        prompts: Optional[List[str]] = None,
        **kwargs,
    ) -> List[ray.ObjectRef]:
        """Asynchronously sample a batch (control-plane boundary)."""
        return self.async_generate(prompts, **kwargs)

    def sample_batch(
        self,
        prompts: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Any]:
        """Synchronously sample a batch (control-plane boundary)."""
        return ray.get(self.async_sample_batch(prompts, **kwargs))

    def update_weights(
        self,
        state_dict_ref: ray.ObjectRef,
        *,
        weight_version: Optional[int] = None,
    ) -> List[int]:
        """
        Update weights on all actors.

        Args:
            state_dict_ref: ObjectRef containing state dict

        Returns:
            List of update results
        """
        refs = [
            actor.update_weights.remote(state_dict_ref, weight_version=weight_version)
            for actor in self._actor_handles
        ]
        return ray.get(refs)

    def update_weights_from_path(
        self,
        checkpoint_path: str,
        *,
        weight_version: Optional[int] = None,
    ) -> List[int]:
        """Update weights on all actors from a shared checkpoint path."""
        refs = [
            actor.update_weights_from_path.remote(
                checkpoint_path,
                weight_version=weight_version,
            )
            for actor in self._actor_handles
        ]
        return ray.get(refs)

    def get_weight_versions(self) -> List[int]:
        """Get per-actor inference weight versions."""
        refs = [actor.get_weight_version.remote() for actor in self._actor_handles]
        versions = ray.get(refs)
        return [int(v) for v in versions]

    def async_offload(self) -> List[ray.ObjectRef]:
        """Asynchronously offload all actors to CPU."""
        return [actor.offload.remote() for actor in self._actor_handles]

    def async_onload(self) -> List[ray.ObjectRef]:
        """Asynchronously load all actors back to GPU."""
        return [actor.onload.remote() for actor in self._actor_handles]

    def offload(self) -> None:
        """Synchronously offload all actors."""
        ray.get(self.async_offload())

    def onload(self) -> None:
        """Synchronously load all actors."""
        ray.get(self.async_onload())

    def onload_weights(self) -> None:
        refs = [actor.onload_weights.remote() for actor in self._actor_handles]
        ray.get(refs)

    def onload_post_update(self) -> None:
        refs = [actor.onload_post_update.remote() for actor in self._actor_handles]
        ray.get(refs)

    def onload_runtime_cache(self) -> None:
        refs = [actor.onload_runtime_cache.remote() for actor in self._actor_handles]
        ray.get(refs)


class TrainingActorGroup(BaseActorGroup):
    """Actor group for distributed training."""

    def __init__(
        self,
        num_actors: int,
        pg: PlacementGroup,
        bundle_indices: List[int],
        master_addr: Optional[str] = None,
        master_port: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize training actor group.

        Args:
            num_actors: Number of training actors
            pg: Placement group
            bundle_indices: Bundle indices for scheduling
            master_addr: Master node address for distributed training
            master_port: Master node port
            **kwargs: Additional actor kwargs
        """
        from .actors import TrainingActor
        from .actors.base import RayActor

        # Get master addr/port if not provided
        if master_addr is None or master_port is None:
            master_addr, master_port = RayActor._get_current_node_ip_and_free_port()

        super().__init__(
            actor_class=TrainingActor,
            num_actors=num_actors,
            pg=pg,
            bundle_indices=bundle_indices,
            master_addr=master_addr,
            master_port=master_port,
            **kwargs,
        )

        self.master_addr = master_addr
        self.master_port = master_port

    def async_train(
        self,
        rollout_id: int,
        batch_ref: Any,
    ) -> List[ray.ObjectRef]:
        """
        Asynchronously train on a batch across all actors.

        Args:
            rollout_id: Current rollout ID
            batch_ref: ObjectRef containing training batch

        Returns:
            List of ObjectRefs for training metrics
        """
        if isinstance(batch_ref, list):
            if len(batch_ref) != len(self._actor_handles):
                raise ValueError(
                    f"batch_ref list length {len(batch_ref)} does not match num_actors {len(self._actor_handles)}"
                )
            return [
                actor.train.remote(rollout_id, ref)
                for actor, ref in zip(self._actor_handles, batch_ref, strict=False)
            ]
        return [
            actor.train.remote(rollout_id, batch_ref)
            for actor in self._actor_handles
        ]

    def train(
        self,
        rollout_id: int,
        batch_ref: Any,
    ) -> List[Dict[str, Any]]:
        """
        Synchronously train on a batch.

        Args:
            rollout_id: Current rollout ID
            batch_ref: ObjectRef containing training batch

        Returns:
            List of training metrics from each actor
        """
        refs = self.async_train(rollout_id, batch_ref)
        return ray.get(refs)

    def update_weights(self) -> None:
        """
        Broadcast weights from rank 0 to all other ranks.

        For FSDP, this is typically handled internally.
        """
        refs = [actor.update_weights.remote() for actor in self._actor_handles]
        ray.get(refs)

    def get_weights(self) -> ray.ObjectRef:
        """
        Get weights from rank 0 for syncing to inference actors.

        IMPORTANT: For FSDP, we must call get_weights on ALL actors simultaneously
        because FSDP.state_dict() triggers an ALLGATHER collective that requires
        all ranks to participate. Only rank 0 returns the full state dict
        (due to rank0_only=True), other ranks return empty dicts.

        Returns:
            ObjectRef containing state dict from rank 0
        """
        # Call get_weights on all actors - required for FSDP collectives
        refs = [actor.get_weights.remote() for actor in self._actor_handles]
        # Wait for all to complete (FSDP requires synchronization)
        # but only return rank 0's result
        ray.get(refs[1:])  # Wait for other ranks to complete their part
        return refs[0]  # Return rank 0's ObjectRef

    def export_weights_to_path(self, checkpoint_path: str) -> str:
        """
        Export synchronized weights to a shared path (FSDP-safe collective).

        All ranks participate; rank 0 writes the file.
        """
        refs = [actor.export_weights_to_path.remote(checkpoint_path) for actor in self._actor_handles]
        ray.get(refs[1:])
        ray.get(refs[0])
        return checkpoint_path

    def save_model(self, path: str) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        refs = [actor.save_model.remote(path) for actor in self._actor_handles]
        ray.get(refs)

    def load_checkpoint(self, path: str) -> None:
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint
        """
        refs = [actor.load_checkpoint.remote(path) for actor in self._actor_handles]
        ray.get(refs)

    def async_offload(self) -> List[ray.ObjectRef]:
        """Asynchronously offload all actors to CPU."""
        return [actor.offload.remote() for actor in self._actor_handles]

    def async_onload(self) -> List[ray.ObjectRef]:
        """Asynchronously load all actors back to GPU."""
        return [actor.onload.remote() for actor in self._actor_handles]

    def offload(self) -> None:
        """Synchronously offload all training actors to CPU."""
        ray.get(self.async_offload())

    def onload(self) -> None:
        """Synchronously load all training actors back to GPU."""
        ray.get(self.async_onload())

    def onload_weights(self) -> None:
        refs = [actor.onload_weights.remote() for actor in self._actor_handles]
        ray.get(refs)

    def onload_post_update(self) -> None:
        refs = [actor.onload_post_update.remote() for actor in self._actor_handles]
        ray.get(refs)

    def onload_runtime_cache(self) -> None:
        refs = [actor.onload_runtime_cache.remote() for actor in self._actor_handles]
        ray.get(refs)

    def clear_memory(self) -> None:
        """Clear GPU cache without full offload."""
        refs = [actor.clear_memory.remote() for actor in self._actor_handles]
        ray.get(refs)

    def _get_batch_size(
        self,
        prompts: Optional[List[str]],
        **kwargs,
    ) -> int:
        if prompts is not None:
            return len(prompts)
        for key in (
            "prompt_embeds",
            "pooled_prompt_embeds",
            "encoder_attention_mask",
            "text_ids",
            "latents",
            "image_ids",
        ):
            val = kwargs.get(key)
            if val is None:
                continue
            if hasattr(val, "shape") and len(getattr(val, "shape", ())) > 0:
                return val.shape[0]
            if isinstance(val, list):
                return len(val)
        raise ValueError("TrainingActorGroup.generate requires prompts or batched tensors")

    def _pad_batched_value(self, val: Any, batch_size: int, target_size: int) -> Any:
        if val is None or target_size <= batch_size:
            return val
        pad_count = target_size - batch_size
        if hasattr(val, "shape") and len(getattr(val, "shape", ())) > 0 and val.shape[0] == batch_size:
            pad = val[-1:].repeat(pad_count, *([1] * (val.dim() - 1)))
            return torch.cat([val, pad], dim=0)
        if isinstance(val, list) and len(val) == batch_size:
            return val + [val[-1]] * pad_count
        return val

    def _slice_sampler_output(self, output: Any, start: int, end: int) -> Any:
        from diffusionrl.types import SamplerOutput

        if not isinstance(output, SamplerOutput):
            return output
        return SamplerOutput(
            latents=output.latents[start:end],
            timesteps=output.timesteps,
            trajectories=output.trajectories[start:end] if output.trajectories is not None else None,
            log_probs=output.log_probs.slice(start, end) if output.log_probs is not None else None,
            embeddings=output.embeddings.slice(start, end) if output.embeddings is not None else None,
            decoded_images=output.decoded_images[start:end] if output.decoded_images is not None else None,
            metadata=output.metadata,
            contract_version=output.contract_version,
            step_indices=output.step_indices,
        )

    def async_generate(
        self,
        prompts: Optional[List[str]] = None,
        **kwargs,
    ) -> List[ray.ObjectRef]:
        batch_size = self._get_batch_size(prompts, **kwargs)
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0 for generate")

        target_size = max(batch_size, self.num_actors)
        if target_size > batch_size:
            if prompts is not None:
                prompts = self._pad_batched_value(prompts, batch_size, target_size)
            for key in (
                "prompt_embeds",
                "pooled_prompt_embeds",
                "encoder_attention_mask",
                "text_ids",
                "latents",
                "image_ids",
            ):
                if key in kwargs:
                    kwargs[key] = self._pad_batched_value(kwargs.get(key), batch_size, target_size)

        # Distribute batch across actors
        prompts_per_actor = target_size // self.num_actors
        remainder = target_size % self.num_actors

        refs = []
        start = 0
        batched_keys = (
            "prompt_embeds",
            "pooled_prompt_embeds",
            "encoder_attention_mask",
            "text_ids",
            "latents",
            "image_ids",
        )
        for i, actor in enumerate(self._actor_handles):
            count = prompts_per_actor + (1 if i < remainder else 0)
            end = start + count

            if count > 0:
                actor_prompts = prompts[start:end] if prompts is not None else None
                actor_kwargs = dict(kwargs)
                for key in batched_keys:
                    if key not in actor_kwargs:
                        continue
                    val = actor_kwargs[key]
                    if val is None:
                        continue
                    if hasattr(val, "shape") and len(getattr(val, "shape", ())) > 0 and val.shape[0] == target_size:
                        actor_kwargs[key] = val[start:end]
                    elif isinstance(val, list) and len(val) == target_size:
                        actor_kwargs[key] = val[start:end]
                refs.append(actor.generate.remote(prompts=actor_prompts, **actor_kwargs))

            start = end

        return refs

    def generate(
        self,
        prompts: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Any]:
        original_batch_size = self._get_batch_size(prompts, **kwargs)
        target_size = max(original_batch_size, self.num_actors)

        refs = self.async_generate(prompts=prompts, **kwargs)
        outputs = ray.get(refs)

        if target_size == original_batch_size:
            return outputs

        # Trim padded samples
        prompts_per_actor = target_size // self.num_actors
        remainder = target_size % self.num_actors
        counts = [
            prompts_per_actor + (1 if i < remainder else 0)
            for i in range(self.num_actors)
        ]

        trimmed = []
        remaining = original_batch_size
        for output, count in zip(outputs, counts):
            if remaining <= 0:
                break
            keep = min(count, remaining)
            if keep < count:
                output = self._slice_sampler_output(output, 0, keep)
            trimmed.append(output)
            remaining -= keep

        return trimmed

    def sample_batch(
        self,
        prompts: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Any]:
        """Control-plane sampling API used by RolloutManager."""
        return self.generate(prompts=prompts, **kwargs)


def create_inference_actor_group(
    args,
    pg_result,
) -> InferenceActorGroup:
    """
    Factory function to create InferenceActorGroup from args.

    GPU Allocation Strategy:
    - FSDP engine: 1 GPU per actor (default)
    - FastVideo engine: sp_size GPUs per actor (for sequence parallelism)

    Args:
        args: GRPOArguments instance
        pg_result: Tuple of (PlacementGroup, bundle_indices, gpu_ids)

    Returns:
        Initialized InferenceActorGroup
    """
    pg, bundle_indices, gpu_ids = pg_result

    # Get sampler_engine_type (should be set by validate_args in arguments.py)
    sampler_engine_type = getattr(args, "sampler_engine_type", None)

    # Fallback to fsdp if not set (should not happen if validate_args was called)
    if sampler_engine_type is None:
        logger.warning(
            "sampler_engine_type not set. This should have been auto-selected in validate_args(). "
            "Falling back to 'fsdp'."
        )
        sampler_engine_type = "fsdp"

    # Determine GPUs per actor based on engine type
    engine_kwargs = getattr(args, "engine_kwargs", {})
    if not isinstance(engine_kwargs, dict):
        logger.warning("engine_kwargs is not a dict in create_inference_actor_group; resetting to empty dict.")
        engine_kwargs = {}

    if sampler_engine_type == "fastvideo":
        # FastVideo GPU allocation:
        # - num_gpus: GPUs per FastVideo instance (per Ray actor)
        # - sp_size: Sequence parallelism size (must <= num_gpus)
        #
        # Scenarios:
        # 1. sp_size=1, num_gpus=1: Single GPU, no parallelism
        # 2. sp_size=4, num_gpus=4: 4 GPUs with SP
        # 3. sp_size=1, num_gpus=4: 4 GPUs, no SP (data parallel within executor)
        #
        # For multi-node: each node gets one actor, actor uses local GPUs only
        # (MultiprocExecutor uses multiprocessing, doesn't support cross-node)

        sp_size = engine_kwargs.get("sp_size", getattr(args, "sp_size", 1))
        tp_size = engine_kwargs.get("tp_size", getattr(args, "tp_size", 1))

        # Determine num_gpus per actor
        # If fastvideo_num_gpus is set, use it; otherwise use sp_size
        fastvideo_num_gpus = getattr(args, "fastvideo_num_gpus", None)
        if fastvideo_num_gpus is not None:
            num_gpus_per_actor = fastvideo_num_gpus
        else:
            # Default: each actor gets sp_size GPUs
            # This ensures SP works correctly
            num_gpus_per_actor = sp_size

        # Validate
        if sp_size > num_gpus_per_actor:
            raise ValueError(
                f"sp_size ({sp_size}) must be <= num_gpus_per_actor ({num_gpus_per_actor})"
            )

        # Update engine_kwargs
        engine_kwargs["sp_size"] = sp_size
        engine_kwargs["num_gpus"] = num_gpus_per_actor
        engine_kwargs["tp_size"] = tp_size

        logger.info(
            f"FastVideo engine: num_gpus_per_actor={num_gpus_per_actor}, "
            f"sp_size={sp_size}, tp_size={tp_size}"
        )
    elif sampler_engine_type == "fsdp":
        # FSDP GPU allocation:
        # - fsdp_num_gpus: GPUs per FSDP inference actor (default: 1)
        # - fsdp_sharding_strategy: Sharding strategy (NO_SHARD for inference)
        #
        # Scenarios:
        # 1. fsdp_num_gpus=1: Single GPU, data parallel across actors (default)
        # 2. fsdp_num_gpus=4: 4 GPUs with FSDP model parallelism
        #
        # For multi-node: each node gets one actor, actor uses local GPUs only
        # (torch.distributed doesn't support cross-node within single actor)

        fsdp_num_gpus = getattr(args, "fsdp_num_gpus", 1)
        fsdp_sharding_strategy = getattr(args, "fsdp_inference_sharding_strategy", "NO_SHARD")

        num_gpus_per_actor = fsdp_num_gpus

        # Update engine_kwargs
        engine_kwargs["num_gpus"] = num_gpus_per_actor
        engine_kwargs["fsdp_sharding_strategy"] = fsdp_sharding_strategy
        engine_kwargs.setdefault("cpu_offload", getattr(args, "fsdp_cpu_offload", False))

        if num_gpus_per_actor > 1:
            logger.info(
                f"FSDP engine: num_gpus_per_actor={num_gpus_per_actor}, "
                f"sharding_strategy={fsdp_sharding_strategy}"
            )
        else:
            logger.info("FSDP engine: single GPU per actor (default)")
    else:
        # SGLang/other: 1 GPU per actor
        num_gpus_per_actor = 1

    # Calculate number of actors
    # For FastVideo with sp_size>1, each actor uses sp_size GPUs
    total_gpus = args.inference_num_nodes * args.inference_num_gpus_per_node

    # In colocate mode with single-GPU setup, use fractional GPU allocation
    # This allows both inference and training actors to share the same GPU bundle
    colocate = getattr(args, "colocate_inference_training", False)
    if colocate and num_gpus_per_actor == 1:
        num_gpus_per_actor = float(getattr(args, "colocate_inference_gpu_fraction", 0.4))
        logger.info(
            f"Colocate mode: InferenceActors using {num_gpus_per_actor} GPU each"
        )

    # Multi-GPU engine (non-colocate): use Slime pattern
    if num_gpus_per_actor > 1 and not colocate:
        if not bool(getattr(args, "allow_noset_multi_gpu_inference", False)):
            raise ValueError(
                "Multi-GPU inference actor layout requires --allow-noset-multi-gpu-inference=true. "
                "Default layout only supports integer single-GPU actors."
            )
        from diffusionrl.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST

        actual_gpus_per_engine = int(num_gpus_per_actor)
        ray_num_gpus = 0.5  # Fractional claim to satisfy Ray scheduler
        num_actors = total_gpus // actual_gpus_per_engine

        if num_actors < 1:
            raise ValueError(
                f"Not enough GPUs for inference. "
                f"Total GPUs: {total_gpus}, GPUs per engine: {actual_gpus_per_engine}"
            )

        noset_env = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST}

        logger.info(
            f"Creating {num_actors} inference actors (Slime pattern), "
            f"{actual_gpus_per_engine} GPU(s) per engine, "
            f"ray_num_gpus={ray_num_gpus}"
        )

        group = InferenceActorGroup(
            num_actors=num_actors,
            pg=pg,
            bundle_indices=bundle_indices,
            gpu_ids=gpu_ids,
            num_gpus_per_actor=ray_num_gpus,
            num_gpus_per_engine=actual_gpus_per_engine,
            capture_child_tasks=True,
            runtime_env={"env_vars": noset_env},
            num_gpus_allocated=actual_gpus_per_engine,
            force_set_cuda_visible_devices=True,
        )
    else:
        # Single GPU or colocate mode: standard scheduling
        if colocate and num_gpus_per_actor < 1:
            num_actors = total_gpus
        else:
            num_actors = int(total_gpus / num_gpus_per_actor)

        if num_actors < 1:
            raise ValueError(
                f"Not enough GPUs for inference. "
                f"Total GPUs: {total_gpus}, GPUs per actor: {num_gpus_per_actor}"
            )

        logger.info(
            f"Creating {num_actors} inference actors, "
            f"{num_gpus_per_actor} GPU(s) per actor"
        )

        group = InferenceActorGroup(
            num_actors=num_actors,
            pg=pg,
            bundle_indices=bundle_indices,
            num_gpus_per_actor=num_gpus_per_actor,
        )

    # Initialize actors with engine configuration
    # Ensure engine_kwargs includes LoRA settings for inference-side adapters
    engine_kwargs = dict(engine_kwargs)
    engine_kwargs.setdefault("use_lora", getattr(args, "use_lora", False))
    engine_kwargs.setdefault("lora_rank", getattr(args, "lora_rank", 16))
    engine_kwargs.setdefault("lora_alpha", getattr(args, "lora_alpha", 16))
    engine_kwargs.setdefault("lora_target_modules", getattr(args, "lora_target_modules", None))

    engine_config = {
        # Engine selection (sampler_engine_type is the single source of truth)
        "sampler_engine_type": sampler_engine_type,
        # Model configuration
        "sampler_path": args.sampler_path,
        "model_path": args.model_path,
        "pretrained_model_path": args.pretrained_model_path,
        # LoRA configuration (must match training for on-policy)
        "lora_rank": getattr(args, "lora_rank", 16),
        "lora_alpha": getattr(args, "lora_alpha", 16),
        "lora_target_modules": getattr(args, "lora_target_modules", None),
        # Sampling configuration
        "num_inference_steps": args.num_inference_steps,
        "eta": args.eta,
        "sde_type": args.sde_type,
        "shift": args.shift,
        "guidance_scale": args.guidance_scale,
        # Output dimensions
        "height": args.height,
        "width": args.width,
        "num_frames": getattr(args, "num_frames", 16),
        # Engine-specific kwargs (includes sp_size, num_gpus for FastVideo)
        "engine_kwargs": engine_kwargs,
    }
    group.init(engine_config)

    return group


def create_training_actor_group(
    args,
    pg_result,
) -> TrainingActorGroup:
    """
    Factory function to create TrainingActorGroup from args.

    Args:
        args: GRPOArguments instance
        pg_result: Tuple of (PlacementGroup, bundle_indices, gpu_ids)

    Returns:
        Initialized TrainingActorGroup
    """
    pg, bundle_indices, gpu_ids = pg_result

    num_actors = args.training_num_nodes * args.training_num_gpus_per_node

    # In colocate mode, use fractional GPU allocation to allow sharing
    # Both inference and training actors will claim 0.5 GPU each
    colocate = getattr(args, "colocate_inference_training", False)
    num_gpus_per_actor = float(getattr(args, "colocate_training_gpu_fraction", 0.4)) if colocate else 1.0

    if colocate:
        logger.info(
            f"Colocate mode: TrainingActors using {num_gpus_per_actor} GPU each"
        )

    group = TrainingActorGroup(
        num_actors=num_actors,
        pg=pg,
        bundle_indices=bundle_indices,
        num_gpus_per_actor=num_gpus_per_actor,
    )

    # Initialize actors
    config = {
        "model_config": {
            "model_path": args.model_path,
            "pretrained_model_path": args.pretrained_model_path,
            # LoRA configuration (must match inference for on-policy)
            "use_lora": getattr(args, "use_lora", False),
            "lora_rank": getattr(args, "lora_rank", 16),
            "lora_alpha": getattr(args, "lora_alpha", 16),
            "lora_target_modules": getattr(args, "lora_target_modules", None),
            # Gradient checkpointing (default disabled; enable explicitly)
            "use_gradient_checkpointing": getattr(args, "use_gradient_checkpointing", False),
        },
        "optimizer_config": {
            "learning_rate": args.learning_rate,
            "adam_beta1": args.adam_beta1,
            "adam_beta2": args.adam_beta2,
            "adam_epsilon": args.adam_epsilon,
            "weight_decay": args.weight_decay,
        },
        "scheduler_config": {
            "type": args.lr_scheduler_type,
            "warmup_steps": args.warmup_steps,
            "total_steps": args.num_rollout,
        },
        "loss_config": {
            # Loss type selection
            "loss_type": getattr(args, "loss_type", "grpo"),
            # GRPO loss parameters
            "clip_range": args.clip_range,
            "clip_range_mode": args.clip_range_mode,
            "use_kl_penalty": args.use_kl_penalty,
            "kl_coef": args.kl_coef,
            "eta": args.eta,
            "sde_type": args.sde_type,
            "guidance_scale": args.guidance_scale,
            # Timestep filtering parameters (MixGRPO)
            "ignore_last": getattr(args, "ignore_last", False),
            "frozen_init_timesteps": getattr(args, "frozen_init_timesteps", 0),
            # NFT loss parameters
            "beta": getattr(args, "nft_beta", 0.1),
            "adv_clip_max": getattr(args, "nft_adv_clip_max", 5.0),
            "adv_mode": getattr(args, "nft_adv_mode", "raw"),
            "use_adaptive_weight": getattr(args, "nft_use_adaptive_weight", True),
            "shift": getattr(args, "shift", 3.0),
            "nft_timestep_mode": getattr(args, "nft_timestep_mode", "random"),
            "nft_shuffle_timesteps": getattr(args, "nft_shuffle_timesteps", True),
            "nft_apply_shift": getattr(args, "nft_apply_shift", False),
            # EMA parameters (for NFT dual adapter)
            "use_ema": getattr(args, "use_ema", False),
            "ema_decay": getattr(args, "ema_decay", 0.001),
            "decay_type": getattr(args, "ema_decay_type", "constant"),
            "ema_flat_steps": getattr(args, "ema_flat_steps", 0),
            "ema_uprate": getattr(args, "ema_uprate", 0.001),
            "ema_uphold": getattr(args, "ema_uphold", 0.5),
        },
        # Note: algorithm_config removed - algorithm instantiation happens in
        # RolloutManager for advantage computation, not in TrainingActor.
        # Training uses loss_fn directly (functional dispatch pattern).
        "training_config": {
            "max_grad_norm": args.max_grad_norm,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "prompts_per_batch": getattr(args, "prompts_per_batch", 1),
            "num_samples_per_prompt": getattr(args, "num_samples_per_prompt", 1),
            "batch_size": args.batch_size,
            "gradient_steps_per_epoch": getattr(args, "gradient_steps_per_epoch", 1),
            "num_inner_epochs": getattr(args, "num_inner_epochs", 1),
            "world_size": num_actors,
            "fastvideo_replay_log_probs": getattr(args, "fastvideo_replay_log_probs", False),
        },
        "sampling_config": {
            "sampler_path": args.sampler_path,
            "num_inference_steps": args.num_inference_steps,
            "eta": args.eta,
            "sde_type": args.sde_type,
            "shift": args.shift,
            "guidance_scale": args.guidance_scale,
            "height": args.height,
            "width": args.width,
            "num_frames": getattr(args, "num_frames", 16),
            "sampling_adapter": getattr(args, "sampling_adapter", None),
            "init_same_noise": getattr(args, "init_same_noise", False),
            "num_samples_per_prompt": getattr(args, "num_samples_per_prompt", 1),
            "sampler_kwargs": getattr(args, "engine_kwargs", {}).get("sampler_kwargs", {}),
        },
        "use_fsdp": args.use_fsdp,
        "fsdp_config": {
            "sharding_strategy": args.fsdp_sharding_strategy,
            "cpu_offload": args.fsdp_cpu_offload,
            "backward_prefetch": args.fsdp_backward_prefetch,
        },
    }
    group.init(config)

    return group
