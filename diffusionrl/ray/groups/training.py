"""Training worker-group implementation."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import ray
import torch
from ray.util.placement_group import PlacementGroup

from .base import BaseActorGroup

logger = logging.getLogger(__name__)

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
        from ..actors import TrainingActor
        from ..actors.base import RayActor

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
        self._buffer_consumer_spec_cache: Optional[Dict[str, Any]] = None
        self._train_backend_info_cache: Optional[Dict[str, Any]] = None

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

    def get_train_backend_info(self, *, force_refresh: bool = False) -> Dict[str, Any]:
        """Return training backend metadata declared by actor rank 0."""
        if self._train_backend_info_cache is not None and not force_refresh:
            return dict(self._train_backend_info_cache)
        if not self._actor_handles:
            return {}
        info = ray.get(self._actor_handles[0].get_train_backend_info.remote())
        if isinstance(info, dict):
            self._train_backend_info_cache = dict(info)
            return dict(info)
        return {}

    def get_buffer_consumer_spec(self, *, force_refresh: bool = False) -> Dict[str, Any]:
        """Describe how rollout buffer should partition payloads for this group."""
        if self._buffer_consumer_spec_cache is not None and not force_refresh:
            return dict(self._buffer_consumer_spec_cache)
        if not self._actor_handles:
            return {
                "dp_size": self.num_actors,
                "partition_train_data": True,
                "partition_mode": "data_parallel",
            }
        spec = ray.get(self._actor_handles[0].get_buffer_consumer_spec.remote())
        if not isinstance(spec, dict):
            spec = {}
        spec = dict(spec)
        spec.setdefault("dp_size", int(self.num_actors))
        spec.setdefault("partition_train_data", True)
        spec.setdefault("partition_mode", "data_parallel")
        self._buffer_consumer_spec_cache = spec
        return dict(spec)

    def get_weights(self) -> ray.ObjectRef:
        """
        Get weights from rank 0 for syncing to rollout actors.

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

    def export_weights_to_path(
        self,
        checkpoint_path: str,
        export_format: str = "state_dict",
    ) -> str:
        """
        Export synchronized weights to a shared path (FSDP-safe collective).

        All ranks participate; rank 0 writes the file.
        """
        refs = [
            actor.export_weights_to_path.remote(
                checkpoint_path,
                export_format=export_format,
            )
            for actor in self._actor_handles
        ]
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
        from diffusionrl.types.sampling import RolloutOutput

        if not isinstance(output, RolloutOutput):
            return output
        return RolloutOutput(
            latents=output.latents[start:end],
            timesteps=output.timesteps,
            trajectories=output.trajectories[start:end] if output.trajectories is not None else None,
            log_probs=output.log_probs.slice(start, end) if output.log_probs is not None else None,
            embeddings=output.embeddings.slice(start, end) if output.embeddings is not None else None,
            decoded_images=output.decoded_images[start:end] if output.decoded_images is not None else None,
            metadata=output.metadata,
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

__all__ = ["TrainingActorGroup"]
