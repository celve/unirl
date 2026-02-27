"""Rollout worker-group implementation."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import ray
from ray.util.placement_group import PlacementGroup

from .base import BaseActorGroup

logger = logging.getLogger(__name__)

class RolloutActorGroup(BaseActorGroup):
    """Actor group for rollout/generation."""

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
        sampler_engine_type: str = "fsdp",
        **kwargs,
    ):
        """
        Initialize rollout actor group.

        Args:
            num_actors: Number of rollout actors
            pg: Placement group
            bundle_indices: Bundle indices for scheduling
            gpu_ids: Physical GPU IDs (for Slime multi-GPU pattern)
            num_gpus_per_actor: GPUs per actor (Ray resource claim, default 1.0)
            num_gpus_per_engine: Actual GPUs each engine needs (>1 for Slime pattern)
            capture_child_tasks: Whether child processes inherit PG scheduling
            runtime_env: Runtime environment variables
            **kwargs: Additional actor kwargs including num_gpus_allocated
        """
        from ..actors import RolloutActor

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
            actor_class=RolloutActor,
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
        self._sampler_engine_type = str(sampler_engine_type or "fsdp").lower()
        self._warned_ignored_embedding_kwargs = False

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
        if not isinstance(prompts, list) or len(prompts) == 0:
            raise ValueError(
                "RolloutActorGroup.generate requires non-empty text prompts. "
                "Prompt-embedding-only input is no longer supported."
            )
        batch_size = len(prompts)

        # Distribute batch across actors
        prompts_per_actor = batch_size // self.num_actors
        remainder = batch_size % self.num_actors

        refs = []
        start = 0
        embedding_keys = (
            "prompt_embeds",
            "pooled_prompt_embeds",
            "negative_prompt_embeds",
            "negative_pooled_prompt_embeds",
            "encoder_attention_mask",
            "text_ids",
            "image_ids",
        )
        if (
            not self._warned_ignored_embedding_kwargs
            and any(key in kwargs for key in embedding_keys)
        ):
            logger.warning(
                "RolloutActorGroup(%s) now uses prompt-only generation input; "
                "embedding kwargs are ignored.",
                self._sampler_engine_type,
            )
            self._warned_ignored_embedding_kwargs = True

        batched_keys = ("latents",)
        for i, actor in enumerate(self._actor_handles):
            # Give extra prompts to first few actors
            count = prompts_per_actor + (1 if i < remainder else 0)
            end = start + count

            if count > 0:
                actor_prompts = prompts[start:end] if prompts is not None else None
                actor_kwargs = dict(kwargs)
                for key in embedding_keys:
                    actor_kwargs.pop(key, None)
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

    def update_weights(
        self,
        state_dict_ref: ray.ObjectRef,
    ) -> None:
        """
        Update weights on all actors.

        Args:
            state_dict_ref: ObjectRef containing state dict

        Returns:
            List of update results
        """
        refs = [actor.update_weights.remote(state_dict_ref) for actor in self._actor_handles]
        ray.get(refs)

    def update_weights_from_path(
        self,
        checkpoint_path: str,
    ) -> None:
        """Update weights on all actors from a shared checkpoint path."""
        refs = [
            actor.update_weights_from_path.remote(checkpoint_path)
            for actor in self._actor_handles
        ]
        results = ray.get(refs)

        checksum_rows: List[tuple[int, tuple[tuple[str, str], ...]]] = []
        for idx, payload in enumerate(results):
            if not isinstance(payload, dict):
                continue
            raw_checksum = payload.get("checksum")
            if not isinstance(raw_checksum, dict) or not raw_checksum:
                continue
            normalized = tuple(
                sorted((str(k), str(v)) for k, v in raw_checksum.items())
            )
            rank = int(payload.get("rank", idx))
            checksum_rows.append((rank, normalized))

        if len(checksum_rows) <= 1:
            return

        checksum_groups: dict[tuple[tuple[str, str], ...], List[int]] = {}
        for rank, normalized in checksum_rows:
            checksum_groups.setdefault(normalized, []).append(rank)

        if len(checksum_groups) > 1:
            details = {
                str(dict(items)): ranks
                for items, ranks in checksum_groups.items()
            }
            raise RuntimeError(
                "Checksum mismatch across rollout actors after update_weights_from_path: "
                f"{details}"
            )
        checksum_payload = dict(next(iter(checksum_groups.keys())))
        logger.info(
            "Verified consistent rollout checksum across %d actors: %s",
            len(checksum_rows),
            checksum_payload,
        )

    def async_sleep(self) -> List[ray.ObjectRef]:
        """Asynchronously put all actors into sleep mode."""
        return [actor.sleep.remote() for actor in self._actor_handles]

    def async_wake_up(self) -> List[ray.ObjectRef]:
        """Asynchronously wake all actors up."""
        return [actor.wake_up.remote() for actor in self._actor_handles]

    def sleep(self) -> None:
        """Synchronously put all actors into sleep mode."""
        ray.get(self.async_sleep())

    def wake_up(self) -> None:
        """Synchronously wake all actors up."""
        ray.get(self.async_wake_up())

__all__ = ["RolloutActorGroup"]
