"""Rollout worker-group implementation."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import ray
from ray.util.placement_group import PlacementGroup

from diffusionrl.types import RolloutRequest
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
        self._num_gpus_allocated = int(kwargs.get("num_gpus_allocated", 1) or 1)
        self._weight_update_target_by_actor: Dict[int, str] = {
            idx: f"actor_rank:{idx}" for idx in range(len(self._actor_handles))
        }
        self._weight_update_actor_indices: List[int] = list(range(len(self._actor_handles)))
        self._weight_update_targets_ready: bool = False

    def refresh_weight_update_targets(self) -> Dict[str, Any]:
        """Refresh actor->logical update-target mapping for deduplicated sync calls."""
        if not self._actor_handles:
            self._weight_update_target_by_actor = {}
            self._weight_update_actor_indices = []
            self._weight_update_targets_ready = True
            return {
                "num_actors": 0,
                "num_unique_targets": 0,
                "selected_actor_indices": [],
            }

        refs = [actor.get_weight_update_target.remote() for actor in self._actor_handles]
        try:
            payloads = ray.get(refs)
        except Exception as exc:
            logger.warning(
                "Failed to collect rollout weight-update targets; fallback to per-actor updates. %s",
                exc,
            )
            self._weight_update_target_by_actor = {
                idx: f"actor_rank:{idx}" for idx in range(len(self._actor_handles))
            }
            self._weight_update_actor_indices = list(range(len(self._actor_handles)))
            self._weight_update_targets_ready = True
            return {
                "num_actors": int(len(self._actor_handles)),
                "num_unique_targets": int(len(self._actor_handles)),
                "selected_actor_indices": list(self._weight_update_actor_indices),
            }

        target_by_actor: Dict[int, str] = {}
        first_actor_by_target: Dict[str, int] = {}
        for idx, payload in enumerate(payloads):
            target = None
            if isinstance(payload, dict):
                raw_target = payload.get("target")
                if isinstance(raw_target, str) and raw_target.strip():
                    target = raw_target.strip()
            if not target:
                target = f"actor_rank:{idx}"
            target_by_actor[idx] = target
            if target not in first_actor_by_target:
                first_actor_by_target[target] = idx

        selected = sorted(int(v) for v in first_actor_by_target.values())
        self._weight_update_target_by_actor = target_by_actor
        self._weight_update_actor_indices = selected
        self._weight_update_targets_ready = True

        if len(selected) < len(self._actor_handles):
            logger.info(
                "RolloutActorGroup(%s) deduplicates weight updates: %d actors -> %d logical targets",
                self._sampler_engine_type,
                len(self._actor_handles),
                len(selected),
            )

        return {
            "num_actors": int(len(self._actor_handles)),
            "num_unique_targets": int(len(selected)),
            "selected_actor_indices": list(selected),
        }

    def _ensure_weight_update_targets(self) -> None:
        if not self._weight_update_targets_ready:
            self.refresh_weight_update_targets()

    def _get_weight_update_handles(self) -> List[Any]:
        self._ensure_weight_update_targets()
        return [self._actor_handles[idx] for idx in self._weight_update_actor_indices]

    def async_generate(self, request: RolloutRequest) -> List[ray.ObjectRef]:
        """
        Asynchronously generate samples across all actors.

        Args:
            request: RolloutRequest with prompts and generation parameters.

        Returns:
            List of ObjectRefs for generation results
        """
        if not isinstance(request.prompts, list) or len(request.prompts) == 0:
            raise ValueError(
                "RolloutActorGroup.generate requires non-empty text prompts. "
                "Prompt-embedding-only input is no longer supported."
            )
        batch_size = len(request.prompts)

        # Distribute batch across actors
        prompts_per_actor = batch_size // self.num_actors
        remainder = batch_size % self.num_actors

        refs = []
        start = 0
        for i, actor in enumerate(self._actor_handles):
            # Give extra prompts to first few actors
            count = prompts_per_actor + (1 if i < remainder else 0)
            end = start + count

            if count > 0:
                actor_request = request.slice_prompts(start, end)
                refs.append(actor.generate.remote(actor_request))

            start = end

        return refs

    def generate(self, request: RolloutRequest) -> List[Any]:
        """
        Synchronously generate samples.

        Args:
            request: RolloutRequest with prompts and generation parameters.

        Returns:
            List of generation outputs
        """
        refs = self.async_generate(request)
        return ray.get(refs)

    def async_encode_prompt(
        self,
        prompts: List[str],
        **kwargs,
    ) -> ray.ObjectRef:
        """Asynchronously encode prompts using one rollout actor."""
        if not self._actor_handles:
            raise RuntimeError("Rollout actor group is empty.")
        return self._actor_handles[0].encode_prompt.remote(prompts=list(prompts), **kwargs)

    def encode_prompt(
        self,
        prompts: List[str],
        **kwargs,
    ) -> Dict[str, Any]:
        """Synchronously encode prompts using one rollout actor."""
        ref = self.async_encode_prompt(prompts=prompts, **kwargs)
        result = ray.get(ref)
        if isinstance(result, dict):
            return result
        raise TypeError(
            f"RolloutActorGroup.encode_prompt expected dict payload, got {type(result).__name__}."
        )

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
        refs = [
            actor.update_weights.remote(state_dict_ref)
            for actor in self._get_weight_update_handles()
        ]
        ray.get(refs)

    def update_weights_from_path(
        self,
        checkpoint_path: str,
    ) -> None:
        """Update weights on all actors from a shared checkpoint path."""
        refs = [
            actor.update_weights_from_path.remote(checkpoint_path)
            for actor in self._get_weight_update_handles()
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

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        target_modules: Optional[List[str]] = None,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ) -> None:
        """Update weights on all actors from serialized tensor payload."""
        refs = [
            actor.update_weights_from_tensor.remote(
                serialized_named_tensors=list(serialized_named_tensors),
                target_modules=target_modules,
                load_format=load_format,
                flush_cache=flush_cache,
            )
            for actor in self._get_weight_update_handles()
        ]
        ray.get(refs)

    def init_weights_update_group(
        self,
        *,
        master_address: str,
        master_port: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
    ) -> None:
        """Initialize distributed weight-update group on all rollout actors."""
        refs = []
        selected_handles = self._get_weight_update_handles()
        for idx, actor in enumerate(selected_handles):
            rank_offset = 1 + idx * self._num_gpus_allocated
            refs.append(
                actor.init_weights_update_group.remote(
                    master_address=master_address,
                    master_port=int(master_port),
                    rank_offset=int(rank_offset),
                    world_size=int(world_size),
                    group_name=str(group_name),
                    backend=str(backend),
                )
            )
        ray.get(refs)

    def destroy_weights_update_group(
        self,
        *,
        group_name: str,
    ) -> None:
        """Destroy distributed weight-update group on all rollout actors."""
        refs = [
            actor.destroy_weights_update_group.remote(group_name=str(group_name))
            for actor in self._get_weight_update_handles()
        ]
        ray.get(refs)

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
        """Update weights on all actors from distributed broadcast."""
        refs = [
            actor.update_weights_from_distributed.remote(
                names=list(names),
                dtypes=list(dtypes),
                shapes=[list(shape) for shape in shapes],
                group_name=str(group_name),
                target_modules=target_modules,
                flush_cache=flush_cache,
            )
            for actor in self._get_weight_update_handles()
        ]
        ray.get(refs)

    def get_weight_sync_topology(self) -> dict:
        """Return rollout-side topology for weight-sync group construction."""
        self._ensure_weight_update_targets()
        num_unique_targets = int(len(self._weight_update_actor_indices))
        return {
            "num_actors": int(len(self._actor_handles)),
            "num_weight_update_targets": num_unique_targets,
            "num_gpus_per_actor": int(self._num_gpus_allocated),
            "total_gpus": int(num_unique_targets * self._num_gpus_allocated),
        }

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
