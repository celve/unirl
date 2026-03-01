"""Base worker-group lifecycle primitives."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Type

import ray
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



__all__ = ["BaseActorGroup"]
