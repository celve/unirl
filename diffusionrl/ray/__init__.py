"""Distributed control-plane facade with lazy exports."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    # Placement groups
    "RuntimePlacementConfig": ("diffusionrl.ray.placement_group", "RuntimePlacementConfig"),
    "PlacementGroupResult": ("diffusionrl.ray.placement_group", "PlacementGroupResult"),
    "create_placement_groups": ("diffusionrl.ray.placement_group", "create_placement_groups"),
    "create_placement_groups_from_args": ("diffusionrl.ray.placement_group", "create_placement_groups_from_args"),
    "create_placement_groups_from_runtime": ("diffusionrl.ray.placement_group", "create_placement_groups_from_runtime"),
    "remove_placement_group": ("diffusionrl.ray.placement_group", "remove_placement_group"),
    "BufferActor": ("diffusionrl.ray.buffer_actor", "BufferActor"),
    "create_buffer_actor": ("diffusionrl.ray.buffer_actor", "create_buffer_actor"),
    # Actor groups
    "ActorGroupHandle": ("diffusionrl.ray.group_base", "ActorGroupHandle"),
    "PlacementGroupActorPool": ("diffusionrl.ray.group_base", "PlacementGroupActorPool"),
    "ActorGroup": ("diffusionrl.ray.group_base", "ActorGroup"),
    "RolloutActorGroup": ("diffusionrl.ray.rollout_group", "RolloutActorGroup"),
    "TrainingActorGroup": ("diffusionrl.ray.training_group", "TrainingActorGroup"),
    "RolloutGroupRuntime": ("diffusionrl.ray.group_runtime", "RolloutGroupRuntime"),
    "TrainingGroupRuntime": ("diffusionrl.ray.group_runtime", "TrainingGroupRuntime"),
    "create_rollout_actor_group": ("diffusionrl.ray.group_factory", "create_rollout_actor_group"),
    "create_training_actor_group": ("diffusionrl.ray.group_factory", "create_training_actor_group"),
    # Actors
    "RayActor": ("diffusionrl.ray.actor_base", "RayActor"),
    "BaseTrainRayActor": ("diffusionrl.ray.actor_base", "BaseTrainRayActor"),
    "RolloutActor": ("diffusionrl.ray.rollout_actor", "RolloutActor"),
    "TrainingActor": ("diffusionrl.ray.training_actor", "TrainingActor"),
    # Utils - NOSET
    "NOSET_VISIBLE_DEVICES_ENV_VARS_LIST": (
        "diffusionrl.ray.ray_utils",
        "NOSET_VISIBLE_DEVICES_ENV_VARS_LIST",
    ),
    # Utils - Lock
    "DistributedLock": ("diffusionrl.ray.ray_utils", "DistributedLock"),
    "LockContext": ("diffusionrl.ray.ray_utils", "LockContext"),
    "create_distributed_lock": ("diffusionrl.ray.ray_utils", "create_distributed_lock"),
    "get_distributed_lock": ("diffusionrl.ray.ray_utils", "get_distributed_lock"),
    # Utils - Resource info
    "GPUInfo": ("diffusionrl.ray.ray_utils", "GPUInfo"),
    "NodeInfo": ("diffusionrl.ray.ray_utils", "NodeInfo"),
    "get_node_info": ("diffusionrl.ray.ray_utils", "get_node_info"),
    "get_node_ip": ("diffusionrl.ray.ray_utils", "get_node_ip"),
    "get_free_port": ("diffusionrl.ray.ray_utils", "get_free_port"),
    "get_consecutive_free_ports": ("diffusionrl.ray.ray_utils", "get_consecutive_free_ports"),
    # Utils - Ray operations
    "ray_get_with_retry": ("diffusionrl.ray.ray_utils", "ray_get_with_retry"),
    "ray_wait_with_progress": ("diffusionrl.ray.ray_utils", "ray_wait_with_progress"),
    "ray_get_async": ("diffusionrl.ray.ray_utils", "ray_get_async"),
    "batch_ray_get": ("diffusionrl.ray.ray_utils", "batch_ray_get"),
    "wait_for_placement_group": ("diffusionrl.ray.ray_utils", "wait_for_placement_group"),
    # Utils - Timing
    "timed": ("diffusionrl.ray.ray_utils", "timed"),
    # Utils - Memory
    "clear_gpu_memory": ("diffusionrl.ray.ray_utils", "clear_gpu_memory"),
    "get_gpu_memory_usage": ("diffusionrl.ray.ray_utils", "get_gpu_memory_usage"),
    "log_gpu_memory_usage": ("diffusionrl.ray.ray_utils", "log_gpu_memory_usage"),
    # Utils - Health check
    "check_actor_health": ("diffusionrl.ray.ray_utils", "check_actor_health"),
    "wait_for_actors_ready": ("diffusionrl.ray.ray_utils", "wait_for_actors_ready"),
    # Utils - Actor sampling
    "ActorSamplingExecutor": ("diffusionrl.ray.training_actor_sampling", "ActorSamplingExecutor"),
}

__all__ = list(_LAZY_ATTRS.keys())


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))
