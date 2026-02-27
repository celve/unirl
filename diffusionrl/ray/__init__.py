"""Distributed control-plane facade with lazy exports."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    # Placement groups
    "GRPOPlacementConfig": ("diffusionrl.ray.placement_group", "GRPOPlacementConfig"),
    "PlacementGroupResult": ("diffusionrl.ray.placement_group", "PlacementGroupResult"),
    "create_placement_groups": ("diffusionrl.ray.placement_group", "create_placement_groups"),
    "create_placement_groups_from_args": ("diffusionrl.ray.placement_group", "create_placement_groups_from_args"),
    "remove_placement_group": ("diffusionrl.ray.placement_group", "remove_placement_group"),
    # Rollout manager
    "RolloutManager": ("diffusionrl.ray.rollout_manager", "RolloutManager"),
    "create_rollout_manager": ("diffusionrl.ray.rollout_manager", "create_rollout_manager"),
    "RolloutBufferActor": ("diffusionrl.ray.rollout_buffer", "RolloutBufferActor"),
    "create_rollout_buffer_actor": ("diffusionrl.ray.rollout_buffer", "create_rollout_buffer_actor"),
    # Actor groups
    "BaseActorGroup": ("diffusionrl.ray.groups.base", "BaseActorGroup"),
    "RolloutActorGroup": ("diffusionrl.ray.groups.rollout", "RolloutActorGroup"),
    "TrainingActorGroup": ("diffusionrl.ray.groups.training", "TrainingActorGroup"),
    "create_rollout_actor_group": ("diffusionrl.ray.groups.factory", "create_rollout_actor_group"),
    "create_training_actor_group": ("diffusionrl.ray.groups.factory", "create_training_actor_group"),
    # Actors
    "RayActor": ("diffusionrl.ray.actors", "RayActor"),
    "BaseTrainRayActor": ("diffusionrl.ray.actors", "BaseTrainRayActor"),
    "RolloutActor": ("diffusionrl.ray.actors", "RolloutActor"),
    "TrainingActor": ("diffusionrl.ray.actors", "TrainingActor"),
    # Utils - NOSET
    "NOSET_VISIBLE_DEVICES_ENV_VARS_LIST": (
        "diffusionrl.ray.utils.distributed",
        "NOSET_VISIBLE_DEVICES_ENV_VARS_LIST",
    ),
    # Utils - Lock
    "DistributedLock": ("diffusionrl.ray.utils.distributed", "DistributedLock"),
    "LockContext": ("diffusionrl.ray.utils.distributed", "LockContext"),
    "create_distributed_lock": ("diffusionrl.ray.utils.distributed", "create_distributed_lock"),
    "get_distributed_lock": ("diffusionrl.ray.utils.distributed", "get_distributed_lock"),
    # Utils - Resource info
    "GPUInfo": ("diffusionrl.ray.utils.distributed", "GPUInfo"),
    "NodeInfo": ("diffusionrl.ray.utils.distributed", "NodeInfo"),
    "get_node_info": ("diffusionrl.ray.utils.distributed", "get_node_info"),
    "get_node_ip": ("diffusionrl.ray.utils.distributed", "get_node_ip"),
    "get_free_port": ("diffusionrl.ray.utils.distributed", "get_free_port"),
    "get_consecutive_free_ports": ("diffusionrl.ray.utils.distributed", "get_consecutive_free_ports"),
    # Utils - Ray operations
    "ray_get_with_retry": ("diffusionrl.ray.utils.distributed", "ray_get_with_retry"),
    "ray_wait_with_progress": ("diffusionrl.ray.utils.distributed", "ray_wait_with_progress"),
    "ray_get_async": ("diffusionrl.ray.utils.distributed", "ray_get_async"),
    "batch_ray_get": ("diffusionrl.ray.utils.distributed", "batch_ray_get"),
    "wait_for_placement_group": ("diffusionrl.ray.utils.distributed", "wait_for_placement_group"),
    # Utils - Timing
    "Timer": ("diffusionrl.ray.utils.distributed", "Timer"),
    "timed": ("diffusionrl.ray.utils.distributed", "timed"),
    # Utils - Memory
    "clear_gpu_memory": ("diffusionrl.ray.utils.distributed", "clear_gpu_memory"),
    "get_gpu_memory_usage": ("diffusionrl.ray.utils.distributed", "get_gpu_memory_usage"),
    "log_gpu_memory_usage": ("diffusionrl.ray.utils.distributed", "log_gpu_memory_usage"),
    # Utils - Health check
    "check_actor_health": ("diffusionrl.ray.utils.distributed", "check_actor_health"),
    "wait_for_actors_ready": ("diffusionrl.ray.utils.distributed", "wait_for_actors_ready"),
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
