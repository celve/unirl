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
    "InferenceActorGroup": ("diffusionrl.ray.groups.inference", "InferenceActorGroup"),
    "TrainingActorGroup": ("diffusionrl.ray.groups.training", "TrainingActorGroup"),
    "create_inference_actor_group": ("diffusionrl.ray.groups.factory", "create_inference_actor_group"),
    "create_training_actor_group": ("diffusionrl.ray.groups.factory", "create_training_actor_group"),
    # Actors
    "RayActor": ("diffusionrl.ray.actors", "RayActor"),
    "BaseTrainRayActor": ("diffusionrl.ray.actors", "BaseTrainRayActor"),
    "InferenceActor": ("diffusionrl.ray.actors", "InferenceActor"),
    "TrainingActor": ("diffusionrl.ray.actors", "TrainingActor"),
    # Sampling mode plugins
    "SamplingModePlugin": ("diffusionrl.ray.sampling_mode", "SamplingModePlugin"),
    "InferenceSamplingMode": ("diffusionrl.ray.sampling_mode", "InferenceSamplingMode"),
    "TrainingSamplingMode": ("diffusionrl.ray.sampling_mode", "TrainingSamplingMode"),
    "create_sampling_mode_plugin": ("diffusionrl.ray.sampling_mode", "create_sampling_mode_plugin"),
    # Utils - NOSET
    "NOSET_VISIBLE_DEVICES_ENV_VARS_LIST": ("diffusionrl.ray.utils", "NOSET_VISIBLE_DEVICES_ENV_VARS_LIST"),
    # Utils - Lock
    "DistributedLock": ("diffusionrl.ray.utils", "DistributedLock"),
    "LockContext": ("diffusionrl.ray.utils", "LockContext"),
    "create_distributed_lock": ("diffusionrl.ray.utils", "create_distributed_lock"),
    "get_distributed_lock": ("diffusionrl.ray.utils", "get_distributed_lock"),
    # Utils - Resource info
    "GPUInfo": ("diffusionrl.ray.utils", "GPUInfo"),
    "NodeInfo": ("diffusionrl.ray.utils", "NodeInfo"),
    "get_node_info": ("diffusionrl.ray.utils", "get_node_info"),
    "get_node_ip": ("diffusionrl.ray.utils", "get_node_ip"),
    "get_free_port": ("diffusionrl.ray.utils", "get_free_port"),
    "get_consecutive_free_ports": ("diffusionrl.ray.utils", "get_consecutive_free_ports"),
    # Utils - Ray operations
    "ray_get_with_retry": ("diffusionrl.ray.utils", "ray_get_with_retry"),
    "ray_wait_with_progress": ("diffusionrl.ray.utils", "ray_wait_with_progress"),
    "ray_get_async": ("diffusionrl.ray.utils", "ray_get_async"),
    "batch_ray_get": ("diffusionrl.ray.utils", "batch_ray_get"),
    "wait_for_placement_group": ("diffusionrl.ray.utils", "wait_for_placement_group"),
    # Utils - Timing
    "Timer": ("diffusionrl.ray.utils", "Timer"),
    "timed": ("diffusionrl.ray.utils", "timed"),
    # Utils - Memory
    "clear_gpu_memory": ("diffusionrl.ray.utils", "clear_gpu_memory"),
    "get_gpu_memory_usage": ("diffusionrl.ray.utils", "get_gpu_memory_usage"),
    "log_gpu_memory_usage": ("diffusionrl.ray.utils", "log_gpu_memory_usage"),
    # Utils - Health check
    "check_actor_health": ("diffusionrl.ray.utils", "check_actor_health"),
    "wait_for_actors_ready": ("diffusionrl.ray.utils", "wait_for_actors_ready"),
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
