"""Distributed control-plane facade with lazy exports."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    # Placement groups
    "RuntimePlacementConfig": ("diffusionrl.ray.placement_group", "RuntimePlacementConfig"),
    "PlacementGroupResult": ("diffusionrl.ray.placement_group", "PlacementGroupResult"),
    "create_placement_groups": ("diffusionrl.ray.placement_group", "create_placement_groups"),
    "create_placement_groups_from_runtime": ("diffusionrl.ray.placement_group", "create_placement_groups_from_runtime"),
    "remove_placement_group": ("diffusionrl.ray.placement_group", "remove_placement_group"),
    "BufferActor": ("diffusionrl.ray.buffer_actor", "BufferActor"),
    "create_buffer_actor": ("diffusionrl.ray.buffer_actor", "create_buffer_actor"),
    # Actor groups
    "ActorHandleGroup": ("diffusionrl.ray.group_base", "ActorHandleGroup"),
    "PlacementGroupActorPool": ("diffusionrl.ray.group_base", "PlacementGroupActorPool"),
    "ActorGroup": ("diffusionrl.ray.group_base", "ActorGroup"),
    # Actors
    "RayActor": ("diffusionrl.ray.actor_base", "RayActor"),
    "BaseTrainRayActor": ("diffusionrl.ray.actor_base", "BaseTrainRayActor"),
    "RolloutActor": ("diffusionrl.ray.rollout_actor", "RolloutActor"),
    "RolloutActorConfig": ("diffusionrl.ray.actor_config", "RolloutActorConfig"),
    "TrainingActorConfig": ("diffusionrl.ray.actor_config", "TrainingActorConfig"),
    # Utils - node-level constants
    "NOSET_VISIBLE_DEVICES_ENV_VARS_LIST": (
        "diffusionrl.ray.utils.node",
        "NOSET_VISIBLE_DEVICES_ENV_VARS_LIST",
    ),
    # Utils - network helpers
    "get_node_ip": ("diffusionrl.ray.utils.net", "get_node_ip"),
    "get_free_port": ("diffusionrl.ray.utils.net", "get_free_port"),
    # Mixins
    "TrainingWeightSyncMixin": (
        "diffusionrl.ray.mixins.training_weight_sync",
        "TrainingWeightSyncMixin",
    ),
    "RolloutWeightSyncMixin": (
        "diffusionrl.ray.mixins.rollout_weight_sync",
        "RolloutWeightSyncMixin",
    ),
    "RolloutPipelineMixin": (
        "diffusionrl.ray.mixins.rollout_pipeline",
        "RolloutPipelineMixin",
    ),
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
