"""
diffusionrl Ray Module - Distributed training infrastructure.

Exposes factory functions for creating placement groups and actor groups.
"""
from .placement_group import (
    GRPOPlacementConfig,
    PlacementGroupResult,
    create_placement_groups,
    create_placement_groups_from_args,
    remove_placement_group,
)
from .rollout_manager import (
    RolloutManager,
    create_rollout_manager,
)
from .actor_group import (
    BaseActorGroup,
    InferenceActorGroup,
    TrainingActorGroup,
    create_inference_actor_group,
    create_training_actor_group,
)
from .actors import (
    RayActor,
    BaseTrainRayActor,
    InferenceActor,
    TrainingActor,
)
from .utils import (
    NOSET_VISIBLE_DEVICES_ENV_VARS_LIST,
    DistributedLock,
    LockContext,
    create_distributed_lock,
    get_distributed_lock,
    GPUInfo,
    NodeInfo,
    get_node_info,
    get_node_ip,
    get_free_port,
    get_consecutive_free_ports,
    ray_get_with_retry,
    ray_wait_with_progress,
    ray_get_async,
    batch_ray_get,
    wait_for_placement_group,
    Timer,
    timed,
    clear_gpu_memory,
    get_gpu_memory_usage,
    log_gpu_memory_usage,
    check_actor_health,
    wait_for_actors_ready,
)

__all__ = [
    # Placement groups
    "GRPOPlacementConfig",
    "PlacementGroupResult",
    "create_placement_groups",
    "create_placement_groups_from_args",
    "remove_placement_group",
    # Rollout manager
    "RolloutManager",
    "create_rollout_manager",
    # Actor groups
    "BaseActorGroup",
    "InferenceActorGroup",
    "TrainingActorGroup",
    "create_inference_actor_group",
    "create_training_actor_group",
    # Actors
    "RayActor",
    "BaseTrainRayActor",
    "InferenceActor",
    "TrainingActor",
    # Utils - NOSET
    "NOSET_VISIBLE_DEVICES_ENV_VARS_LIST",
    # Utils - Lock
    "DistributedLock",
    "LockContext",
    "create_distributed_lock",
    "get_distributed_lock",
    # Utils - Resource info
    "GPUInfo",
    "NodeInfo",
    "get_node_info",
    "get_node_ip",
    "get_free_port",
    "get_consecutive_free_ports",
    # Utils - Ray operations
    "ray_get_with_retry",
    "ray_wait_with_progress",
    "ray_get_async",
    "batch_ray_get",
    "wait_for_placement_group",
    # Utils - Timing
    "Timer",
    "timed",
    # Utils - Memory
    "clear_gpu_memory",
    "get_gpu_memory_usage",
    "log_gpu_memory_usage",
    # Utils - Health check
    "check_actor_health",
    "wait_for_actors_ready",
]
