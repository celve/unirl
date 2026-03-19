"""
diffusionrl Ray Placement Groups - GPU allocation and resource management.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import ray
from ray.util.placement_group import PlacementGroup

from diffusionrl.config.arguments import is_training_actor_sampling_mode
from diffusionrl.config.rollout_topology import rollout_mode_is_colocated

logger = logging.getLogger(__name__)

# Type alias: (PlacementGroup, bundle_indices, gpu_ids)
PlacementGroupResult = Tuple[PlacementGroup, List[int], List[int]]


@dataclass
class RuntimePlacementConfig:
    """Resource allocation configuration for diffusionRL training."""

    # Rollout resources
    rollout_num_nodes: int = 1
    rollout_num_gpus_per_node: int = 4

    # Training resources
    training_num_nodes: int = 1
    training_num_gpus_per_node: int = 4

    # Dedicated reward computation resources (optional)
    reward_dedicated_num_gpus: int = 0  # 0 means no dedicated reward GPU pool

    # Deployment strategy
    colocate_rollout: bool = False
    strategy: str = "PACK"  # "PACK" or "SPREAD"

    # Node-level dedicated reward configuration
    reward_dedicated_num_nodes: int = 0  # Dedicated reward nodes (0 = use reward_dedicated_num_gpus directly)
    reward_dedicated_num_gpus_per_node: int = 0  # GPUs per dedicated reward node
    reward_dedicated_gpus_per_actor: int = 1  # GPUs per dedicated reward actor


@ray.remote(num_cpus=1)
class InfoActor:
    """
    Actor to gather GPU information from a placement group bundle.

    Used to detect which GPU each bundle is allocated to and the node IP.
    """

    def __init__(self):
        import socket
        import os
        self.ip = socket.gethostbyname(socket.gethostname())
        # Get CUDA_VISIBLE_DEVICES
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_devices:
            self.gpu_ids = [int(x) for x in cuda_devices.split(",")]
        else:
            self.gpu_ids = []

    def get_info(self) -> Tuple[str, List[int]]:
        """Return node IP and GPU IDs."""
        return self.ip, self.gpu_ids


def _get_gpu_info_from_pg(
    pg: PlacementGroup,
    num_bundles: int,
) -> List[Tuple[str, int, int]]:
    """
    Get GPU information for each bundle in a placement group.

    Returns:
        List of (node_ip, gpu_id, bundle_index) tuples
    """
    # Create InfoActors on each bundle
    info_actors = []
    for i in range(num_bundles):
        actor = InfoActor.options(
            scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=i,
            )
        ).remote()
        info_actors.append((actor, i))

    # Gather information
    gpu_info = []
    for actor, bundle_idx in info_actors:
        ip, gpu_ids = ray.get(actor.get_info.remote())
        # Each bundle typically has one GPU
        gpu_id = gpu_ids[0] if gpu_ids else 0
        gpu_info.append((ip, gpu_id, bundle_idx))
        ray.kill(actor)

    return gpu_info


def _reorder_bundles_by_node(
    gpu_info: List[Tuple[str, int, int]],
) -> Tuple[List[int], List[int]]:
    """
    Reorder bundles so that GPUs on the same node are consecutive.

    This is important for efficient NCCL communication.

    Returns:
        (bundle_indices, gpu_ids) - reordered
    """
    # Group by node
    node_to_bundles: Dict[str, List[Tuple[int, int]]] = {}
    for ip, gpu_id, bundle_idx in gpu_info:
        if ip not in node_to_bundles:
            node_to_bundles[ip] = []
        node_to_bundles[ip].append((gpu_id, bundle_idx))

    # Sort within each node by GPU ID
    for ip in node_to_bundles:
        node_to_bundles[ip].sort(key=lambda x: x[0])

    # Flatten in node order
    bundle_indices = []
    gpu_ids = []
    for ip in sorted(node_to_bundles.keys()):
        for gpu_id, bundle_idx in node_to_bundles[ip]:
            bundle_indices.append(bundle_idx)
            gpu_ids.append(gpu_id)

    return bundle_indices, gpu_ids


def _create_placement_group(
    num_gpus: int,
    strategy: str = "PACK",
    name: Optional[str] = None,
) -> PlacementGroup:
    """
    Create a placement group with uniform single-GPU bundles.

    Multi-GPU rollout engines (for example SGLang TP) are supported via the Slime pattern:
    NOSET_VISIBLE_DEVICES + base_gpu_id + manual CUDA_VISIBLE_DEVICES,
    so every bundle is always {"GPU": 1, "CPU": 1}.

    Args:
        num_gpus: Total number of GPUs to allocate
        strategy: Placement strategy ("PACK" or "SPREAD")
        name: Optional name for the placement group

    Returns:
        Created PlacementGroup
    """
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = ray.util.placement_group(bundles, strategy=strategy, name=name)
    logger.info(f"Waiting for placement group '{name}' with {num_gpus} GPUs...")
    ray.get(pg.ready())
    logger.info(f"Placement group '{name}' ready")
    return pg


def _create_colocate_pg(
    total_gpus: int,
    strategy: str = "PACK",
    name: str = "grpo_colocated",
) -> PlacementGroupResult:
    """
    Create a colocated placement group with CPU=2 per GPU bundle.

    Returns:
        (pg, bundle_indices, gpu_ids)
    """
    bundles = [{"GPU": 1, "CPU": 2} for _ in range(total_gpus)]
    pg = ray.util.placement_group(
        bundles,
        strategy=strategy,
        name=name,
    )
    logger.info(
        f"Waiting for placement group '{name}' with {total_gpus} GPUs (2 CPUs per bundle for colocate)..."
    )
    ray.get(pg.ready())
    logger.info(f"Placement group '{name}' ready")

    gpu_info = _get_gpu_info_from_pg(pg, total_gpus)
    bundle_indices, gpu_ids = _reorder_bundles_by_node(gpu_info)
    return pg, bundle_indices, gpu_ids


def create_placement_groups(
    config: RuntimePlacementConfig,
) -> Dict[str, Optional[PlacementGroupResult]]:
    """
    Create placement groups for diffusionRL training (unified single_pg mode).

    Creates one placement group with uniform {"GPU": 1} bundles, then
    slices bundles by linear offsets for rollout / training / reward roles.

    Args:
        config: Placement configuration

    Returns:
        Dictionary with keys:
            - "rollout": (pg, bundle_indices, gpu_ids) or None
            - "training": (pg, bundle_indices, gpu_ids) or None
            - "reward": (pg, bundle_indices, gpu_ids) or None
    """
    return _create_single_pg(config)


def _create_single_pg(
    config: RuntimePlacementConfig,
) -> Dict[str, Optional[PlacementGroupResult]]:
    """
    Create a single placement group and slice bundles by linear offsets.
    """
    result: Dict[str, Optional[PlacementGroupResult]] = {
        "rollout": None,
        "training": None,
        "reward": None,
    }

    # Calculate total GPUs needed
    rollout_total_gpus = config.rollout_num_nodes * config.rollout_num_gpus_per_node
    training_total_gpus = config.training_num_nodes * config.training_num_gpus_per_node

    # Calculate total reward GPUs
    if config.reward_dedicated_num_nodes > 0:
        reward_total_gpus = config.reward_dedicated_num_nodes * config.reward_dedicated_num_gpus_per_node
    else:
        reward_total_gpus = config.reward_dedicated_num_gpus

    if config.colocate_rollout:
        # Colocate: rollout and training share same GPU bundles
        shared_gpus = max(rollout_total_gpus, training_total_gpus)
        total_gpus = shared_gpus + reward_total_gpus

        if total_gpus <= 0:
            return result

        pg, bundle_indices, gpu_ids = _create_colocate_pg(
            total_gpus=total_gpus,
            strategy=config.strategy,
            name="diffusionrl_colocated",
        )

        # Shared bundles for rollout/training
        result["rollout"] = (pg, bundle_indices[:shared_gpus], gpu_ids[:shared_gpus])
        result["training"] = (pg, bundle_indices[:shared_gpus], gpu_ids[:shared_gpus])

        # Reward bundles (if any) are allocated after shared bundles
        if reward_total_gpus > 0:
            reward_start = shared_gpus
            reward_end = shared_gpus + reward_total_gpus
            result["reward"] = (pg, bundle_indices[reward_start:reward_end], gpu_ids[reward_start:reward_end])

        return result

    # Non-colocate: single PG sliced by linear offsets for rollout/training/reward.
    total_gpus = rollout_total_gpus + training_total_gpus + reward_total_gpus
    if total_gpus <= 0:
        return result

    pg = _create_placement_group(
        num_gpus=total_gpus,
        strategy=config.strategy,
        name="diffusionrl_single",
    )

    gpu_info = _get_gpu_info_from_pg(pg, total_gpus)
    bundle_indices, gpu_ids = _reorder_bundles_by_node(gpu_info)

    cursor = 0
    if rollout_total_gpus > 0:
        rollout_end = cursor + rollout_total_gpus
        result["rollout"] = (pg, bundle_indices[cursor:rollout_end], gpu_ids[cursor:rollout_end])
        cursor = rollout_end

    if training_total_gpus > 0:
        train_end = cursor + training_total_gpus
        result["training"] = (pg, bundle_indices[cursor:train_end], gpu_ids[cursor:train_end])
        cursor = train_end

    if reward_total_gpus > 0:
        reward_end = cursor + reward_total_gpus
        if reward_end > len(bundle_indices):
            raise ValueError(
                f"Not enough GPUs for reward allocation: requested {reward_total_gpus}, "
                f"available {len(bundle_indices) - cursor}"
            )
        result["reward"] = (pg, bundle_indices[cursor:reward_end], gpu_ids[cursor:reward_end])

    return result


def create_placement_groups_from_args(args) -> Dict[str, Optional[PlacementGroupResult]]:
    """
    Create placement groups from TrainingArguments.

    Always uses unified single_pg mode with uniform {"GPU": 1} bundles.
    Multi-GPU engines are handled at the actor level via NOSET + base_gpu_id.

    Args:
        args: TrainingArguments instance

    Returns:
        Dictionary of placement group results
    """
    debug_mode = str(getattr(args.debug, "debug_mode", "none") or "none").strip().lower()
    rollout_num_nodes = int(args.ray.rollout_num_nodes)
    rollout_num_gpus_per_node = int(args.ray.rollout_num_gpus_per_node)
    training_num_nodes = int(args.ray.training_num_nodes)
    training_num_gpus_per_node = int(args.ray.training_num_gpus_per_node)
    reward_dedicated_num_gpus = int(args.reward.reward_dedicated_num_gpus)
    reward_dedicated_num_nodes = int(getattr(args.reward, "reward_dedicated_num_nodes", 0))
    reward_dedicated_num_gpus_per_node = int(getattr(args.reward, "reward_dedicated_num_gpus_per_node", 0))
    training_actor_sampling_mode = is_training_actor_sampling_mode(args)

    if debug_mode == "train_only":
        rollout_num_nodes = 0
        rollout_num_gpus_per_node = 0
        reward_dedicated_num_gpus = 0
        reward_dedicated_num_nodes = 0
        reward_dedicated_num_gpus_per_node = 0
        logger.info(
            "Debug mode train_only: rollout/reward placement is disabled."
        )
    elif training_actor_sampling_mode:
        rollout_num_nodes = 0
        rollout_num_gpus_per_node = 0
        logger.info(
            "Direct training-actor sampling active: rollout placement is disabled."
        )

    config = RuntimePlacementConfig(
        rollout_num_nodes=rollout_num_nodes,
        rollout_num_gpus_per_node=rollout_num_gpus_per_node,
        training_num_nodes=training_num_nodes,
        training_num_gpus_per_node=training_num_gpus_per_node,
        reward_dedicated_num_gpus=reward_dedicated_num_gpus,
        colocate_rollout=rollout_mode_is_colocated(args.rollout.mode),
        strategy=args.ray.placement_strategy,
        # Node-level reward configuration
        reward_dedicated_num_nodes=reward_dedicated_num_nodes,
        reward_dedicated_num_gpus_per_node=reward_dedicated_num_gpus_per_node,
        reward_dedicated_gpus_per_actor=getattr(args.reward, "reward_dedicated_gpus_per_actor", 1),
    )
    return create_placement_groups(config)

def remove_placement_group(pg: PlacementGroup) -> None:
    """Remove a placement group and free its resources."""
    try:
        ray.util.remove_placement_group(pg)
        logger.info("Placement group removed")
    except Exception as e:
        logger.warning(f"Failed to remove placement group: {e}")
