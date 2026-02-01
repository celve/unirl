"""
diffusionrl Ray Placement Groups - GPU allocation and resource management.

Reference: slime/ray/placement_group.py
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import ray
from ray.util.placement_group import PlacementGroup

logger = logging.getLogger(__name__)

# Type alias: (PlacementGroup, bundle_indices, gpu_ids)
PlacementGroupResult = Tuple[PlacementGroup, List[int], List[int]]


@dataclass
class GRPOPlacementConfig:
    """Resource allocation configuration for GRPO training."""

    # Inference resources
    inference_num_nodes: int = 1
    inference_num_gpus_per_node: int = 4

    # Training resources
    training_num_nodes: int = 1
    training_num_gpus_per_node: int = 4

    # Dedicated reward computation resources (optional)
    reward_dedicated_num_gpus: int = 0  # 0 means no dedicated reward GPU pool

    # Deployment strategy
    colocate_inference_training: bool = False
    strategy: str = "PACK"  # "PACK" or "SPREAD"

    # Node-level dedicated reward configuration
    reward_dedicated_num_nodes: int = 0  # Dedicated reward nodes (0 = use reward_dedicated_num_gpus directly)
    reward_dedicated_num_gpus_per_node: int = 0  # GPUs per dedicated reward node
    reward_dedicated_gpus_per_actor: int = 1  # GPUs per dedicated reward actor
    reward_placement_strategy: str = "PACK"  # Reward PG strategy


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


def _group_bundles_by_node(
    gpu_info: List[Tuple[str, int, int]],
) -> List[Tuple[str, List[int], List[int]]]:
    """
    Group bundles by node and sort GPUs within each node.

    Returns:
        List of (node_ip, bundle_indices, gpu_ids) in sorted node order.
    """
    node_to_entries: Dict[str, List[Tuple[int, int]]] = {}
    for ip, gpu_id, bundle_idx in gpu_info:
        node_to_entries.setdefault(ip, []).append((gpu_id, bundle_idx))

    for ip in node_to_entries:
        node_to_entries[ip].sort(key=lambda x: x[0])

    grouped: List[Tuple[str, List[int], List[int]]] = []
    for ip in sorted(node_to_entries.keys()):
        gpu_bundle_pairs = node_to_entries[ip]
        gpu_ids = [gpu for gpu, _ in gpu_bundle_pairs]
        bundle_indices = [bundle for _, bundle in gpu_bundle_pairs]
        grouped.append((ip, bundle_indices, gpu_ids))

    return grouped


def _take_nodes(
    grouped: List[Tuple[str, List[int], List[int]]],
    start_node: int,
    num_nodes: int,
    gpus_per_node: int,
) -> Tuple[List[int], List[int]]:
    """
    Take a contiguous range of nodes and return their bundle indices/GPU ids.
    """
    if num_nodes <= 0:
        return [], []
    if start_node + num_nodes > len(grouped):
        raise ValueError(
            f"Not enough nodes for allocation: requested nodes [{start_node}, {start_node + num_nodes}), "
            f"available={len(grouped)}"
        )

    bundle_indices: List[int] = []
    gpu_ids: List[int] = []
    for _ip, bundles, gpus in grouped[start_node:start_node + num_nodes]:
        if gpus_per_node > len(bundles):
            raise ValueError(
                f"Not enough GPUs on node for allocation: requested {gpus_per_node}, available {len(bundles)}"
            )
        bundle_indices.extend(bundles[:gpus_per_node])
        gpu_ids.extend(gpus[:gpus_per_node])

    return bundle_indices, gpu_ids


def _flatten_grouped(
    grouped: List[Tuple[str, List[int], List[int]]],
    start_node: int,
) -> Tuple[List[int], List[int]]:
    """Flatten grouped bundles from a starting node index."""
    bundle_indices: List[int] = []
    gpu_ids: List[int] = []
    for _ip, bundles, gpus in grouped[start_node:]:
        bundle_indices.extend(bundles)
        gpu_ids.extend(gpus)
    return bundle_indices, gpu_ids


def _create_placement_group(
    num_gpus: int,
    strategy: str = "PACK",
    name: Optional[str] = None,
) -> PlacementGroup:
    """
    Create a placement group with uniform single-GPU bundles.

    Multi-GPU engines (e.g. FastVideo SP) are supported via the Slime pattern:
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
    config: GRPOPlacementConfig,
) -> Dict[str, Optional[PlacementGroupResult]]:
    """
    Create placement groups for GRPO training (unified single_pg mode).

    Creates one placement group with uniform {"GPU": 1} bundles, then
    slices bundles by node for inference / training / reward roles.

    Args:
        config: Placement configuration

    Returns:
        Dictionary with keys:
            - "inference": (pg, bundle_indices, gpu_ids) or None
            - "training": (pg, bundle_indices, gpu_ids) or None
            - "reward": (pg, bundle_indices, gpu_ids) or None
    """
    return _create_single_pg(config)


def _create_single_pg(
    config: GRPOPlacementConfig,
) -> Dict[str, Optional[PlacementGroupResult]]:
    """
    Create a single placement group and slice bundles by node for inference/training/reward.
    """
    result: Dict[str, Optional[PlacementGroupResult]] = {
        "inference": None,
        "training": None,
        "reward": None,
    }

    # Calculate total GPUs needed
    inference_total_gpus = config.inference_num_nodes * config.inference_num_gpus_per_node
    training_total_gpus = config.training_num_nodes * config.training_num_gpus_per_node

    # Calculate total reward GPUs
    if config.reward_dedicated_num_nodes > 0:
        reward_total_gpus = config.reward_dedicated_num_nodes * config.reward_dedicated_num_gpus_per_node
    else:
        reward_total_gpus = config.reward_dedicated_num_gpus

    if config.colocate_inference_training:
        # Colocate: inference and training share same GPU bundles
        shared_gpus = max(inference_total_gpus, training_total_gpus)
        total_gpus = shared_gpus + reward_total_gpus

        if total_gpus <= 0:
            return result

        pg, bundle_indices, gpu_ids = _create_colocate_pg(
            total_gpus=total_gpus,
            strategy=config.strategy,
            name="grpo_colocated",
        )

        # Shared bundles for inference/training
        result["inference"] = (pg, bundle_indices[:shared_gpus], gpu_ids[:shared_gpus])
        result["training"] = (pg, bundle_indices[:shared_gpus], gpu_ids[:shared_gpus])

        # Reward bundles (if any) are allocated after shared bundles
        if reward_total_gpus > 0:
            reward_start = shared_gpus
            reward_end = shared_gpus + reward_total_gpus
            result["reward"] = (pg, bundle_indices[reward_start:reward_end], gpu_ids[reward_start:reward_end])

        return result

    # Non-colocate: single PG sliced by node for inference/training/reward
    total_gpus = inference_total_gpus + training_total_gpus + reward_total_gpus
    if total_gpus <= 0:
        return result

    pg = _create_placement_group(
        num_gpus=total_gpus,
        strategy=config.strategy,
        name="grpo_single",
    )

    gpu_info = _get_gpu_info_from_pg(pg, total_gpus)
    grouped = _group_bundles_by_node(gpu_info)

    # Check if we have a single node scenario requiring GPU-level slicing
    num_available_nodes = len(grouped)
    required_nodes = config.inference_num_nodes + config.training_num_nodes

    if num_available_nodes == 1 and required_nodes > 1:
        # Single node separate mode: slice by GPU index instead of by node
        # Flatten all GPU info from the single node
        _, all_bundle_indices, all_gpu_ids = grouped[0]
        gpu_cursor = 0

        if inference_total_gpus > 0:
            inf_end = gpu_cursor + inference_total_gpus
            result["inference"] = (pg, all_bundle_indices[gpu_cursor:inf_end], all_gpu_ids[gpu_cursor:inf_end])
            gpu_cursor = inf_end

        if training_total_gpus > 0:
            train_end = gpu_cursor + training_total_gpus
            result["training"] = (pg, all_bundle_indices[gpu_cursor:train_end], all_gpu_ids[gpu_cursor:train_end])
            gpu_cursor = train_end

        # Reward allocation from remaining GPUs
        if reward_total_gpus > 0:
            reward_end = gpu_cursor + reward_total_gpus
            if reward_end > len(all_bundle_indices):
                raise ValueError(
                    f"Not enough GPUs for reward allocation: requested {reward_total_gpus}, "
                    f"available {len(all_bundle_indices) - gpu_cursor}"
                )
            result["reward"] = (pg, all_bundle_indices[gpu_cursor:reward_end], all_gpu_ids[gpu_cursor:reward_end])

        return result

    # Multi-node scenario: slice by nodes for inference/training
    node_cursor = 0
    if inference_total_gpus > 0:
        inf_bundle_indices, inf_gpu_ids = _take_nodes(
            grouped,
            node_cursor,
            config.inference_num_nodes,
            config.inference_num_gpus_per_node,
        )
        node_cursor += config.inference_num_nodes
        result["inference"] = (pg, inf_bundle_indices, inf_gpu_ids)

    if training_total_gpus > 0:
        train_bundle_indices, train_gpu_ids = _take_nodes(
            grouped,
            node_cursor,
            config.training_num_nodes,
            config.training_num_gpus_per_node,
        )
        node_cursor += config.training_num_nodes
        result["training"] = (pg, train_bundle_indices, train_gpu_ids)

    # Reward allocation
    if reward_total_gpus > 0:
        if config.reward_dedicated_num_nodes > 0:
            reward_bundle_indices, reward_gpu_ids = _take_nodes(
                grouped,
                node_cursor,
                config.reward_dedicated_num_nodes,
                config.reward_dedicated_num_gpus_per_node,
            )
            result["reward"] = (pg, reward_bundle_indices, reward_gpu_ids)
        else:
            remaining_bundle_indices, remaining_gpu_ids = _flatten_grouped(grouped, node_cursor)
            if reward_total_gpus > len(remaining_bundle_indices):
                raise ValueError(
                    f"Not enough GPUs for reward allocation: requested {reward_total_gpus}, "
                    f"available {len(remaining_bundle_indices)}"
                )
            result["reward"] = (
                pg,
                remaining_bundle_indices[:reward_total_gpus],
                remaining_gpu_ids[:reward_total_gpus],
            )

    return result


def create_placement_groups_from_args(args) -> Dict[str, Optional[PlacementGroupResult]]:
    """
    Create placement groups from GRPOArguments.

    Always uses unified single_pg mode with uniform {"GPU": 1} bundles.
    Multi-GPU engines are handled at the actor level via NOSET + base_gpu_id.

    Args:
        args: GRPOArguments instance

    Returns:
        Dictionary of placement group results
    """
    config = GRPOPlacementConfig(
        inference_num_nodes=args.inference_num_nodes,
        inference_num_gpus_per_node=args.inference_num_gpus_per_node,
        training_num_nodes=args.training_num_nodes,
        training_num_gpus_per_node=args.training_num_gpus_per_node,
        reward_dedicated_num_gpus=args.reward_dedicated_num_gpus,
        colocate_inference_training=args.colocate_inference_training,
        strategy=args.placement_strategy,
        # Node-level reward configuration
        reward_dedicated_num_nodes=getattr(args, "reward_dedicated_num_nodes", 0),
        reward_dedicated_num_gpus_per_node=getattr(args, "reward_dedicated_num_gpus_per_node", 0),
        reward_dedicated_gpus_per_actor=getattr(args, "reward_dedicated_gpus_per_actor", 1),
        reward_placement_strategy=getattr(args, "reward_placement_strategy", "PACK"),
    )
    return create_placement_groups(config)


def remove_placement_group(pg: PlacementGroup) -> None:
    """Remove a placement group and free its resources."""
    try:
        ray.util.remove_placement_group(pg)
        logger.info("Placement group removed")
    except Exception as e:
        logger.warning(f"Failed to remove placement group: {e}")


def get_bundle_resources(pg: PlacementGroup, bundle_index: int) -> Dict[str, float]:
    """Get the resources allocated to a specific bundle."""
    bundle_specs = pg.bundle_specs
    if bundle_index < len(bundle_specs):
        return bundle_specs[bundle_index]
    return {}
