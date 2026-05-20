"""
diffusionrl Ray placement - GPU allocation and resource management.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from diffusionrl.config.registration import register_config
from diffusionrl.config.require import require

logger = logging.getLogger(__name__)


@register_config(
    group="placement",
    name="default",
    target="diffusionrl.ray.placement.Placement.from_config",
)
@dataclass
class PlacementConfig:
    """Spec for placement-group allocation. Pure inputs, no side effects."""

    # Rollout resources
    num_rollout_nodes: int = 1
    num_rollout_gpus_per_node: int = 8
    num_rollout_gpus_per_actor: int = 1

    # Training resources
    num_train_nodes: int = 1
    num_train_gpus_per_node: int = 8
    num_train_gpus_per_actor: int = 1

    colocate: bool = True
    # Per-actor Ray GPU claim when train and rollout share each bundle. Two
    # actors with num_gpus=1 cannot live on the same 1-GPU bundle, so colocate
    # mode lies to Ray's scheduler: both sides take a fractional slice (sum
    # ≤ 1.0). In practice each side uses the full GPU at different times via
    # offload_train / offload_rollout. Ignored when colocate=False.
    colocate_gpu_fraction: float = 0.5
    strategy: str = "PACK"  # "PACK" or "SPREAD"
    allow_noset_multi_gpu_inference: bool = False

    def __post_init__(self):
        require(self.num_rollout_gpus_per_actor >= 1, "num_rollout_gpus_per_actor must be >= 1")
        require(self.num_train_gpus_per_actor >= 1, "num_train_gpus_per_actor must be >= 1")
        require(
            0.0 < self.colocate_gpu_fraction <= 1.0,
            f"colocate_gpu_fraction must be in (0, 1]; got {self.colocate_gpu_fraction!r}",
        )
        require(
            self.total_rollout_gpus % self.num_rollout_gpus_per_actor == 0,
            "total_rollout_gpus not divisible by num_rollout_gpus_per_actor",
        )
        require(
            self.total_train_gpus % self.num_train_gpus_per_actor == 0,
            "total_train_gpus not divisible by num_train_gpus_per_actor",
        )
        # ``num_rollout_nodes=0`` and ``num_rollout_gpus_per_node=0`` is the
        # canonical "no separate rollout" config (NFT all-in colocated training);
        # only the train-side counts are required to be >= 1.
        require(
            self.num_train_nodes >= 1, f"PlacementConfig.num_train_nodes must be >= 1; got {self.num_train_nodes!r}"
        )
        require(
            self.num_train_gpus_per_node >= 1,
            f"PlacementConfig.num_train_gpus_per_node must be >= 1; got {self.num_train_gpus_per_node!r}",
        )

    @property
    def total_rollout_gpus(self) -> int:
        return self.num_rollout_nodes * self.num_rollout_gpus_per_node

    @property
    def total_train_gpus(self) -> int:
        return self.num_train_nodes * self.num_train_gpus_per_node

    @property
    def total_gpus(self) -> int:
        if self.colocate:
            return max(self.total_rollout_gpus, self.total_train_gpus)
        return self.total_rollout_gpus + self.total_train_gpus

    @property
    def num_rollout_actors(self) -> int:
        if self.total_rollout_gpus <= 0:
            return 0
        return self.total_rollout_gpus // self.num_rollout_gpus_per_actor

    @property
    def num_train_actors(self) -> int:
        if self.total_train_gpus <= 0:
            return 0
        return self.total_train_gpus // self.num_train_gpus_per_actor


@dataclass
class ActorPlacement:
    """Self-contained spawn recipe for a single actor.

    All stripe GPUs are colocated on ``node_ip``. ``gpu_ids`` length equals the
    stride (1 for single-GPU actors, > 1 for NOSET TP actors).
    """

    rank: int
    bundle_idx: int
    gpu_ids: Tuple[int, ...]
    node_ip: str


@dataclass
class Placement:
    """Live placement group + role-specific bundle layout."""

    config: PlacementConfig
    pg: PlacementGroup
    node_ips: Tuple[str, ...] = field(default=())
    gpu_ids: Tuple[int, ...] = field(default=())
    train_master: Optional[Tuple[str, int]] = None

    @classmethod
    def from_config(cls, config: PlacementConfig) -> "Placement":
        total_gpus = config.total_gpus
        require(total_gpus > 0, "PlacementConfig must allocate >= 1 GPU")
        bundle = {"GPU": 1, "CPU": 2 if config.colocate else 1}
        bundles = [dict(bundle) for _ in range(total_gpus)]
        name = "diffusionrl_colocated" if config.colocate else "diffusionrl_separate"
        pg = ray.util.placement_group(bundles, strategy=config.strategy, name=name)
        ray.get(pg.ready())
        probed = _probe_pg(pg, total_gpus)
        node_ips = tuple(ip for ip, _, _ in probed)
        gpu_ids = tuple(gid for _, gid, _ in probed)

        if config.total_train_gpus > 0:
            rank0_bundle = 0 if config.colocate else config.total_rollout_gpus
            train_master = (probed[rank0_bundle][0], int(probed[rank0_bundle][2]))
        else:
            train_master = None

        return cls(
            config=config,
            pg=pg,
            node_ips=node_ips,
            gpu_ids=gpu_ids,
            train_master=train_master,
        )

    @property
    def rollout_bundles(self) -> List[int]:
        r = self.config.total_rollout_gpus
        return list(range(r)) if r > 0 else []

    @property
    def train_bundles(self) -> List[int]:
        t = self.config.total_train_gpus
        if t <= 0:
            return []
        if self.config.colocate:
            return list(range(t))
        r = self.config.total_rollout_gpus
        return list(range(r, r + t))

    @property
    def rollout_actors(self) -> List[ActorPlacement]:
        return self._build_actors(self.rollout_bundles, self.config.num_rollout_gpus_per_actor)

    @property
    def train_actors(self) -> List[ActorPlacement]:
        return self._build_actors(self.train_bundles, self.config.num_train_gpus_per_actor)

    def _build_actors(self, bundles: List[int], stride: int) -> List[ActorPlacement]:
        actors: List[ActorPlacement] = []
        for rank in range(len(bundles) // stride):
            stripe = bundles[rank * stride : (rank + 1) * stride]
            gpu_ids = tuple(self.gpu_ids[b] for b in stripe) if self.gpu_ids else ()
            node_ip = self.node_ips[stripe[0]] if self.node_ips else ""
            actors.append(
                ActorPlacement(
                    rank=rank,
                    bundle_idx=stripe[0],
                    gpu_ids=gpu_ids,
                    node_ip=node_ip,
                )
            )
        return actors

    def destroy(self) -> None:
        try:
            ray.util.remove_placement_group(self.pg)
            logger.info("Placement group removed")
        except Exception as e:
            logger.warning("Failed to remove placement group: %s", e)


@ray.remote(num_cpus=1)
class _BundleProbe:
    """Per-bundle probe: reports node IP, physical GPU id, and a free port.

    ``port`` is picked on the bundle's node so the caller can use it as a
    distributed-rendezvous master port that is guaranteed free on that node
    at probe time.
    """

    def __init__(self):
        import os
        import socket

        from diffusionrl.ray.utils.net import get_free_port

        self.ip = socket.gethostbyname(socket.gethostname())
        # Prefer ray.get_gpu_ids() — works even when the bundle's runtime_env
        # disables CVD (e.g. NOSET). Returns strings; coerce to int.
        ray_ids = ray.get_gpu_ids()
        if ray_ids:
            self.gpu_ids = [int(x) for x in ray_ids]
        else:
            cv = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            self.gpu_ids = [int(x) for x in cv.split(",")] if cv else []
        self.port = int(get_free_port(start_port=29500))

    def info(self) -> Tuple[str, List[int], int]:
        return self.ip, self.gpu_ids, self.port


def _probe_pg(pg: PlacementGroup, n: int) -> List[Tuple[str, int, int]]:
    # Claim num_gpus=1 per probe so Ray actually reserves the bundle's GPU
    # and ray.get_gpu_ids() / CVD are populated. Probes are killed before
    # rollout actors schedule, so the 1-GPU claim is transient.
    probes = [
        _BundleProbe.options(
            num_gpus=1,
            scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=i),
        ).remote()
        for i in range(n)
    ]
    results = [ray.get(p.info.remote()) for p in probes]
    for p in probes:
        try:
            ray.kill(p)
        except Exception:
            pass
    return [(ip, gpu_ids[0] if gpu_ids else 0, port) for ip, gpu_ids, port in results]
