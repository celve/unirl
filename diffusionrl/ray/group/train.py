"""Training actor group: spawn + dispatch + control plane for the
``RolloutResp`` / Policy-stack path.

Spawns :class:`diffusionrl.ray.train_actor.TrainActor` handles and exposes
the control-plane surface (``train``, weight-sync group setup, ``save_model``,
``offload`` / ``onload``, ``apply_eval_ema``).

Construction is single-step: ``TrainActorGroup(cfg=cfg, placement=placement)``
validates the cfg shape, spawns Ray handles, syncs on ``health_check``, and
rebroadcasts the master endpoint via rank 0 — symmetric with
:class:`RolloutActorGroup`.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, List

import ray
from omegaconf import DictConfig
from ray.actor import ActorHandle
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from diffusionrl.ray.group.base import ActorGroup
from diffusionrl.training.sharding import shard_resp_per_actor
from diffusionrl.training.validate import _validate_cfg_for_train
from diffusionrl.types.rollout_resp import RolloutResp

if TYPE_CHECKING:
    from diffusionrl.ray.placement import Placement
    from diffusionrl.training import TrackMiniBatchResult

logger = logging.getLogger(__name__)

_DETERMINISM_ENV_VARS = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}


class TrainActorGroup(ActorGroup):
    """Train actor group for the Policy-stack ``RolloutResp`` path.

    Inherits handle storage and ``scatter_gather(_async)`` from
    :class:`ActorGroup`. Adds the typed control-plane surface (weight-sync
    group setup, save/load/offload, eval-EMA swap) plus the data-plane
    ``train`` that does a balanced split of a ``RolloutResp`` across DP
    ranks before dispatching one shard per actor.
    """

    def __init__(self, *, cfg: DictConfig, placement: "Placement") -> None:
        from diffusionrl.config.instantiate import materialize
        from diffusionrl.ray.train_actor import TrainActor

        _validate_cfg_for_train(cfg)

        topology = materialize(cfg.training.topology)
        actors = placement.train_actors
        if not actors:
            raise RuntimeError("placement has no train actors")
        if topology.actor_count is not None and int(topology.actor_count) != len(actors):
            raise RuntimeError(
                f"topology.actor_count={topology.actor_count} does not match placement.num_train_actors={len(actors)}"
            )

        master_addr, master_port = placement.train_master
        seed = int(cfg.run.seed)
        # Forward NCCL / network-transport env from the launch (driver) environment
        # into every actor's runtime_env, so e.g. NCCL_IB_DISABLE / NCCL_SOCKET_IFNAME
        # set once on the head reach all actors on every node. Ray does NOT propagate
        # a worker node's shell env to that node's actors, but runtime_env does.
        env_vars: Dict[str, str] = dict(_DETERMINISM_ENV_VARS)
        env_vars.update({k: v for k, v in os.environ.items() if k.startswith("NCCL_")})
        runtime_env: Dict[str, Any] = {"env_vars": env_vars}

        # In colocate mode the train actor shares the bundle with a rollout
        # actor that already claimed ``colocate_gpu_fraction`` of the GPU;
        # asking for a full 1.0 here over-subscribes and the train actor
        # hangs in scheduling.
        ray_num_gpus = float(placement.config.colocate_gpu_fraction) if placement.config.colocate else 1.0

        logger.info(
            "Creating %d TrainActor handles (ray_num_gpus=%s, colocate=%s)",
            len(actors),
            ray_num_gpus,
            placement.config.colocate,
        )

        train_actors: List[ActorHandle] = []
        for actor in actors:
            strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement.pg,
                placement_group_bundle_index=actor.bundle_idx,
                placement_group_capture_child_tasks=False,
            )
            train_actors.append(
                TrainActor.options(
                    num_gpus=ray_num_gpus,
                    num_cpus=1.0,
                    scheduling_strategy=strategy,
                    runtime_env=runtime_env,
                ).remote(
                    cfg=cfg,
                    world_size=len(actors),
                    rank=actor.rank,
                    master_addr=master_addr,
                    master_port=master_port,
                    seed=seed,
                )
            )

        # Force eager init so any actor __init__ errors surface here, not
        # at the first downstream RPC.
        ray.get([a.health_check.remote() for a in train_actors])

        # Multinode-correct master rebroadcast: pull rank 0's actual node
        # IP and a free port (the placement-supplied master may be the
        # driver's IP; port 10000 is RDMA-blocked cross-node on H20 pods).
        rank0_info = ray.get(train_actors[0].get_master_info.remote(29500))
        master_addr = str(rank0_info["master_addr"])
        master_port = int(rank0_info["master_port"])
        ray.get([a.set_master_info.remote(master_addr, master_port) for a in train_actors])

        super().__init__(train_actors)
        self.master_addr: str = master_addr
        self.master_port: int = master_port
        # Keep-local data plane: each actor trains on the rollout it produced and
        # cached (TrainActor.train_local); train() then skips the driver shard.
        self._keep_local: bool = bool(cfg.training.execution.get("keep_local", False))
        logger.info(
            "TrainActorGroup ready: %d actors (master=%s:%d)",
            len(train_actors),
            master_addr,
            master_port,
        )

    # ------------------------------------------------------------------
    # Data plane: train (balanced shard split)
    # ------------------------------------------------------------------

    def train(self, rollout_id: int, training_resp: RolloutResp) -> "Dict[str, List[TrackMiniBatchResult]]":
        """Shard a multi-track ``RolloutResp`` across DP ranks and dispatch one shard per actor.

        Sharding strategy (see :func:`diffusionrl.training.sharding.shard_resp_per_actor`):

        - Identify the unique root track (``parent_track is None``).
        - Balanced-split root indices across actors using floor/remainder
          (same allocation as the legacy single-track path).
        - Per child track, build per-actor index sets by walking the
          lineage tree downward — every leaf sample on actor A has its
          full ancestor chain on actor A. Preserves the ``(n_groups,
          branch)`` reshape invariance that ``RolloutTrack.compute_advantages``
          relies on per actor.

        Each actor receives a multi-track ``RolloutResp`` shard via
        ``actor.train.remote(rollout_id, shard)`` and returns
        ``Dict[str, TrackMiniBatchResult]`` (one per track). This function
        transposes the per-actor list into ``Dict[track_name, List[result]]``
        (per-track, per-actor) so downstream metric aggregation can iterate
        cleanly per track.

        Keep-local mode (``cfg.training.execution.keep_local``, direct sampling):
        skip sharding entirely — each actor trains on the rollout it produced and
        cached locally via ``TrainActor.train_local``. ``training_resp`` here is
        only the light per-track metadata the driver logs; its heavy payload is
        never read.

        Fail-fast on:

        - Multi-root resps (ambiguous sharding choice).
        - Root track's ``batch_size < num_train_actors`` (FSDP collectives
          require all ranks to receive at least one sample).
        """
        n = self.num_actors
        if self._keep_local:
            # Skip the driver shard/scatter — each actor trains the rollout it
            # produced and cached locally (producer == consumer, direct sampling).
            refs = [a.train_local.remote(rollout_id) for a in self._actors]
        elif n == 1:
            refs = [a.train.remote(rollout_id, training_resp) for a in self._actors]
        else:
            shards = shard_resp_per_actor(training_resp, n)
            refs = [actor.train.remote(rollout_id, shard) for actor, shard in zip(self._actors, shards)]
        per_actor: List[Dict[str, Any]] = ray.get(refs)

        # Transpose: List[Dict[track_name, TrackMiniBatchResult]] →
        # Dict[track_name, List[TrackMiniBatchResult]] (per-track, per-actor).
        if not per_actor:
            return {}
        track_names = list(per_actor[0].keys())
        return {name: [actor_result[name] for actor_result in per_actor] for name in track_names}

    # ------------------------------------------------------------------
    # Control plane
    # ------------------------------------------------------------------

    def setup_weight_sync(
        self,
        *,
        sync_cfg,
        placement_cfg,
        rollout_runtime,
        param_name_prefix: str = "",
        packed_modules: dict | None = None,
        track_sync_specs: dict | None = None,
    ) -> None:
        ray.get(
            [
                a.setup_weight_sync.remote(
                    sync_cfg=sync_cfg,
                    placement_cfg=placement_cfg,
                    rollout_runtime=rollout_runtime,
                    param_name_prefix=param_name_prefix,
                    packed_modules=packed_modules,
                    track_sync_specs=track_sync_specs,
                )
                for a in self._actors
            ]
        )

    def sync_weights_to_rollout(self) -> None:
        ray.get([a.sync_weights_to_rollout.remote() for a in self._actors])

    def teardown_weight_sync(self) -> None:
        ray.get([a.teardown_weight_sync.remote() for a in self._actors])

    def save_model(self, path: str) -> None:
        ray.get([a.save_model.remote(path) for a in self._actors])

    def load_checkpoint(self, path: str) -> None:
        ray.get([a.load_checkpoint.remote(path) for a in self._actors])

    def offload(self) -> None:
        ray.get([a.offload.remote() for a in self._actors])

    def onload(self) -> None:
        ray.get([a.onload.remote() for a in self._actors])

    def clear_memory(self) -> None:
        ray.get([a.clear_memory.remote() for a in self._actors])

    def apply_eval_ema(self) -> None:
        ray.get([a.apply_eval_ema.remote() for a in self._actors])

    def restore_from_eval(self) -> None:
        ray.get([a.restore_from_eval.remote() for a in self._actors])

    @contextmanager
    def use_eval_ema(self):
        """Swap eval EMA weights in for the duration of a sampling/eval block."""
        self.apply_eval_ema()
        try:
            yield
        finally:
            self.restore_from_eval()


__all__ = ["TrainActorGroup"]
