"""New training actor group: spawn + dispatch + control plane for the
``RolloutResp`` / Policy-stack path.

Sibling of :class:`diffusionrl.ray.group.train.TrainActorGroup` (legacy types
+ ``TrainBackend``). Spawns :class:`diffusionrl.ray.new_train_actor.NewTrainActor`
handles and exposes the same control-plane surface (``train``, weight-sync
group setup, ``save_model``, ``offload`` / ``onload``, ``apply_eval_ema``).

Construction is single-step: ``NewTrainActorGroup(cfg=cfg, placement=placement)``
validates the cfg shape, spawns Ray handles, syncs on ``health_check``, and
rebroadcasts the master endpoint via rank 0 — symmetric with
:class:`NewRolloutActorGroup`. The legacy ``bootstrap`` classmethod factory is
intentionally absent here.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, List

import ray
from omegaconf import DictConfig
from ray.actor import ActorHandle
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from diffusionrl.ray.group.base import ActorGroup
from diffusionrl.types.rollout_resp import RolloutResp

if TYPE_CHECKING:
    from diffusionrl.ray.placement import Placement
    from diffusionrl.training_new import TrainOptimizerStepResult

logger = logging.getLogger(__name__)

_DETERMINISM_ENV_VARS = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}


def _validate_cfg_for_new_train(cfg: DictConfig) -> None:
    """Fail-fast preflight: every leaf the new train actor reads must be present."""
    if cfg.model.get("_target_") is None:
        raise ValueError("cfg.model must carry _target_ (use a registered pipeline preset)")
    algorithms = cfg.get("algorithms")
    if algorithms is None or len(algorithms) == 0:
        raise ValueError(
            "cfg.algorithms (plural) must be a non-empty slot-keyed dict of "
            "StageAlgorithm presets. The new train actor reads cfg.algorithms — "
            "the singular cfg.algorithm is the legacy / driver-side surface."
        )
    for slot, alg_node in algorithms.items():
        if alg_node.get("_target_") is None:
            raise ValueError(f"cfg.algorithms.{slot} must carry _target_ (use a registered StageAlgorithm preset)")
    policies = cfg.training.get("policies")
    if policies is None or len(policies) == 0:
        raise ValueError(
            "cfg.training.policies must be a non-empty list of policy presets "
            "(e.g. [LoRAPolicyConfig, FSDPPolicyConfig, EMAPolicyConfig])"
        )
    for idx, node in enumerate(policies):
        if node.get("_target_") is None:
            raise ValueError(
                f"cfg.training.policies[{idx}] must carry _target_ (use a registered training/policy preset)"
            )
    if cfg.training.get("policy_source") is None:
        raise ValueError(
            "cfg.training.policy_source must be set (the slot name on the "
            'pipeline whose Stage anchors the Policy stack, e.g. "diffusion")'
        )


class NewTrainActorGroup(ActorGroup):
    """Train actor group for the Policy-stack ``RolloutResp`` path.

    Inherits handle storage and ``scatter_gather(_async)`` from
    :class:`ActorGroup`. Adds the typed control-plane surface (weight-sync
    group setup, save/load/offload, eval-EMA swap) plus the data-plane
    ``train`` that does a balanced split of a ``RolloutResp`` across DP
    ranks before dispatching one shard per actor.
    """

    def __init__(self, *, cfg: DictConfig, placement: "Placement") -> None:
        from diffusionrl.config.instantiate import materialize
        from diffusionrl.ray.new_train_actor import NewTrainActor

        _validate_cfg_for_new_train(cfg)

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
        runtime_env: Dict[str, Any] = {"env_vars": dict(_DETERMINISM_ENV_VARS)}

        # In colocate mode the train actor shares the bundle with a rollout
        # actor that already claimed ``colocate_gpu_fraction`` of the GPU;
        # asking for a full 1.0 here over-subscribes and the train actor
        # hangs in scheduling. Matches the legacy ``TrainActorGroup`` path
        # (group/train.py:123).
        ray_num_gpus = float(placement.config.colocate_gpu_fraction) if placement.config.colocate else 1.0

        logger.info(
            "Creating %d NewTrainActor handles (ray_num_gpus=%s, colocate=%s)",
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
                NewTrainActor.options(
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
        logger.info(
            "NewTrainActorGroup ready: %d actors (master=%s:%d)",
            len(train_actors),
            master_addr,
            master_port,
        )

    # ------------------------------------------------------------------
    # Data plane: train (balanced shard split)
    # ------------------------------------------------------------------

    def train(self, rollout_id: int, training_resp: RolloutResp) -> "List[List[TrainOptimizerStepResult]]":
        """Slice the RolloutResp across DP ranks and dispatch one shard per actor.

        Per-actor shard sizes use a balanced split: every actor receives at
        least ``floor(batch_size / num_actors)`` samples, and the first
        ``batch_size % num_actors`` actors each receive one extra sample.
        ``batch_size >= num_actors`` is required so every FSDP rank gets at
        least one sample (collectives deadlock if any rank is skipped).

        Returns ``List[List[TrainOptimizerStepResult]]`` — outer list is per
        optimizer step (``num_updates_per_batch`` entries), inner list is
        per actor. This lets the driver log metrics per optimizer step.
        """
        n = self.num_actors
        if n == 1:
            refs = [a.train.remote(rollout_id, training_resp) for a in self._actors]
            per_actor: List[List[Any]] = ray.get(refs)  # each actor returns List[Result]
        else:
            batch_size = int(training_resp.batch_size)
            if batch_size < n:
                raise ValueError(
                    f"RolloutResp.batch_size ({batch_size}) is smaller than "
                    f"num_train_actors ({n}); every train actor must receive at "
                    "least one sample (FSDP collectives require all ranks to "
                    "participate)."
                )
            base = batch_size // n
            remainder = batch_size % n
            refs: List[ray.ObjectRef] = []
            cursor = 0
            for i, actor in enumerate(self._actors):
                shard_size = base + (1 if i < remainder else 0)
                refs.append(actor.train.remote(rollout_id, training_resp.slice(cursor, cursor + shard_size)))
                cursor += shard_size
            per_actor = ray.get(refs)  # List[List[Result]], per-actor × per-update

        # Transpose: per-actor × per-update → per-update × per-actor
        num_updates = len(per_actor[0]) if per_actor else 0
        return [[per_actor[a][u] for a in range(n)] for u in range(num_updates)]

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
    ) -> None:
        ray.get(
            [
                a.setup_weight_sync.remote(
                    sync_cfg=sync_cfg,
                    placement_cfg=placement_cfg,
                    rollout_runtime=rollout_runtime,
                    param_name_prefix=param_name_prefix,
                    packed_modules=packed_modules,
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


__all__ = ["NewTrainActorGroup"]
