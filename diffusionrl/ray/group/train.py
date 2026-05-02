"""Training actor group: spawn + dispatch + control plane in one class."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import ray
from omegaconf import DictConfig
from ray.actor import ActorHandle
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from diffusionrl.ray.group.base import ActorGroup
from diffusionrl.rollout.plan import RolloutPlan
from diffusionrl.transfer.buffer import BufferHandle
from diffusionrl.types.request import RolloutRequest
from diffusionrl.types.response import RolloutResponse

if TYPE_CHECKING:
    from diffusionrl.ray.placement import Placement

logger = logging.getLogger(__name__)


def _backend_name_from_cfg(cfg: DictConfig) -> str:
    """Extract the short backend identifier (e.g. ``"fsdp"``) from cfg.

    Mirrors what ``train.py`` previously derived on the driver: strip the
    dotted-path prefix off ``cfg.training.backend._target_``, drop a
    trailing ``Backend`` suffix, lowercase.
    """
    backend_target = str(cfg.training.backend._target_)
    backend_cls_name = backend_target.rsplit(".", 1)[-1]
    return (
        backend_cls_name[: -len("Backend")].lower()
        if backend_cls_name.lower().endswith("backend")
        else backend_cls_name.lower()
    )


class TrainActorGroup(ActorGroup):
    """Training actor group: spawn + dispatch + control plane in one class.

    Inherits handle storage and ``scatter_gather(_async)`` from
    :class:`ActorGroup`. Adds training-specific spawn (``bootstrap``)
    plus the typed control-plane surface (``update_weights``, ``offload``,
    etc.). Broadcast and rank-0 dispatch are inline list comprehensions
    rather than stringly-typed helpers.

    Custom backend actor classes (e.g. Megatron's
    ``requires_custom_actor_class=True``) are intentionally not supported
    here.
    """

    def __init__(self, actors: Sequence[ActorHandle], *, rollout_plan: RolloutPlan) -> None:
        super().__init__(actors)
        self.rollout_plan = rollout_plan
        self._train_backend_info_cache: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Spawn
    # ------------------------------------------------------------------

    @classmethod
    def bootstrap(
        cls,
        *,
        cfg: DictConfig,
        placement: "Placement",
        colocate: bool = False,
        colocate_gpu_fraction: float = 1.0,
        num_gpus_per_actor: Optional[int] = None,
        runtime_env: Optional[Dict[str, Any]] = None,
    ) -> Tuple["TrainActorGroup", str, int]:
        """Spawn ``TrainActor`` handles directly from a Hydra ``DictConfig``.

        The composed ``cfg`` is shipped to each actor via ``ConfigActor``;
        the actor materializes registered sections on its own via
        ``build()`` / ``materialize`` at the read site. All per-run
        derivations (topology for actor-count cross-check, backend name +
        capabilities for the custom-actor-class preflight) are computed
        here from cfg — no driver wiring required.

        Returns ``(group, master_addr, master_port)``; master values reflect
        the rank-0 rebroadcast.
        """
        from diffusionrl.config.instantiate import materialize
        from diffusionrl.ray.train_actor import TrainActor
        from diffusionrl.training.types import resolve_train_backend_capabilities

        if cfg.model.get("_target_") is None:
            raise ValueError("cfg.model must carry _target_ (use a registered model preset)")
        if cfg.algorithm.get("_target_") is None:
            raise ValueError("cfg.algorithm must carry _target_ (use a registered algorithm preset)")
        if cfg.training.backend.get("_target_") is None:
            raise ValueError("cfg.training.backend must carry _target_ (use a registered training/backend preset)")

        backend_name = _backend_name_from_cfg(cfg)
        backend_capabilities = resolve_train_backend_capabilities(backend_name)
        caps = backend_capabilities.as_dict()
        if bool(caps.get("requires_custom_actor_class", False)):
            raise NotImplementedError(
                "TrainActorGroup.bootstrap does not support backends that "
                f"require a custom actor class. backend={backend_name!r} "
                "declared requires_custom_actor_class=True."
            )

        topology = materialize(cfg.training.topology)
        actors = placement.train_actors
        if not actors:
            raise RuntimeError("placement has no train actors")
        if topology.actor_count is not None and int(topology.actor_count) != len(actors):
            raise RuntimeError(
                f"topology.actor_count={topology.actor_count} does not match placement.num_train_actors={len(actors)}"
            )

        master_addr, master_port = placement.train_master

        if num_gpus_per_actor is not None:
            ray_num_gpus = float(num_gpus_per_actor)
        else:
            ray_num_gpus = float(colocate_gpu_fraction) if colocate else 1.0
        if colocate:
            logger.info("Colocate mode: TrainActors using %s GPU each", ray_num_gpus)

        seed = int(cfg.run.seed)

        # Determinism env vars propagated to every TrainActor process before
        # Python imports torch. Required for cuBLAS deterministic workspace;
        # cudnn.deterministic / torch.use_deterministic_algorithms are set by
        # set_seed() in TrainActor.__init__.
        _DETERMINISM_ENV_VARS = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
        base_runtime_env: Dict[str, Any] = dict(runtime_env) if runtime_env else {}
        base_env_vars = dict(base_runtime_env.get("env_vars") or {})
        base_env_vars.update(_DETERMINISM_ENV_VARS)
        base_runtime_env["env_vars"] = base_env_vars

        logger.info("Creating %d TrainActor handles (ray_num_gpus=%s)", len(actors), ray_num_gpus)

        train_actors: List[ActorHandle] = []
        for actor in actors:
            strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement.pg,
                placement_group_bundle_index=actor.bundle_idx,
                placement_group_capture_child_tasks=False,
            )
            options: Dict[str, Any] = {
                "num_gpus": ray_num_gpus,
                "num_cpus": 1.0,
                "scheduling_strategy": strategy,
                "runtime_env": base_runtime_env,
            }
            train_actors.append(
                TrainActor.options(**options).remote(
                    cfg=cfg,
                    world_size=len(actors),
                    rank=actor.rank,
                    master_addr=master_addr,
                    master_port=master_port,
                    seed=seed,
                )
            )

        # Force eager init to complete by round-tripping a no-op call.
        ray.get([a.health_check.remote() for a in train_actors])

        # Multinode fix: use rank 0's actual node IP (not the driver's) and a
        # port >= 29500 (port 10000 is blocked cross-node on H20 RDMA pods).
        rank0_info = ray.get(train_actors[0].get_master_info.remote(29500))
        master_addr = str(rank0_info["master_addr"])
        master_port = int(rank0_info["master_port"])
        ray.get([a.set_master_info.remote(master_addr, master_port) for a in train_actors])

        group = cls(train_actors, rollout_plan=materialize(cfg.rollout.plan))
        logger.info(
            "TrainActorGroup ready: %d actors (master=%s:%d)",
            len(train_actors),
            master_addr,
            master_port,
        )
        return group, master_addr, master_port

    # ------------------------------------------------------------------
    # Data plane: train (sliced dispatch)
    # ------------------------------------------------------------------

    def train(
        self,
        rollout_id: int,
        training_batch: Any,
    ) -> List[Any]:
        """Slice the training batch across DP ranks and dispatch one shard per actor.

        Per-actor shard sizes use a balanced split: every actor receives at
        least ``floor(batch_size / num_actors)`` samples, and the first
        ``batch_size % num_actors`` actors each receive one extra sample.
        Uneven shard sizes are intentional — divisibility is not required, only
        ``batch_size >= num_actors`` so every FSDP2 rank gets at least one
        sample (collectives deadlock if any rank is skipped).

        The per-rank slice lives in the dispatch layer here rather than
        inside the actor, so ``TrainActor.train`` only ever sees its own
        shard.

        Returns one ``BatchResult`` per actor (in actor-rank order).
        """
        n = self.num_actors
        if n == 1:
            # Broadcast the full batch to avoid the unnecessary tensor.clone()
            # that batch.slice(0, batch_size) would do (Batched.slice clones
            # tensor fields, not a view).
            refs = [a.train.remote(rollout_id, training_batch) for a in self._actors]
            return ray.get(refs)

        batch_size = int(training_batch.batch_size)
        if batch_size < n:
            raise ValueError(
                f"TrainingBatch.batch_size ({batch_size}) is smaller than "
                f"num_train_actors ({n}); every train actor must receive at "
                "least one sample (FSDP2 collectives require all ranks to "
                "participate)."
            )

        base = batch_size // n
        remainder = batch_size % n
        refs: List[ray.ObjectRef] = []
        cursor = 0
        for i, actor in enumerate(self._actors):
            shard_size = base + (1 if i < remainder else 0)
            refs.append(actor.train.remote(rollout_id, training_batch.slice(cursor, cursor + shard_size)))
            cursor += shard_size
        return ray.get(refs)

    # ------------------------------------------------------------------
    # Data plane: generate (training-actor-sampling-mode)
    # ------------------------------------------------------------------
    # Exposes a ``generate*`` surface so future training-actor-sampling
    # workloads can run generation on the training actors directly.
    # ``train.py`` itself does not call these today.

    def _build_shards(self, request: RolloutRequest) -> List[Optional[RolloutRequest]]:
        return self.rollout_plan.shard(
            request,
            num_actors=self.num_actors,
            pad_to_actor_count=True,
        )

    def async_generate(self, request: RolloutRequest) -> List[ray.ObjectRef]:
        return self.scatter_gather_async("generate", self._build_shards(request))

    def generate(self, request: RolloutRequest) -> List[RolloutResponse]:
        shards = self._build_shards(request)
        outputs = self.scatter_gather("generate", shards)
        return self.rollout_plan.trim(outputs, request=request)

    def generate_buffered(self, request: RolloutRequest) -> List[BufferHandle]:
        nested = self.scatter_gather("generate_buffered", self._build_shards(request))
        return [handle for actor_handles in nested for handle in actor_handles]

    # ------------------------------------------------------------------
    # Control plane
    # ------------------------------------------------------------------

    def update_weights(self) -> None:
        ray.get([a.update_weights.remote() for a in self._actors])

    def setup_weight_sync(
        self,
        *,
        sync_cfg,
        placement_cfg,
        rollout_runtime,
    ) -> None:
        """Fan out weight-sync handler setup to all training actors."""
        ray.get(
            [
                a.setup_weight_sync.remote(
                    sync_cfg=sync_cfg,
                    placement_cfg=placement_cfg,
                    rollout_runtime=rollout_runtime,
                )
                for a in self._actors
            ]
        )

    def sync_weights_to_rollout(self) -> None:
        """Fan out handler-based weight sync to all training actors."""
        ray.get([a.sync_weights_to_rollout.remote() for a in self._actors])

    def teardown_weight_sync(self) -> None:
        """Fan out weight-sync handler teardown to all training actors."""
        ray.get([a.teardown_weight_sync.remote() for a in self._actors])

    def get_train_backend_info(self, force_refresh: bool = False) -> Dict[str, Any]:
        if self._train_backend_info_cache is not None and not force_refresh:
            return dict(self._train_backend_info_cache)
        info = ray.get(self._actors[0].get_train_backend_info.remote())
        if isinstance(info, dict):
            self._train_backend_info_cache = dict(info)
            return dict(info)
        return {}

    def export_weights_to_path(
        self,
        checkpoint_path: str,
        *,
        export_format: str = "state_dict",
    ) -> str:
        refs = [a.export_weights_to_path.remote(checkpoint_path, export_format=export_format) for a in self._actors]
        if len(refs) > 1:
            ray.get(refs[1:])
        if refs:
            ray.get(refs[0])
        return checkpoint_path

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
