"""Training actor group: spawn + dispatch + control plane in one class."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import ray
from ray.actor import ActorHandle
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from diffusionrl.ray.generate_sharding import (
    build_generate_shard_plan,
    trim_generate_outputs,
)
from diffusionrl.ray.group_base import ActorHandleGroup
from diffusionrl.transfer.buffer import BufferHandle
from diffusionrl.types.request import RolloutRequest
from diffusionrl.types.response import RolloutResponse

logger = logging.getLogger(__name__)


class TrainActorGroup:
    """Training actor group: spawn + dispatch + control plane in one class.

    Used by ``diffusionrl.train``. Replaces a previous two-layer split
    where a control-plane ``GroupRuntime`` wrapped a data-plane actor
    group via ``from_group``. This class composes ``ActorHandleGroup``
    directly — it does NOT inherit from ``ActorGroup`` /
    ``PlacementGroupActorPool`` because the bootstrap flow does its own
    inline spawn (no auto-spawn-on-init).

    Custom backend actor classes (e.g. Megatron's
    ``requires_custom_actor_class=True``) are intentionally not supported
    here.
    """

    def __init__(self, handle: ActorHandleGroup):
        self._handle = handle.snapshot()
        self.num_actors = int(self._handle.num_actors)
        self._expected_global_batch_size_cache: Optional[int] = None
        self._train_backend_info_cache: Optional[Dict[str, Any]] = None

    @property
    def handle(self) -> ActorHandleGroup:
        """The underlying actor handle group (for dispatch primitives)."""
        return self._handle

    def get_actors(self) -> List[ActorHandle]:
        """Return the raw actor handles."""
        return self._handle.get_actors()

    def dispose(self) -> None:
        """Kill all managed actors. Safe to call multiple times."""
        for actor in self._handle.get_actors():
            try:
                ray.kill(actor)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Spawn
    # ------------------------------------------------------------------

    @classmethod
    def bootstrap(
        cls,
        *,
        launch_config: Any,
        training_pgs: Tuple[Any, List[int], Any],
    ) -> Tuple["TrainActorGroup", str, int]:
        """Spawn ``TrainActor`` handles eagerly and wrap in this group.

        Picks the distributed rendezvous addr/port, constructs
        ``num_train_actors`` ``TrainActor`` handles with eager-init kwargs
        built from the resolved ``TrainingLaunch``, and round-trips a
        ``health_check`` to force eager initialization to complete.

        Returns ``(group, master_addr, master_port)``. Use ``group.get_actors()``
        for raw handles, ``group.dispose()`` for teardown.
        """
        from diffusionrl.config.validation import validate_training_actor_init_config
        from diffusionrl.ray.actor_config import build_train_actor_init_kwargs
        from diffusionrl.ray.placement_group import InfoActor
        from diffusionrl.ray.train_actor import TrainActor
        from diffusionrl.ray.utils.net import get_free_port

        training_pg, training_bundle_indices, _training_gpu_ids = training_pgs

        training_launch = launch_config.training
        actor_init_config = deepcopy(training_launch.actor_init_config)
        validate_training_actor_init_config(actor_init_config)

        caps = training_launch.backend_capabilities.as_dict() if training_launch.backend_capabilities else {}
        if bool(caps.get("requires_custom_actor_class", False)):
            raise NotImplementedError(
                "TrainActorGroup.bootstrap does not support backends that "
                "require a custom actor class. "
                f"backend={training_launch.backend_name!r} declared "
                "requires_custom_actor_class=True; custom-actor backends "
                "(Megatron etc.) are not wired up."
            )

        topology = training_launch.topology
        num_train_actors = int(topology.actor_count) if topology.actor_count is not None else 0
        if num_train_actors < 1:
            raise RuntimeError(
                f"Resolved training topology must declare actor_count >= 1, got {topology.actor_count!r}."
            )

        # Pick distributed rendezvous (addr, port) on the driver before any
        # TrainActor constructor runs. We briefly spawn an InfoActor on the
        # first training bundle to read that node's IP.
        info_actor = InfoActor.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=training_pg,
                placement_group_bundle_index=training_bundle_indices[0],
            ),
        ).remote()
        try:
            node_ip, _gpu_ids = ray.get(info_actor.get_info.remote())
        finally:
            try:
                ray.kill(info_actor)
            except Exception:
                pass
        master_addr = str(node_ip)
        master_port = int(get_free_port(start_port=29500))

        launch_spec = training_launch.launch_spec
        if launch_spec.num_gpus_per_actor is not None:
            ray_num_gpus = float(launch_spec.num_gpus_per_actor)
        else:
            ray_num_gpus = float(training_launch.colocate_gpu_fraction) if training_launch.colocate else 1.0

        if training_launch.colocate:
            logger.info("Colocate mode: TrainActors using %s GPU each", ray_num_gpus)
        logger.info("Creating %d TrainActor handles", num_train_actors)

        # Determinism env vars propagated to every TrainActor process before
        # Python imports torch. Required for cuBLAS deterministic workspace;
        # cudnn.deterministic and torch.use_deterministic_algorithms are set
        # by set_seed() in TrainActor.__init__.
        _DETERMINISM_ENV_VARS = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}

        train_actors: List[ActorHandle] = []
        for rank in range(num_train_actors):
            ctor_kwargs = build_train_actor_init_kwargs(
                training_launch=training_launch,
                world_size=num_train_actors,
                rank=rank,
                master_addr=master_addr,
                master_port=master_port,
                sampling_config=getattr(launch_config, "sampling_spec", None),
            )
            bundle_idx = training_bundle_indices[rank]
            strategy = PlacementGroupSchedulingStrategy(
                placement_group=training_pg,
                placement_group_bundle_index=bundle_idx,
                placement_group_capture_child_tasks=False,
            )
            # Merge determinism env vars into runtime_env (preserve any
            # launch-spec-provided runtime_env env vars).
            base_runtime_env: Dict[str, Any] = dict(launch_spec.runtime_env) if launch_spec.runtime_env else {}
            base_env_vars = dict(base_runtime_env.get("env_vars") or {})
            base_env_vars.update(_DETERMINISM_ENV_VARS)
            base_runtime_env["env_vars"] = base_env_vars

            options: Dict[str, Any] = {
                "num_gpus": ray_num_gpus,
                "num_cpus": 1.0,
                "scheduling_strategy": strategy,
                "runtime_env": base_runtime_env,
            }
            train_actors.append(TrainActor.options(**options).remote(**ctor_kwargs))

        # Force eager init to complete by round-tripping a no-op call.
        ray.get([a.health_check.remote() for a in train_actors])

        # Multinode fix: use rank 0's actual node IP (not the driver's) and a
        # port >= 29500 (port 10000 is blocked cross-node on H20 RDMA pods).
        rank0_info = ray.get(train_actors[0].get_master_info.remote(29500))
        master_addr = str(rank0_info["master_addr"])
        master_port = int(rank0_info["master_port"])
        ray.get([a.set_master_info.remote(master_addr, master_port) for a in train_actors])

        handle_group = ActorHandleGroup(train_actors, num_actors=num_train_actors)
        group = cls(handle_group)
        logger.info(
            "TrainActorGroup ready: %d actors (master=%s:%d)",
            num_train_actors,
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
        n = int(self.num_actors)
        if n == 1:
            # No-op slice; broadcast the full batch via call_all to avoid the
            # unnecessary tensor.clone() that batch.slice(0, batch_size) would
            # do (Batched.slice clones tensor fields, not a view).
            refs = self._handle.call_all_async("train", rollout_id, training_batch)
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
        per_actor_args: List[Any] = []
        cursor = 0
        for i in range(n):
            shard_size = base + (1 if i < remainder else 0)
            per_actor_args.append((rollout_id, training_batch.slice(cursor, cursor + shard_size)))
            cursor += shard_size

        refs = self._handle.call_per_actor_async("train", per_actor_args=per_actor_args)
        return ray.get(refs)

    # ------------------------------------------------------------------
    # Data plane: generate (training-actor-sampling-mode)
    # ------------------------------------------------------------------
    # Exposes a ``generate*`` surface so future training-actor-sampling
    # workloads can run generation on the training actors directly.
    # ``train.py`` itself does not call these today.

    def _build_generate_plan(self, request: RolloutRequest):
        return build_generate_shard_plan(
            request,
            num_actors=self.num_actors,
            pad_to_actor_count=True,
        )

    def async_generate(self, request: RolloutRequest) -> List[ray.ObjectRef]:
        plan = self._build_generate_plan(request)
        return self._handle.scatter_gather_async("generate", plan.shards)

    def generate(self, request: RolloutRequest) -> List[RolloutResponse]:
        plan = self._build_generate_plan(request)
        outputs = ray.get(self._handle.scatter_gather_async("generate", plan.shards))
        return trim_generate_outputs(outputs, plan=plan)

    def generate_buffered(self, request: RolloutRequest) -> List[BufferHandle]:
        plan = self._build_generate_plan(request)
        nested = ray.get(self._handle.scatter_gather_async("generate_buffered", plan.shards))
        return [handle for actor_handles in nested for handle in actor_handles]

    # ------------------------------------------------------------------
    # Control plane
    # ------------------------------------------------------------------

    def update_weights(self) -> None:
        self._handle.call_all("update_weights")

    def setup_weight_sync(self, config: dict) -> None:
        """Fan out weight-sync handler setup to all training actors."""
        self._handle.call_all("setup_weight_sync", config)

    def sync_weights_to_rollout(self) -> None:
        """Fan out handler-based weight sync to all training actors."""
        self._handle.call_all("sync_weights_to_rollout")

    def teardown_weight_sync(self) -> None:
        """Fan out weight-sync handler teardown to all training actors."""
        self._handle.call_all("teardown_weight_sync")

    def get_train_backend_info(self, force_refresh: bool = False) -> Dict[str, Any]:
        if self._train_backend_info_cache is not None and not force_refresh:
            return dict(self._train_backend_info_cache)
        info = self._handle.call_rank0("get_train_backend_info")
        if isinstance(info, dict):
            self._train_backend_info_cache = dict(info)
            return dict(info)
        return {}

    def get_expected_global_batch_size(self, force_refresh: bool = False) -> int:
        if self._expected_global_batch_size_cache is not None and not force_refresh:
            return int(self._expected_global_batch_size_cache)
        expected_global_batch_size = self._handle.call_rank0("get_expected_global_batch_size")
        try:
            resolved = int(expected_global_batch_size)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid expected_global_batch_size payload: {expected_global_batch_size!r}") from exc
        self._expected_global_batch_size_cache = resolved
        return resolved

    def export_weights_to_path(
        self,
        checkpoint_path: str,
        *,
        export_format: str = "state_dict",
    ) -> str:
        refs = self._handle.call_all_async(
            "export_weights_to_path",
            checkpoint_path,
            export_format=export_format,
        )
        if len(refs) > 1:
            ray.get(refs[1:])
        if refs:
            ray.get(refs[0])
        return checkpoint_path

    def save_model(self, path: str) -> None:
        self._handle.call_all("save_model", path)

    def load_checkpoint(self, path: str) -> None:
        self._handle.call_all("load_checkpoint", path)

    def offload(self) -> None:
        self._handle.call_all("offload")

    def onload(self) -> None:
        self._handle.call_all("onload")

    def clear_memory(self) -> None:
        self._handle.call_all("clear_memory")

    def apply_eval_ema(self) -> None:
        self._handle.call_all("apply_eval_ema")

    def restore_from_eval(self) -> None:
        self._handle.call_all("restore_from_eval")

    @contextmanager
    def use_eval_ema(self):
        """Swap eval EMA weights in for the duration of a sampling/eval block."""
        self.apply_eval_ema()
        try:
            yield
        finally:
            self.restore_from_eval()


__all__ = ["TrainActorGroup"]
