"""Bucketed CUDA-IPC weight sync for vllm-omni rollout actors.

Trainer-side counterpart to
``diffusionrl.rollout.engine.vllm_omni.weight_sync.dit_extension.DiTWeightSyncExtension``
(and ``ar_extension.HI3ARWeightSyncExtension``). Mirrors
``UpdateWeightFromTensor`` (``distributed/weight_sync/tensor.py``) in
shape but uses verbatim bucketed CUDA-IPC over ZMQ rather than SGLang's
serialized-tensor RPC.

Per-stage fan-out: HI3 t2i has 2 stages (AR + DiT) each on 4 GPUs. The
trainer's FSDP shards across all 8 GPUs; each train rank ships the full
state dict twice (once per rollout stage) over per-stage sockets.

Registered as ``sync.ipc_bucketed`` so existing ``cfg.training.sync.*``
selection picks it up.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional, Tuple

import ray
import torch.distributed as dist

from diffusionrl.config.registration import register_config
from diffusionrl.config.require import require
from diffusionrl.distributed.weight_sync.base import BucketedUpdateWeight
from diffusionrl.rollout.engine.vllm_omni.weight_sync.bucketed_transfer import (
    BucketedWeightSender,
)
from diffusionrl.rollout.engine.vllm_omni.weight_sync.ipc_dispatch import (
    replica_rank_from_env,
    zmq_handle,
)


@register_config(
    group="sync",
    name="ipc_bucketed",
    target="diffusionrl.distributed.weight_sync.ipc.UpdateWeightFromIPC",
    expand=True,
)
@dataclass(frozen=True)
class IPCBucketedSyncConfig:
    """Push model weights to vllm-omni rollout engines via bucketed CUDA-IPC."""

    bucket_size: int = 2048
    bucket_size_mb: int = 2048
    flush_cache: bool = True
    use_shm: bool = False
    target_modules: Tuple[str, ...] = field(default_factory=lambda: ("transformer",))
    # Stage IDs to ship to. Default ``None`` (i.e. "all stages the engine
    # exposes") is interpreted on the rollout side; override here to scope
    # sync to e.g. only the DiT stage during early bring-up.
    stage_ids: Tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        require(self.bucket_size >= 1, f"bucket_size must be >= 1; got {self.bucket_size!r}")
        require(self.bucket_size_mb >= 1, f"bucket_size_mb must be >= 1; got {self.bucket_size_mb!r}")
        require(
            len(self.target_modules) >= 1,
            f"target_modules must not be empty; got {self.target_modules!r}",
        )


class UpdateWeightFromIPC(BucketedUpdateWeight):
    """Bucketed CUDA-IPC push to vllm-omni rollout engines.

    For each (replica × stage) pair we keep one ``BucketedWeightSender``
    primed: at the start of an ``update_weights`` call we tell every
    rollout actor to spawn matching receivers via
    ``actor.update_weights_from_ipc.remote(stage_ids=[...])`` (non-blocking
    Ray call), then run the bucketed bucket-by-bucket pump on this rank's
    sockets.
    """

    def __init__(
        self,
        *,
        model,
        rollout_runtime,
        placement_cfg,
        bucket_size: int = 2048,
        bucket_size_mb: int = 2048,
        flush_cache: bool = True,
        use_shm: bool = False,
        target_modules: Tuple[str, ...] = ("transformer",),
        stage_ids: Tuple[int, ...] = (),
        param_name_prefix: str = "",
    ) -> None:
        super().__init__(
            model=model,
            rollout_runtime=rollout_runtime,
            placement_cfg=placement_cfg,
            bucket_size=bucket_size,
            flush_cache=flush_cache,
            target_modules=target_modules,
            param_name_prefix=param_name_prefix,
        )
        self._bucket_size_mb = int(bucket_size_mb)
        self._use_shm = bool(use_shm)
        self._stage_ids = tuple(int(s) for s in stage_ids)
        self._rollout_actors: list = []
        self._this_rank: int = 0

    # ------------------------------------------------------------------
    # Connection setup
    # ------------------------------------------------------------------

    def connect_rollout_engines(self) -> None:
        """Discover rollout actors and cache each one's stage count.

        ``num_stages`` is cached here (one Ray RPC at setup time) rather
        than during ``update_weights`` — the latter fires
        ``actor.update_weights_from_ipc.remote(...)`` before iterating
        stages, which makes the actor busy waiting for senders. A
        follow-up ``actor.num_stages.remote()`` would queue behind that
        call and deadlock (actor blocked on no senders → num_stages
        never dispatched → sender never enumerates stages).
        """
        self._rollout_actors = list(self._rollout_runtime.get_rollout_actors())
        self._this_rank = int(dist.get_rank()) if dist.is_initialized() else 0
        self._actor_num_stages: list[int] = []
        self._actor_tp_per_stage: list[dict] = []
        for actor in self._rollout_actors:
            try:
                n = int(ray.get(actor.num_stages.remote()))
            except (AttributeError, ray.exceptions.RayError):
                n = 2
            self._actor_num_stages.append(n)
            # Cache TP size per stage to avoid sending to non-existent
            # receivers. HI3 dual-stage AR(TP=4) + DiT(TP=4) = 8 GPU but
            # FSDP train has 8 ranks. Without this, train ranks 4-7 try to
            # send to stage 0 (only 4 receivers) → ZMQ blocks → NCCL deadlock.
            try:
                tp_map = ray.get(actor.tp_per_stage.remote())
                self._actor_tp_per_stage.append({int(k): int(v) for k, v in tp_map.items()})
            except (AttributeError, ray.exceptions.RayError):
                fallback_tp = int(self._placement_cfg.num_rollout_gpus_per_actor)
                self._actor_tp_per_stage.append({sid: fallback_tp for sid in range(n)})

    # ------------------------------------------------------------------
    # Bucket pump
    # ------------------------------------------------------------------

    def update_weights(
        self,
        *,
        peft_config: Optional[dict] = None,
        base_sync_done: bool = False,
    ) -> None:
        """Override the base bucket loop so receivers are wired up once per
        full update (not once per bucket)."""
        if not self._rollout_actors:
            return
        # Resolve per-this-rank receivers BEFORE iterating buckets so the
        # remote receivers are already listening on the sockets when the
        # first bucket arrives.
        actor_idx = self._this_rank // max(1, int(self._placement_cfg.num_rollout_gpus_per_actor))
        has_destination = actor_idx < len(self._rollout_actors)
        if has_destination:
            actor = self._rollout_actors[actor_idx]
            actor_n_stages = self._actor_num_stages[actor_idx]
            local_rank = self._this_rank - actor_idx * int(self._placement_cfg.num_rollout_gpus_per_actor)

            stages = self._stage_ids if self._stage_ids else None
            # Only one trainer per rollout actor fires the receiver-spawn RPC.
            # Ray actor methods are serialized — if all DP ranks each call
            # ``actor.update_weights_from_ipc.remote(...)``, Ray queues N
            # invocations. The 1st invocation's ``collective_rpc`` opens
            # receivers on every TP worker and they consume the buckets each
            # trainer rank pushes; the 2nd..Nth invocations then re-open
            # receivers with no senders left and hang forever. Pick the
            # group-local rank-0 trainer as the spawner; everyone else just
            # opens their ZMQ sender for their own local_rank socket.
            is_actor_lead = local_rank == 0
            spawn_ref = None
            if is_actor_lead:
                spawn_ref = actor.update_weights_from_ipc.remote(
                    peft_config=peft_config,
                    base_sync_done=base_sync_done,
                    use_shm=self._use_shm,
                    stage_ids=list(stages) if stages is not None else None,
                )

            # For each stage, push the full state-dict over the matching socket.
            # We re-use BucketedUpdateWeight's bucketing for the pump itself.
            replica_rank = replica_rank_from_env()
            # If stages is empty (the smoke-friendly default), use the
            # actor's cached ``num_stages``. The engine-side collective_rpc
            # drives one ZMQ socket per (stage, local_rank) pair.
            stage_list = list(stages) if stages else list(range(actor_n_stages))

            for sid in stage_list:
                # Orphan guard: skip ZMQ send if this rank has no matching
                # receiver. HI3 dual-stage AR(TP=4) + DiT(TP=4), but FSDP
                # train has 8 ranks. Ranks 4-7 must drain the iterator (to
                # keep NCCL all_gather in sync) but NOT open a ZMQ sender.
                tp_map = self._actor_tp_per_stage[actor_idx] if actor_idx < len(self._actor_tp_per_stage) else {}
                tp_size = tp_map.get(sid, int(self._placement_cfg.num_rollout_gpus_per_actor))
                if local_rank >= tp_size:
                    # Drain iterator to keep NCCL in sync, but skip ZMQ send
                    for _name, _tensor in self._iter_named_params(
                        peft_config=peft_config,
                        base_sync_done=base_sync_done,
                    ):
                        del _name, _tensor
                    continue

                handle = zmq_handle(
                    replica_rank=replica_rank,
                    stage_id=int(sid),
                    local_rank=int(local_rank),
                )
                self._sender = BucketedWeightSender(
                    zmq_handle=handle,
                    bucket_size_mb=self._bucket_size_mb,
                    use_shm=self._use_shm,
                )
                # Drive the bucket loop directly via the sender's async API.
                asyncio.run(
                    self._sender.async_send_weights(
                        self._iter_named_params(
                            peft_config=peft_config,
                            base_sync_done=base_sync_done,
                        )
                    )
                )
                self._sender = None

            if spawn_ref is not None:
                ray.get(spawn_ref)
        else:
            # Orphan ranks (more train ranks than rollout GPU slots) still
            # need to drain the iterator so DTensor all_gathers inside
            # ``_to_full_tensor`` see every training rank.
            n_stages = (
                len(self._stage_ids)
                if self._stage_ids
                else (self._actor_num_stages[0] if self._actor_num_stages else 1)
            )
            for _ in range(n_stages):
                for _name, _tensor in self._iter_named_params(
                    peft_config=peft_config,
                    base_sync_done=base_sync_done,
                ):
                    del _name, _tensor

        # The non-leader ranks need to wait for the lead's RPC to finish
        # before returning, otherwise the next sync would start while the
        # receivers from this one are still draining (and the smoke's
        # post-sync checksum probe would race the load). The training
        # process group barrier serves that purpose.
        if dist.is_initialized():
            dist.barrier()
        self.weight_version += 1

    def _iter_named_params(
        self,
        *,
        peft_config: Optional[dict] = None,
        base_sync_done: bool = False,
    ):
        """Yield ``(name, tensor)`` pairs from the model state dict.

        The IPC handler overrides ``update_weights`` to manage per-stage
        receivers, so the base-class prefix application in
        ``BucketedUpdateWeight.update_weights`` never runs for this path —
        we have to apply ``param_name_prefix`` here too.
        """
        from diffusionrl.utils.peft_merge import adapt_lora_for_vllm, extract_lora_tensors, raw_state_dict

        prefix = self._param_name_prefix
        if peft_config and base_sync_done:
            yield from adapt_lora_for_vllm(
                extract_lora_tensors(
                    self.model,
                    param_name_prefix=prefix,
                    packed_modules=getattr(self, "_packed_modules", None),
                )
            ).items()
            return

        for name, param in raw_state_dict(self.model):
            if name.endswith(".lora_A") or name.endswith(".lora_B"):
                continue
            if prefix:
                name = prefix + name
            yield name, param

    # ``_infer_default_stage_ids`` was removed: the stage count is now
    # cached on ``self._actor_num_stages`` during ``connect_rollout_engines``
    # to avoid an actor-method deadlock — see that method's docstring.

    # The base class's abstract method — unused here because we override
    # ``update_weights`` directly; provide a no-op stub.
    def update_bucket_weights(self, named_tensors, weight_version=None, is_last_bucket: bool = False) -> None:
        del named_tensors, weight_version, is_last_bucket


__all__ = ["IPCBucketedSyncConfig", "UpdateWeightFromIPC"]
