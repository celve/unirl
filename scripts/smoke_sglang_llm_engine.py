"""End-to-end smoke for the SGLang LLM rollout engine.

Phases:

1. ``ray.init`` (head only — single-node, 1 GPU).
2. Build a thin Ray actor that wraps :class:`SGLangLLMRolloutEngine` directly
   (bypassing Hydra to keep the smoke independent of ``train_new.yaml`` and
   placement / training-side schemas).
3. ``engine.health_check()`` returns True.
4. ``engine.generate(req)`` with a two-prompt :class:`RolloutReq`; assert the
   :class:`RolloutResp` contains :class:`Texts` and a :class:`TextSegment`
   with packed varlen tokens.
5. Weight-sync RPC handshake: ``init_weights_update_group`` followed by
   ``destroy_weights_update_group``. Proves the HTTP weight-sync surface
   is reachable and that ``NCCL_CUMEM_ENABLE=1`` was set before SRT launch.
   We do not drive an actual broadcast here — that requires a trainer-side
   NCCL rank to participate, which is out of smoke scope.
6. Sleep / wake round-trip (only when ``ENABLE_SLEEP=1`` is set, since it
   requires ``enable_memory_saver=true`` + ``enable_weights_cpu_backup=true``
   plus the ``torch_memory_saver`` pip package).
7. Engine shutdown; ``pgrep -af sglang`` should be empty on the host afterwards.

Run on a pod with at least 1 H20 GPU::

    cd ~/diffusionrl && source .venv/bin/activate
    LLM_MODEL=/mnt/gz/share_305110755/hunyuan/public_models/Qwen3-1.7B \
      python scripts/smoke_sglang_llm_engine.py \
      2>&1 | tee /mnt/gz/logs/smoke-sglang-llm.log
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("smoke_sglang_llm")


_DEFAULT_MODEL = "/mnt/gz/share_305110755/hunyuan/public_models/Qwen3-1.7B"
_PROMPTS = [
    "Write a haiku about reinforcement learning.",
    "Explain diffusion models in one sentence.",
]


def _build_engine_remote_class():
    """Return a Ray remote class that owns one engine instance.

    The class is defined inside this helper so the ``ray.remote`` decorator
    isn't applied at module-import time (lets the unit-test path import the
    smoke driver without requiring ray).
    """
    import ray

    @ray.remote(num_gpus=1)
    class _EngineActor:
        def __init__(self, *, model_path: str, enable_sleep: bool) -> None:
            from diffusionrl.rollout.engine.sglang_llm import (
                SGLangLLMEngineConfig,
                SGLangLLMRolloutEngine,
            )

            engine_kwargs: Dict[str, Any] = {
                "mem_fraction_static": 0.7,
                "skip_server_warmup": True,
                "disable_cuda_graph": True,
                # fa3 kernel is broken on this pod's sgl_kernel build
                # (undefined run_mha_fwd symbol); flashinfer is stable.
                "attention_backend": "flashinfer",
            }
            if enable_sleep:
                engine_kwargs["enable_memory_saver"] = True
                engine_kwargs["enable_weights_cpu_backup"] = True

            cfg = SGLangLLMEngineConfig(
                pretrained_model_ckpt_path=model_path,
                tp_size=1,
                max_new_tokens=32,
                temperature=0.7,
                top_p=0.9,
                concurrency=4,
                engine_kwargs=engine_kwargs,
            )
            self._engine = SGLangLLMRolloutEngine(cfg, rank=0)

        def health_check(self) -> bool:
            return bool(self._engine.health_check())

        def generate(self, prompts: List[str]) -> Dict[str, Any]:
            import torch

            from diffusionrl.types.primitives import Texts
            from diffusionrl.types.rollout_req import RolloutReq
            from diffusionrl.types.sampling import ARSamplingParams

            req = RolloutReq(
                sample_ids=[f"s{i}" for i in range(len(prompts))],
                group_ids=["g0"] * len(prompts),
                primitives={"text": Texts(texts=list(prompts))},
                sampling_params=ARSamplingParams(max_new_tokens=16, samples_per_prompt=1),
            )
            resp = self._engine.generate(req)

            track = resp.tracks.get("ar")
            seg = track.segment if track is not None else None
            tokens = getattr(seg, "tokens", None) if seg is not None else None
            cu = getattr(seg, "cu_seqlens", None) if seg is not None else None
            return {
                "texts": list(track.decoded.texts) if track is not None and track.decoded is not None else [],
                "sample_ids": list(track.sample_ids) if track is not None else [],
                "group_ids": list(track.group_ids) if track is not None else [],
                "n_tokens_total": int(tokens.shape[0]) if isinstance(tokens, torch.Tensor) else 0,
                "cu_seqlens": cu.tolist() if isinstance(cu, torch.Tensor) else None,
                "seg_count": int(seg.batch_size) if seg is not None else 0,
            }

        def init_weights_update_group(
            self,
            *,
            master_address: str,
            master_port: int,
            rank_offset: int,
            world_size: int,
            group_name: str,
        ) -> None:
            self._engine.init_weights_update_group(
                master_address=master_address,
                master_port=int(master_port),
                rank_offset=int(rank_offset),
                world_size=int(world_size),
                group_name=group_name,
            )

        def update_weights_from_distributed(
            self,
            *,
            names: List[str],
            dtypes: List[str],
            shapes: List[List[int]],
            group_name: str,
        ) -> None:
            self._engine.update_weights_from_distributed(
                names=list(names),
                dtypes=list(dtypes),
                shapes=[list(s) for s in shapes],
                group_name=group_name,
                flush_cache=False,
            )

        def destroy_weights_update_group(self, *, group_name: str) -> None:
            self._engine.destroy_weights_update_group(group_name=group_name)

        def sleep_wake_round_trip(self) -> Dict[str, Any]:
            t0 = time.perf_counter()
            self._engine.sleep()
            sleep_s = time.perf_counter() - t0
            assert self._engine.is_offloaded, "Engine should be offloaded after sleep()"
            t1 = time.perf_counter()
            self._engine.wake_up()
            wake_s = time.perf_counter() - t1
            assert not self._engine.is_offloaded, "Engine should NOT be offloaded after wake_up()"
            return {"sleep_s": sleep_s, "wake_s": wake_s}

        def shutdown(self) -> None:
            self._engine.shutdown()

    return _EngineActor


def _build_trainer_remote_class():
    """Return a Ray remote class that drives the trainer side of NCCL weight sync.

    Owns one GPU; sets up torch.distributed as rank 0 of a 2-rank NCCL group
    where the engine actor's SGLang scheduler is rank 1. Used by phase 5 to
    drive an end-to-end NCCL broadcast through the engine's
    ``update_weights_from_distributed`` path — matching how the production
    ``UpdateWeightFromDistributed`` handler at
    ``diffusionrl/distributed/weight_sync/nccl.py`` runs.
    """
    import ray

    @ray.remote(num_gpus=1)
    class _TrainerSideActor:
        def __init__(self) -> None:
            import torch

            # Mirror the engine actor's allocator setting so CUDA-IPC compat
            # is symmetric across both sides of the NCCL group.
            try:
                torch.cuda.memory._set_allocator_settings("expandable_segments:False")
            except Exception:
                pass
            self._torch = torch
            self._dist = torch.distributed
            self._nccl_group: Any = None
            self._device = torch.device("cuda")
            # Match the SGLang server's transport defaults so this side
            # negotiates the same NCCL channels.
            os.environ.setdefault("NCCL_CUMEM_ENABLE", "1")
            os.environ.setdefault("NCCL_NVLS_ENABLE", "1")

        def init_pg(
            self,
            *,
            master_address: str,
            master_port: int,
            rank: int,
            world_size: int,
            group_name: str,
        ) -> str:
            # Use diffusionrl's PrefixStore-aware init_process_group: SGLang's
            # init_custom_process_group prefixes rendezvous keys with
            # ``group_name`` (PrefixStore wrap). torch.distributed's stock
            # init_process_group does NOT — using the stock helper here would
            # never rendezvous because the engine and trainer would write
            # rendezvous keys to disjoint namespaces and dist.broadcast would
            # deadlock with both sides spinning in NCCL.
            from diffusionrl.utils.distributed_utils import init_process_group

            init_method = f"tcp://{master_address}:{int(master_port)}"
            self._nccl_group = init_process_group(
                backend="nccl",
                init_method=init_method,
                rank=int(rank),
                world_size=int(world_size),
                group_name=group_name,
            )
            return f"trainer rank={rank}/{world_size} group={group_name!r} init_method={init_method}"

        def broadcast(
            self,
            *,
            name: str,
            dtype: str,
            shape: List[int],
            src: int = 0,
            fill: float = 1.0,
        ) -> Dict[str, Any]:
            torch = self._torch
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            t_dtype = dtype_map.get(dtype.replace("torch.", ""))
            if t_dtype is None:
                raise ValueError(f"unsupported dtype {dtype!r}")
            t = torch.full(tuple(int(s) for s in shape), float(fill), dtype=t_dtype, device=self._device)
            t0 = time.perf_counter()
            self._dist.broadcast(t, src=int(src), group=self._nccl_group, async_op=False)
            elapsed = time.perf_counter() - t0
            return {
                "name": name,
                "shape": list(t.shape),
                "dtype": dtype,
                "elapsed_s": elapsed,
                "checksum": float(t.float().sum().item()),
            }

        def destroy_pg(self) -> None:
            if self._nccl_group is not None:
                self._dist.destroy_process_group(self._nccl_group)
            self._nccl_group = None

    return _TrainerSideActor


def _find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def main() -> int:
    model_path = os.environ.get("LLM_MODEL", _DEFAULT_MODEL)
    enable_sleep = os.environ.get("ENABLE_SLEEP", "0") == "1"

    logger.info("== Phase 1/7: ray.init ==")
    import ray

    ray.init(ignore_reinit_error=True)

    logger.info("== Phase 2/7: build engine actor (model=%s) ==", model_path)
    EngineActor = _build_engine_remote_class()
    actor = EngineActor.remote(model_path=model_path, enable_sleep=enable_sleep)

    logger.info("== Phase 3/7: health_check ==")
    ok = ray.get(actor.health_check.remote())
    if not ok:
        logger.error("health_check returned False")
        return 1
    logger.info("HEALTH OK")

    logger.info("== Phase 4/7: generate ==")
    gen = ray.get(actor.generate.remote(_PROMPTS))
    logger.info("decoded texts: %s", gen["texts"])
    logger.info(
        "tracks[ar].segment: %d segs, %d total tokens, cu_seqlens=%s",
        gen["seg_count"],
        gen["n_tokens_total"],
        gen["cu_seqlens"],
    )
    if not gen["texts"] or not all(gen["texts"]):
        logger.error("GENERATE FAIL: empty texts %r", gen["texts"])
        return 2
    if gen["seg_count"] != len(_PROMPTS):
        logger.error(
            "GENERATE FAIL: expected %d segments, got %d",
            len(_PROMPTS),
            gen["seg_count"],
        )
        return 3
    if gen["n_tokens_total"] <= 0:
        logger.error("GENERATE FAIL: no tokens emitted")
        return 4
    logger.info("GENERATE OK: %s", gen["texts"][0][:80])

    logger.info("== Phase 5/7: NCCL weight-sync end-to-end ==")
    TrainerSideActor = _build_trainer_remote_class()
    trainer = TrainerSideActor.remote()
    # Materialize the trainer (its ctor does CUDA setup) before we kick off
    # the blocking init_weights_update_group call on the engine.
    ray.get(trainer.__ray_ready__.remote()) if False else None
    master_addr = ray.util.get_node_ip_address()
    master_port = _find_free_port()
    group_name = "smoke"

    # Issue both sides of the handshake in parallel. The SGLang scheduler
    # blocks inside /init_weights_update_group until the trainer rank 0 also
    # rendezvous on the same (master_addr, master_port) with the same
    # group_name; without firing both refs we'd deadlock.
    logger.info(
        "NCCL rendezvous: master=%s:%d group=%r world_size=2",
        master_addr,
        master_port,
        group_name,
    )
    engine_init = actor.init_weights_update_group.remote(
        master_address=master_addr,
        master_port=master_port,
        rank_offset=1,
        world_size=2,
        group_name=group_name,
    )
    trainer_init = trainer.init_pg.remote(
        master_address=master_addr,
        master_port=master_port,
        rank=0,
        world_size=2,
        group_name=group_name,
    )
    t0 = time.perf_counter()
    ray.get([engine_init, trainer_init])
    handshake_s = time.perf_counter() - t0
    logger.info("NCCL HANDSHAKE OK in %.2fs", handshake_s)

    # Broadcast a small tensor through the NCCL group: trainer side sends,
    # engine side receives via update_weights_from_distributed (which the
    # SGLang scheduler implements as a NCCL recv). Pick a real Qwen3 param
    # name + shape so the SRT-side application succeeds rather than 4xx'ing.
    # ``model.norm.weight`` is the final RMSNorm at hidden_size=1024 on
    # Qwen3-0.6B (and Qwen3-4B; shape may differ on other model sizes).
    name = "model.norm.weight"
    dtype = "bfloat16"
    shape = [1024]
    logger.info("NCCL broadcast: name=%s dtype=%s shape=%s", name, dtype, shape)
    engine_recv = actor.update_weights_from_distributed.remote(
        names=[name],
        dtypes=[dtype],
        shapes=[shape],
        group_name=group_name,
    )
    trainer_send = trainer.broadcast.remote(name=name, dtype=dtype, shape=shape, src=0, fill=1.0)
    t0 = time.perf_counter()
    _, bcast_info = ray.get([engine_recv, trainer_send])
    bcast_s = time.perf_counter() - t0
    logger.info(
        "WEIGHT SYNC NCCL OK: broadcast %s shape=%s in %.2fs (sender checksum=%.4f)",
        bcast_info["name"],
        bcast_info["shape"],
        bcast_s,
        bcast_info["checksum"],
    )

    # Teardown the NCCL group on both sides — IN PARALLEL. SGLang's
    # destroy_weights_update_group calls dist.destroy_process_group on its
    # side which barriers with the peer rank; issuing the two destroys
    # sequentially deadlocks (engine blocks waiting for trainer rank 0 to
    # destroy too, but the driver hasn't dispatched that ray future yet).
    engine_destroy = actor.destroy_weights_update_group.remote(group_name=group_name)
    trainer_destroy = trainer.destroy_pg.remote()
    ray.get([engine_destroy, trainer_destroy])
    logger.info("NCCL TEARDOWN OK")

    if enable_sleep:
        logger.info("== Phase 6/7: sleep / wake round-trip ==")
        sw = ray.get(actor.sleep_wake_round_trip.remote())
        logger.info(
            "SLEEP/WAKE OK: sleep=%.2fs wake=%.2fs",
            sw["sleep_s"],
            sw["wake_s"],
        )
    else:
        logger.info("== Phase 6/7: sleep / wake SKIPPED (set ENABLE_SLEEP=1 to enable) ==")

    logger.info("== Phase 7/7: shutdown ==")
    ray.get(actor.shutdown.remote())

    # Flush + write a robust pass marker BEFORE ray.kill / interpreter
    # shutdown. CUDA / NCCL destructors firing during ray actor teardown
    # have segfaulted the driver in the past on this image, swallowing
    # the final logger.info before stdio flush.
    logger.info("ALL PHASES PASSED")
    for h in list(logging.getLogger().handlers) + list(logger.handlers):
        try:
            h.flush()
        except Exception:
            pass
    sys.stdout.flush()
    sys.stderr.flush()

    ray.kill(actor)
    ray.kill(trainer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
