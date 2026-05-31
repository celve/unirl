"""End-to-end smoke for the SGLang LLM rollout engine with VLM (image+text) inputs.

Phases:

1. ``ray.init`` (head only — single-node, 1 GPU).
2. Build a Ray actor wrapping :class:`SGLangLLMRolloutEngine` with
   ``image_token`` set for Qwen2.5-VL.
3. ``engine.health_check()`` returns True.
4a. Text-only ``generate`` — baseline backward-compat check.
4b. VLM ``generate`` with synthetic PIL images — validates the full
    image pipeline: ``Images`` primitive → PIL → base64 → HTTP
    ``image_data`` → SRT multimodal → text + logprobs.
5. NCCL weight-sync handshake (same as LLM smoke).
6. Engine shutdown.

Run on a pod with at least 1 H20 GPU::

    cd ~/diffusionrl-vlm && source .venv/bin/activate
    VLM_MODEL=/root/diffusionrl-vlm/models/local/Qwen2.5-VL-3B-Instruct \\
      python scripts/smoke_sglang_vlm_engine.py \\
      2>&1 | tee /mnt/zw/logs/smoke-vlm.log
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("smoke_sglang_vlm")


_DEFAULT_MODEL = "/root/diffusionrl-vlm/models/local/Qwen2.5-VL-3B-Instruct"

_TEXT_ONLY_PROMPTS = [
    "Write a haiku about reinforcement learning.",
    "Explain diffusion models in one sentence.",
]

_VLM_PROMPTS = [
    "What color is this image? Answer in one word.",
    "Describe what you see in this image in one sentence.",
]


def _build_engine_remote_class():
    import ray

    @ray.remote(num_gpus=1)
    class _EngineActor:
        def __init__(self, *, model_path: str) -> None:
            from diffusionrl.rollout.engine.sglang_llm import (
                SGLangLLMEngineConfig,
                SGLangLLMRolloutEngine,
            )

            cfg = SGLangLLMEngineConfig(
                pretrained_model_ckpt_path=model_path,
                image_token="<|vision_start|><|image_pad|><|vision_end|>",
                tp_size=1,
                max_new_tokens=64,
                temperature=0.7,
                top_p=0.9,
                concurrency=4,
                engine_kwargs={
                    "mem_fraction_static": 0.7,
                    "skip_server_warmup": True,
                    "disable_cuda_graph": True,
                    "attention_backend": "flashinfer",
                },
            )
            self._engine = SGLangLLMRolloutEngine(cfg, rank=0)

        def health_check(self) -> bool:
            return bool(self._engine.health_check())

        def generate(
            self,
            prompts: List[str],
            image_colors: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            import PIL.Image
            import torch

            from diffusionrl.types.primitives import Images, Texts
            from diffusionrl.types.rollout_req import RolloutReq
            from diffusionrl.types.sampling import ARSamplingParams

            primitives: Dict[str, Any] = {"text": Texts(texts=list(prompts))}

            if image_colors is not None:
                pil_images = [PIL.Image.new("RGB", (224, 224), color=c) for c in image_colors]
                pixels = torch.stack(
                    [
                        torch.tensor(list(img.getdata()), dtype=torch.uint8)
                        .reshape(224, 224, 3)
                        .permute(2, 0, 1)
                        .float()
                        / 255.0
                        for img in pil_images
                    ]
                )
                primitives["image"] = Images(pixels=pixels)

            req = RolloutReq(
                sample_ids=[f"s{i}" for i in range(len(prompts))],
                group_ids=["g0"] * len(prompts),
                primitives=primitives,
                sampling_params=ARSamplingParams(max_new_tokens=32, samples_per_prompt=1),
                stage_config={},
            )
            resp = self._engine.generate(req)

            track = resp.tracks.get("ar")
            seg = track.segment if track is not None else None
            tokens = getattr(seg, "tokens", None) if seg is not None else None
            log_probs = getattr(seg, "log_probs", None) if seg is not None else None
            return {
                "texts": list(track.decoded.texts) if track is not None and track.decoded is not None else [],
                "sample_ids": list(track.sample_ids) if track is not None else [],
                "n_tokens_total": int(tokens.shape[0]) if isinstance(tokens, torch.Tensor) else 0,
                "n_logprobs_total": int(log_probs.shape[0]) if isinstance(log_probs, torch.Tensor) else 0,
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

        def destroy_weights_update_group(self, *, group_name: str) -> None:
            self._engine.destroy_weights_update_group(group_name=group_name)

        def shutdown(self) -> None:
            self._engine.shutdown()

    return _EngineActor


def _build_trainer_remote_class():
    import ray

    @ray.remote(num_gpus=1)
    class _TrainerSideActor:
        def __init__(self) -> None:
            import torch

            try:
                torch.cuda.memory._set_allocator_settings("expandable_segments:False")
            except Exception:
                pass
            self._torch = torch
            self._dist = torch.distributed
            self._nccl_group: Any = None
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
            from diffusionrl.utils.distributed_utils import init_process_group

            init_method = f"tcp://{master_address}:{int(master_port)}"
            self._nccl_group = init_process_group(
                backend="nccl",
                init_method=init_method,
                rank=int(rank),
                world_size=int(world_size),
                group_name=group_name,
            )
            return f"trainer rank={rank}/{world_size}"

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
    model_path = os.environ.get("VLM_MODEL", _DEFAULT_MODEL)

    logger.info("== Phase 1/6: ray.init ==")
    import ray

    ray.init(ignore_reinit_error=True)

    logger.info("== Phase 2/6: build engine actor (model=%s) ==", model_path)
    EngineActor = _build_engine_remote_class()
    actor = EngineActor.remote(model_path=model_path)

    logger.info("== Phase 3/6: health_check ==")
    ok = ray.get(actor.health_check.remote())
    if not ok:
        logger.error("health_check returned False")
        return 1
    logger.info("HEALTH OK")

    # Phase 4a: text-only generate (backward compat)
    logger.info("== Phase 4a/6: text-only generate ==")
    gen_text = ray.get(actor.generate.remote(_TEXT_ONLY_PROMPTS))
    logger.info("text-only decoded: %s", gen_text["texts"])
    if not gen_text["texts"] or not all(gen_text["texts"]):
        logger.error("TEXT-ONLY GENERATE FAIL: empty texts %r", gen_text["texts"])
        return 2
    if gen_text["n_tokens_total"] <= 0:
        logger.error("TEXT-ONLY GENERATE FAIL: no tokens emitted")
        return 3
    if gen_text["n_logprobs_total"] <= 0:
        logger.error("TEXT-ONLY GENERATE FAIL: no logprobs emitted")
        return 3
    logger.info(
        "TEXT-ONLY GENERATE OK: %d segs, %d tokens, %d logprobs",
        gen_text["seg_count"],
        gen_text["n_tokens_total"],
        gen_text["n_logprobs_total"],
    )

    # Phase 4b: VLM generate (image + text)
    logger.info("== Phase 4b/6: VLM generate (image+text) ==")
    gen_vlm = ray.get(
        actor.generate.remote(
            _VLM_PROMPTS,
            image_colors=["red", "blue"],
        )
    )
    logger.info("VLM decoded: %s", gen_vlm["texts"])
    if not gen_vlm["texts"] or not all(gen_vlm["texts"]):
        logger.error("VLM GENERATE FAIL: empty texts %r", gen_vlm["texts"])
        return 4
    if gen_vlm["n_tokens_total"] <= 0:
        logger.error("VLM GENERATE FAIL: no tokens emitted")
        return 5
    if gen_vlm["n_logprobs_total"] <= 0:
        logger.error("VLM GENERATE FAIL: no logprobs emitted")
        return 5
    logger.info(
        "VLM GENERATE OK: %d segs, %d tokens, %d logprobs, texts=%r",
        gen_vlm["seg_count"],
        gen_vlm["n_tokens_total"],
        gen_vlm["n_logprobs_total"],
        [t[:80] for t in gen_vlm["texts"]],
    )

    # Phase 5: NCCL weight-sync handshake
    logger.info("== Phase 5/6: NCCL weight-sync handshake ==")
    TrainerSideActor = _build_trainer_remote_class()
    trainer = TrainerSideActor.remote()
    master_addr = ray.util.get_node_ip_address()
    master_port = _find_free_port()
    group_name = "smoke_vlm"

    logger.info("NCCL rendezvous: master=%s:%d group=%r", master_addr, master_port, group_name)
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
    logger.info("NCCL HANDSHAKE OK in %.2fs", time.perf_counter() - t0)

    engine_destroy = actor.destroy_weights_update_group.remote(group_name=group_name)
    trainer_destroy = trainer.destroy_pg.remote()
    ray.get([engine_destroy, trainer_destroy])
    logger.info("NCCL TEARDOWN OK")

    # Phase 6: shutdown
    logger.info("== Phase 6/6: shutdown ==")
    ray.get(actor.shutdown.remote())

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
