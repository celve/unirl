#!/usr/bin/env python3
"""SGLang-diffusion FAT grouped-TP boot probe (LIN-535).

Isolates the one runtime unknown behind the diffusion tp>1 design: does the sglang
``DiffGenerator`` actually launch ``num_gpus=tp`` / ``tp_size=tp`` as ONE fat driver
over ``tp`` GPUs — with ``CUDA_VISIBLE_DEVICES`` pinned to a specific allocated pair
(the controller's ``visible_devices``, not a pop-to-all) — and then generate?

Mirrors ``rollout_sd3_sglang_smoke.py`` but constructs the engine with the fat config
the ``Handle`` would stamp on a group head (``num_gpus=tp``, ``tp_size=tp``,
``visible_devices``). One engine, one head, no participants — this validates the fat
RUNTIME + the CVD pin, before the full grouped Handle path.

    PRETRAINED_MODEL=/root/unirl/models/local/stable-diffusion-3.5-medium \
        FAT_TP=2 FAT_GPUS=0,1 .venv-sglang/bin/python scripts/probe_sd3_fat_tp2.py

Exits 0 on PASS, non-zero on failure. GPU-only (needs FAT_TP visible GPUs + SD3.5).
"""

from __future__ import annotations

import os
import sys
import traceback

import torch

from unirl.models.sd3.config import SD3PipelineConfig
from unirl.rollout.engine.sglang_diffusion.config import SGLangDiffusionEngineConfig
from unirl.rollout.engine.sglang_diffusion.engine import SGLangDiffusionRolloutEngine
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import DiffusionSamplingParams


def _log(msg: str) -> None:
    print(f"[sd3-fat-probe] {msg}", flush=True)


def main() -> int:
    model_path = os.environ.get("PRETRAINED_MODEL")
    if not model_path or not os.path.isdir(model_path):
        _log(f"ERROR: PRETRAINED_MODEL must be a local SD3.5 dir; got {model_path!r}")
        return 2
    tp = int(os.environ.get("FAT_TP", "2"))
    gpus = os.environ.get("FAT_GPUS", ",".join(str(i) for i in range(tp)))
    _log(f"torch {torch.__version__} cuda={torch.version.cuda}; FAT driver tp={tp} over GPUs [{gpus}]")

    prompts = ["a photo of a red apple on a wooden table", "an astronaut riding a horse on the moon"]
    input_part = Part.input([f"p{i}" for i in range(len(prompts))], primitive=Texts(texts=prompts))
    diff_params = DiffusionSamplingParams(num_inference_steps=4, guidance_scale=1.0)
    sample = Sample.request(input_part).fork(1, sampling_params=diff_params)

    model_config = SD3PipelineConfig(pretrained_model_ckpt_path=model_path, model_precision="bf16", shift=3.0)
    # The fat coords the Handle would stamp on a group head:
    config = SGLangDiffusionEngineConfig(
        model_family="sd3",
        local_mode=True,
        populate_conditions=True,
        num_gpus=tp,
        tp_size=tp,
        visible_devices=gpus,  # engine pins CUDA_VISIBLE_DEVICES to exactly these before boot
    )

    engine = None
    try:
        _log(f"constructing SGLangDiffusionRolloutEngine FAT (num_gpus={tp} tp_size={tp} CVD={gpus}) — "
             f"boots the DiffGenerator fat driver + loads SD3.5 ...")
        engine = SGLangDiffusionRolloutEngine(
            config, device=torch.device("cuda:0"), rank=0, model_config=model_config
        )
        _log("engine constructed (fat driver booted over the group's GPUs) — generating ...")
        out = engine.generate(sample)

        assert isinstance(out, Sample), f"generate must return a Sample; got {type(out).__name__}"
        gens = out.gen_parts()
        assert gens and gens[-1].segment is not None, "no filled gen segment produced"
        _log(f"SD3 FAT tp={tp} PROBE PASSED ✅  (fat driver launched over GPUs [{gpus}] + generated a latent)")
        return 0
    except Exception:
        _log("SD3 FAT PROBE FAILED ❌")
        traceback.print_exc()
        return 1
    finally:
        if engine is not None:
            try:
                engine.shutdown()
                _log("engine shut down")
            except Exception:
                _log("engine.shutdown() raised (ignored)")


if __name__ == "__main__":
    sys.exit(main())
