"""End-to-end smoke test for the composed PE pipeline.

Wires a real Qwen3-0.6B (prompt rewrite) into a real SD3.5-medium
(diffusion sample) via ``PEPipeline.from_config`` and runs one forward
pass against a tiny prompt batch on a single GPU.

Run on a pod that has both checkpoints copied to pod-local disk and a
populated ``.venv``:

    cd /root/diffusionrl
    source .venv/bin/activate
    python scripts/smoke_pe.py \\
        --diffusion-path /root/diffusionrl/models/local/stable-diffusion-3.5-medium \\
        --llm-path /root/diffusionrl/models/local/Qwen3-0.6B \\
        --out-dir /mnt/gz/logs/smoke_pe-0522

What it asserts (binary pass/fail, no quality bar on the image itself):
- PEPipeline.from_config succeeds with nested _target_ children.
- pe.generate(req) returns a RolloutResp that contains BOTH
  tracks["ar"].decoded (LLM rewritten prompt) and tracks["diffusion"].decoded
  (final diffusion output).
- The rewritten text is non-empty.
- The output image has the expected ``[B, 3, H, W]`` shape and finite
  pixel statistics.

This is a plumbing test — quality of the rewrite / image is *not*
evaluated, since Qwen3-0.6B is a tiny model and 4-step SD3.5
generation isn't expected to look great.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

# Triggers @register_config side effects for sd3 + qwen3 + pe via
# pkgutil.walk_packages on diffusionrl.models.
from diffusionrl.config import register_all_configs
from diffusionrl.models.pe import PEPipeline
from diffusionrl.models.qwen3 import Qwen3PipelineConfig
from diffusionrl.models.sd3 import SD3PipelineConfig
from diffusionrl.sde.runtime import FlowMatchSchedulePolicy, ensure_req_sigmas
from diffusionrl.types.primitives import Texts
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.sampling import ARSamplingParams, ComposedSamplingParams, DiffusionSamplingParams

logger = logging.getLogger("smoke_pe")


def _build_cfg(diffusion_path: str, llm_path: str) -> Any:
    """Construct the PEPipeline DictConfig with structured nested children.

    Each child is wrapped with ``OmegaConf.structured`` so its registered
    schema (which adds ``_target_`` via ``register_config``'s
    ``make_dataclass`` wrap) attaches to the DictConfig — without this,
    untyped child DictConfigs materialize to plain ``dict`` and break
    ``SD3Pipeline.from_config(config=<dict>)``.
    """
    return OmegaConf.create(
        {
            "_target_": "diffusionrl.models.pe.PEPipeline.from_config",
            "diffusion": OmegaConf.structured(
                SD3PipelineConfig(
                    pretrained_model_ckpt_path=diffusion_path,
                    model_precision="bf16",
                    autocast_precision="bf16",
                )
            ),
            "llm": OmegaConf.structured(
                Qwen3PipelineConfig(
                    pretrained_model_ckpt_path=llm_path,
                    tokenizer_ckpt_path=llm_path,
                    trust_remote_code=True,
                    model_precision="bf16",
                    autocast_precision="bf16",
                )
            ),
        }
    )


def _build_req(
    prompts: list[str],
    height: int,
    width: int,
    num_inference_steps: int,
    max_tokens: int,
) -> RolloutReq:
    n = len(prompts)
    return RolloutReq(
        sample_ids=[f"s{i}" for i in range(n)],
        group_ids=[f"g{i}" for i in range(n)],
        primitives={"text": Texts(texts=list(prompts))},
        request_conditions={},
        sampling_params=ComposedSamplingParams(
            ar=ARSamplingParams(
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
            ),
            diffusion=DiffusionSamplingParams(
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=4.5,
            ),
        ),
        stage_config={
            "chat": {
                # Qwen3-0.6B is instruction-tuned; a short system prompt is
                # enough to nudge it toward "rewrite as a richer image prompt".
                # The pipeline will pass this through to Qwen3ChatTemplateStage.
                "system_instruction": (
                    "Rewrite the user's short prompt as a single richer, "
                    "more descriptive image-generation prompt. Output only "
                    "the rewritten prompt, no explanation."
                ),
            },
        },
        sigmas=None,  # pinned by ensure_req_sigmas below
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffusion-path", required=True, help="Path to SD3.5-medium ckpt")
    parser.add_argument("--llm-path", required=True, help="Path to Qwen3 ckpt")
    parser.add_argument("--out-dir", required=True, help="Directory to write rewritten.txt + img.png")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Diffusion steps. 4 is the cheap smoke setting; quality isn't evaluated.",
    )
    parser.add_argument(
        "--prompt",
        default="a cat sitting on a windowsill",
        help="Raw prompt fed to the LLM for rewriting.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max new tokens for the LLM rewrite. Qwen3 emits a <think> block by default; "
        "values <~200 truncate mid-thinking before the actual rewrite. "
        "Qwen3ARParams default is 512; hard cap is the model context (~32k for Qwen3).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    register_all_configs()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building PE pipeline (diffusion=%s, llm=%s)", args.diffusion_path, args.llm_path)
    cfg = _build_cfg(args.diffusion_path, args.llm_path)
    pe = PEPipeline.from_config(cfg)

    # Sanity-check children are wired
    assert pe.diffusion_pipeline is not None, "diffusion child missing"
    assert pe.llm_pipeline is not None, "llm child missing"
    assert pe.bundle.diffusion is pe.diffusion_pipeline.bundle
    assert pe.bundle.llm is pe.llm_pipeline.bundle
    logger.info(
        "PE pipeline built. diffusion_bundle=%s llm_bundle=%s",
        type(pe.bundle.diffusion).__name__,
        type(pe.bundle.llm).__name__,
    )

    # σ schedule policy for SD3.5
    policy = FlowMatchSchedulePolicy.from_pretrained(args.diffusion_path, shift=3.0)
    logger.info("SD3.5 schedule policy: shift=%.2f use_dynamic=%s", policy.shift, policy.use_dynamic_shifting)

    req = _build_req([args.prompt], args.height, args.width, args.steps, args.max_tokens)
    ensure_req_sigmas(req, policy)
    logger.info("σ pinned: shape=%s dtype=%s", tuple(req.sigmas.shape), req.sigmas.dtype)

    logger.info("Running pe.generate(req) with prompt=%r", args.prompt)
    with torch.inference_mode():
        resp = pe.generate(req)

    # ----- LLM half checks -----
    assert "ar" in resp.tracks, "tracks['ar'] missing — LLM child produced no output"
    rewritten = resp.tracks["ar"].decoded
    assert isinstance(rewritten, Texts), f"tracks['ar'].decoded wrong type: {type(rewritten).__name__}"
    assert len(rewritten.texts) == 1, f"expected 1 rewritten text, got {len(rewritten.texts)}"
    assert rewritten.texts[0].strip(), "rewritten text is empty/whitespace"
    rewritten_str = rewritten.texts[0]
    logger.info("LLM rewritten prompt: %r", rewritten_str)

    # ----- Diffusion half checks -----
    assert "diffusion" in resp.tracks, "tracks['diffusion'] missing — diffusion child produced no output"
    images = resp.tracks["diffusion"].decoded
    pixels = images.pixels
    assert pixels.dim() == 4, f"image pixels expected [B,C,H,W], got shape {tuple(pixels.shape)}"
    assert pixels.shape[0] == 1, f"expected batch=1, got {pixels.shape[0]}"
    assert pixels.shape[1] == 3, f"expected 3 channels, got {pixels.shape[1]}"
    assert pixels.shape[2] == args.height
    assert pixels.shape[3] == args.width
    assert torch.isfinite(pixels).all(), "image contains NaN/Inf"
    p_min = float(pixels.min())
    p_max = float(pixels.max())
    p_mean = float(pixels.mean())
    logger.info(
        "Image pixel stats: shape=%s min=%.3f max=%.3f mean=%.3f",
        tuple(pixels.shape),
        p_min,
        p_max,
        p_mean,
    )

    # ----- Persist artifacts -----
    (out_dir / "rewritten.txt").write_text(
        f"raw_prompt: {args.prompt}\nrewritten:  {rewritten_str}\n",
        encoding="utf-8",
    )

    # SD3VAEDecodeStage already normalizes to [0, 1] (see sd3/vae.py:55),
    # so save the pixels as-is — no further tone-mapping.
    from torchvision.utils import save_image

    save_image(pixels.detach().float(), str(out_dir / "img.png"))

    logger.info(
        "SMOKE PE OK — artifacts at %s (rewritten.txt, img.png, %.1f KB image)",
        out_dir,
        (out_dir / "img.png").stat().st_size / 1024.0,
    )
    print("SMOKE_PE_RESULT: PASS")


if __name__ == "__main__":
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")
    main()
