"""Standalone driver for the vLLM-Omni rollout engine.

Modality-keyed cases (one per upstream HI3 mode):

    --modality t2i  --eta 0.3   # AR think → DiT denoise → image (SDE on)
    --modality t2i  --eta 0.0   # ODE regression (no SDE capture)
    --modality it2i --eta 0.3   # AR (image+text) → DiT → edited image
    --modality i2t              # AR-only: image+text → text
    --modality t2t              # AR-only: text → text

The script intentionally does not depend on any ``RolloutActor`` /
``RolloutPipeline`` / ``train.py`` machinery — it builds a ``RolloutReq``
by hand, runs the engine, and inspects ``RolloutResp``. See the plan at
``/Users/linyuwu/.claude/plans/i-think-we-are-fluffy-wilkinson.md`` for
verification details.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import torch

from diffusionrl.rollout.engine.vllm_omni import (
    VLLMOmniEngineConfig,
    VLLMOmniRolloutEngine,
)
from diffusionrl.types.primitives import Image, Images, Texts
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutResp


@dataclass
class CaseResult:
    label: str
    resp: RolloutResp
    saved_paths: list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--model-path",
        required=True,
        help="HF id or local path for HunyuanImage-3.0 base or Instruct.",
    )
    parser.add_argument(
        "--modality",
        choices=["t2i", "it2i", "i2t", "t2t"],
        default="t2i",
        help="Upstream HI3 modality. Drives YAML selection + request shape.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Override the per-modality default prompt.",
    )
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reference-image",
        default=None,
        help="Path to reference PIL image (required for it2i/i2t).",
    )
    parser.add_argument(
        "--out-dir",
        default="/tmp/vllm_omni_proto",
        help="Directory for saved sample PNGs.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1,
        help=(
            "Number of prompts. Default 1 because HI3 in vllm-omni's "
            "build_batch_rope_image_info has historically misiterated for "
            "N>1. Bump once the upstream batching is fixed."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------


_DEFAULT_PROMPTS = {
    "t2i": "a red apple on a wooden table",
    "it2i": "make the apple green",
    "i2t": "Describe this image briefly.",
    "t2t": "What is the capital of France?",
}


def _load_pil(path: str):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"--reference-image is required and must point at a real file; got {path!r}")
    from PIL import Image as PILImage

    return PILImage.open(path).convert("RGB")


def _make_req(args: argparse.Namespace) -> RolloutReq:
    modality = args.modality
    prompt = args.prompt or _DEFAULT_PROMPTS[modality]
    n = max(1, int(args.num_prompts))

    sample_ids = [f"s{i}" for i in range(n)]
    group_ids = ["g0"] * n
    primitives: dict = {"text": Texts(texts=[prompt] * n)}

    if modality in ("it2i", "i2t"):
        from torchvision.transforms.functional import pil_to_tensor

        pil = _load_pil(args.reference_image)
        pixels = pil_to_tensor(pil).to(torch.float32) / 255.0
        primitives["image"] = Images.from_list([Image(pixels=pixels) for _ in range(n)])

    stage_params: dict = {}
    if modality in ("t2i", "it2i"):
        stage_params["diffusion"] = {
            "height": int(args.height),
            "width": int(args.width),
            "num_inference_steps": int(args.num_inference_steps),
            "guidance_scale": float(args.guidance_scale),
            "eta": float(args.eta),
            "seed": int(args.seed),
        }
        stage_params["ar"] = {"max_tokens": 2048, "temperature": 0.6}
    else:
        stage_params["ar"] = {"max_tokens": 1024, "temperature": 0.6}

    return RolloutReq(
        sample_ids=sample_ids,
        group_ids=group_ids,
        primitives=primitives,
        stage_params=stage_params,
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _save_decoded(resp: RolloutResp, out_dir: str, prefix: str) -> list:
    os.makedirs(out_dir, exist_ok=True)
    saved: list = []
    images = resp.decoded.get("image")
    if images is not None:
        pil_list = images.to_list()
        for idx, item in enumerate(pil_list):
            path = os.path.join(out_dir, f"{prefix}_sample{idx}.png")
            item.to_pil().save(path)
            saved.append(path)
    return saved


def _print_report(label: str, resp: RolloutResp, saved_paths: list, *, modality: str, eta: float) -> None:
    print(f"\n=== {label} (modality={modality}, eta={eta}) ===")
    print(f"  sample_ids: {resp.sample_ids}")
    print(f"  group_ids:  {resp.group_ids}")
    if "image" in resp.decoded:
        print(f"  decoded[image].pixels.shape: {tuple(resp.decoded['image'].pixels.shape)}")
    if "text" in resp.decoded:
        texts = resp.decoded["text"].texts
        for i, t in enumerate(texts):
            preview = t[:80] + ("..." if len(t) > 80 else "")
            print(f"  decoded[text][{i}]: {preview!r}")
    seg_image = resp.rollout_traces.get("image")
    if seg_image is not None:
        if seg_image.latents is not None:
            print(f"  rollout_traces[image].latents.shape: {tuple(seg_image.latents.shape)}")
        if seg_image.sde_logp is not None:
            lp = seg_image.sde_logp
            print(
                f"  rollout_traces[image].sde_logp.shape: {tuple(lp.shape)} "
                f"(mean={lp.mean().item():.3f}, std={lp.std().item():.3f})"
            )
        else:
            print("  rollout_traces[image].sde_logp: None  (deterministic ODE)")
        if seg_image.sigmas is not None:
            print(f"  rollout_traces[image].sigmas.shape: {tuple(seg_image.sigmas.shape)}")
    seg_ar = resp.rollout_traces.get("ar")
    if seg_ar is not None:
        n_tok = int(seg_ar.tokens.shape[0]) if seg_ar.tokens is not None else 0
        print(f"  rollout_traces[ar].total_tokens: {n_tok}")
        if seg_ar.lengths is not None:
            print(f"  rollout_traces[ar].lengths: {seg_ar.lengths.tolist()}")
        if seg_ar.log_probs is not None:
            print(f"  rollout_traces[ar].log_probs.shape: {tuple(seg_ar.log_probs.shape)}")
    for p in saved_paths:
        print(f"  saved: {p}")


def _assert_image_modality(resp: RolloutResp, *, expect_sde: bool, n: int) -> None:
    expected_ids = [f"s{i}" for i in range(n)]
    assert resp.sample_ids == expected_ids, resp.sample_ids
    assert "image" in resp.decoded, "decoded should contain 'image'"
    pix = resp.decoded["image"].pixels
    assert pix.shape[0] == n and pix.shape[1] == 3, pix.shape
    seg = resp.rollout_traces["image"]
    if expect_sde:
        assert seg.latents is not None, "SDE on but latents None"
        assert seg.latents.shape[0] == n, seg.latents.shape
        assert seg.sde_logp is not None, "SDE on but sde_logp is None"
        assert seg.sde_logp.shape[0] == n, seg.sde_logp.shape
        assert torch.isfinite(seg.sde_logp).all(), "sde_logp contains non-finite values"
        assert seg.sde_logp.std().item() > 0.0, "sde_logp has zero variance"
        # Position-0 latent capture means latents has T+1 along step dim,
        # one more than sde_logp.
        assert seg.latents.shape[1] == seg.sde_logp.shape[1] + 1, (
            f"expected latents [B, T+1, ...] vs log_probs [B, T]; "
            f"got latents={seg.latents.shape}, sde_logp={seg.sde_logp.shape}"
        )
    else:
        assert seg.sde_logp is None, "SDE off but sde_logp populated"
    assert "ar" in resp.rollout_traces, "AR segment must be present (Stage 0 final_output: true)"


def _assert_ar_only(resp: RolloutResp, *, n: int) -> None:
    expected_ids = [f"s{i}" for i in range(n)]
    assert resp.sample_ids == expected_ids, resp.sample_ids
    assert "text" in resp.decoded
    assert "image" not in resp.rollout_traces, "AR-only modality should not emit image segment"
    assert "ar" in resp.rollout_traces


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def _build_cfg(args: argparse.Namespace) -> VLLMOmniEngineConfig:
    return VLLMOmniEngineConfig(
        model_path=args.model_path,
        modality=args.modality,
        default_eta=float(args.eta),
        default_num_inference_steps=int(args.num_inference_steps),
        default_guidance_scale=float(args.guidance_scale),
    )


def main() -> int:
    args = parse_args()

    cfg = _build_cfg(args)
    engine = VLLMOmniRolloutEngine(cfg)
    try:
        req = _make_req(args)
        resp = engine.generate(req)

        prefix = f"{args.modality}_{int(args.eta * 100):03d}eta"
        saved = _save_decoded(resp, args.out_dir, prefix)
        _print_report(
            f"vllm-omni rollout — {args.modality}",
            resp,
            saved,
            modality=args.modality,
            eta=args.eta,
        )

        if args.modality in ("t2i", "it2i"):
            _assert_image_modality(resp, expect_sde=args.eta > 0.0, n=int(args.num_prompts))
        else:
            _assert_ar_only(resp, n=int(args.num_prompts))
        print("\nOK.")
        return 0
    finally:
        engine.shutdown()


if __name__ == "__main__":
    sys.exit(main())
