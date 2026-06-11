#!/usr/bin/env python3
"""BAGEL all-modality smoke test (t2t / i2t / it2t / t2i / it2i) + replay checks.

Loads the Bagel bundle ONCE (single device, ``enable_vit=True`` when any
image-input mode is requested) and drives ``BagelPipeline.generate(req)`` per
task. Beyond hi3-style generation PASS/FAIL, this also verifies the RL
contracts per mode:

  text-out (t2t/i2t/it2t):
    - generation produces non-empty text
    - GREEDY CROSS-CHECK (t2t): our rl_ops.decode_text at T<=0 emits the exact
      token ids of the pristine vendored ``generate_text(do_sample=False)`` —
      validates the reimplemented bs=1 index bookkeeping byte-for-byte
    - REPLAY PARITY: stage.replay under no_grad, identical weights, same T →
      median |new_logp - old_logp| < 2e-2 nats (the ratio-consistency signal;
      observed ~5e-4 to ~9e-3). bf16 q_len=1 (rollout) vs q_len=n (replay)
      flash-kernel batching makes a few low-confidence tokens diverge up to
      ~0.2-0.3 nats (loose max < 5e-1). The image modes' fp32-trajectory
      replays are bit-exact (ratio==0) — proof the path is exact and the text
      deltas are bf16 noise.
    - GRAD SMOKE: replay with embed_tokens grad-enabled, sum().backward() —
      validates the grad-capable path through the navit und stack (eval+grads)

  image-out (t2i/it2i):
    - generation produces a sane [1, 3, H, W] image (saved to OUT_DIR)
    - DETERMINISM (it2i): two identical runs → bit-equal trajectory latents
    - REPLAY RATIO: diffusion replay over the SDE window, identical weights →
      max |new_logp - sde_logp| < 1e-4 (fp32 trajectory ⇒ ratio==1 regime)

Env knobs:
    BAGEL_PATH         default ByteDance-Seed/BAGEL-7B-MoT (use a local copy)
    OUT_DIR            default /tmp/bagel_smoke_out
    BAGEL_STEPS        default 8     (diffusion steps; σ schedule = steps+1)
    BAGEL_HW           default 512   (square image side)
    BAGEL_MAX_TOKENS   default 64    (AR max_new_tokens)
    BAGEL_MODES        default "t2t,i2t,it2t,t2i,it2i"
    BAGEL_GRAD_SMOKE   default "1"   (set "0" to skip the replay-backward check)
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from typing import Optional

import torch

PRETRAINED = os.environ.get("BAGEL_PATH", "ByteDance-Seed/BAGEL-7B-MoT")
OUT_DIR = os.environ.get("OUT_DIR", "/tmp/bagel_smoke_out")
STEPS = int(os.environ.get("BAGEL_STEPS", "8"))
HW = int(os.environ.get("BAGEL_HW", "512"))
MAX_TOKENS = int(os.environ.get("BAGEL_MAX_TOKENS", "64"))
MODES = [m.strip() for m in os.environ.get("BAGEL_MODES", "t2t,i2t,it2t,t2i,it2i").split(",") if m.strip()]
GRAD_SMOKE = os.environ.get("BAGEL_GRAD_SMOKE", "1") == "1"

AR_TEMPERATURE = 0.7
os.makedirs(OUT_DIR, exist_ok=True)


def log(msg: str) -> None:
    print(f"[bagel-smoke] {msg}", flush=True)


def make_test_image(side: int = 512):
    """A simple, recognizable RGB test image for image-input modes."""
    import PIL.Image
    import PIL.ImageDraw

    img = PIL.Image.new("RGB", (side, side), (30, 60, 140))
    d = PIL.ImageDraw.Draw(img)
    d.ellipse([int(side * 0.25), int(side * 0.25), int(side * 0.75), int(side * 0.75)], fill=(220, 40, 40))
    d.rectangle([int(side * 0.05), int(side * 0.05), int(side * 0.25), int(side * 0.25)], fill=(240, 220, 40))
    return img


def image_primitive(side: int = 512):
    import numpy as np

    from unirl.types.primitives import Image, Images

    arr = np.asarray(make_test_image(side).convert("RGB"), dtype="float32") / 255.0
    return Images.from_list([Image(pixels=torch.from_numpy(arr).permute(2, 0, 1).contiguous())])


def build_pipeline(enable_vit: bool):
    from unirl.models.bagel.config import BagelPipelineConfig
    from unirl.models.bagel.pipeline import BagelPipeline

    config = BagelPipelineConfig(
        pretrained_model_ckpt_path=PRETRAINED,
        model_precision="bf16",
        autocast_precision="bf16",
        trajectory_precision="fp32",
        logprob_precision="fp32",
        shift=3.0,
        enable_vit=enable_vit,
    )
    log(f"loading bundle from {PRETRAINED} (enable_vit={enable_vit}) ...")
    t0 = time.time()
    pipeline = BagelPipeline.from_config(config)
    log(f"bundle loaded in {time.time() - t0:.1f}s")
    return pipeline


def ar_req(task: str, prompt: Optional[str], with_image: bool):
    from unirl.types.primitives import Texts
    from unirl.types.rollout_req import RolloutReq
    from unirl.types.sampling import ARSamplingParams

    prims = {}
    if prompt is not None:
        prims["text"] = Texts(texts=[prompt])
    if with_image:
        prims["image"] = image_primitive(HW)
    return RolloutReq(
        sample_ids=[f"{task}:0"],
        group_ids=[f"g_{task}"],
        primitives=prims,
        sampling_params=ARSamplingParams(temperature=AR_TEMPERATURE, max_new_tokens=MAX_TOKENS, top_p=0.9, top_k=1024),
        stage_config={"task": task},
    )


def image_req(task: str, prompt: str, with_image: bool, pipeline):
    from unirl.models.bagel.diffusion import BagelDiffusionParams
    from unirl.sde.runtime import ensure_req_sigmas
    from unirl.types.primitives import Texts
    from unirl.types.rollout_req import RolloutReq

    prims = {"text": Texts(texts=[prompt])}
    if with_image:
        prims["image"] = image_primitive(HW)
    params = BagelDiffusionParams(
        num_inference_steps=STEPS,
        height=HW,
        width=HW,
        seed=42,
        eta=1.0,
        cfg_text_scale=4.0 if with_image else 1.0,
        cfg_img_scale=1.5 if with_image else 1.0,
        cfg_interval=(0.4, 1.0) if with_image else (0.0, 1.0),
        sde_indices=[0, 1],  # SDE log-probs on the first two steps → replay-ratio check
    )
    req = RolloutReq(
        sample_ids=[f"{task}:0"],
        group_ids=[f"g_{task}"],
        primitives=prims,
        sampling_params=params,
        stage_config={"task": task},
    )
    ensure_req_sigmas(req, pipeline.build_schedule_policy())
    return req


def run_text_mode(pipeline, task: str, results: dict) -> None:
    from unirl.models.bagel.conditions import BagelARConditions

    prompt = {
        "t2t": "Write one vivid sentence about a sunrise over the ocean.",
        "i2t": None,  # pure image -> text (explicit task)
        "it2t": "Describe this image in detail.",
    }[task]
    try:
        log(f"=== {task} ===")
        req = ar_req(task, prompt, with_image=task in ("i2t", "it2t"))
        t0 = time.time()
        with torch.no_grad():
            resp = pipeline.generate(req)
        track = resp.tracks["ar"]
        txt = track.decoded.texts[0]
        log(f"{task} output ({time.time() - t0:.1f}s, {len(txt)} chars): {txt[:200]!r}")

        # Replay parity: same weights, same temperature → small |dlogp|.
        conditions = BagelARConditions.from_dict(track.conditions)
        with torch.no_grad():
            new_logp = pipeline.ar.replay(conditions, segment=track.segment, temperature=AR_TEMPERATURE)
        old_logp = track.segment.log_probs.to(new_logp.device)
        d = (new_logp - old_logp).abs()
        p95 = torch.quantile(d.float(), 0.95).item()
        n_over = int((d > 5e-2).sum().item())
        log(f"{task} replay parity: median|dlogp|={d.median().item():.2e} p95={p95:.2e} "
            f"max|dlogp|={d.max().item():.2e} ({n_over}/{d.numel()} tok >5e-2)")
        # bf16 rollout (q_len=1 per-token flash-varlen) vs replay (q_len=n teacher-forced)
        # diverge up to ~0.2-0.3 nats on a handful of low-confidence tokens; the bulk
        # (median, printed p95) is the ratio-consistency signal. The image modes'
        # fp32-trajectory replays are bit-exact (ratio==0), proving the path — so the
        # text deltas are bf16 noise, not a bug. Median floats up to ~1e-2 on short,
        # sampling-variant rollouts; assert median < 2e-2 + a loose max sanity bound.
        ok = (bool(txt and txt.strip())
              and d.median().item() < 2e-2
              and d.max().item() < 5e-1)

        if GRAD_SMOKE:
            lm = pipeline.bundle.transformer
            emb = lm.model.embed_tokens.weight
            emb.requires_grad_(True)
            try:
                with torch.enable_grad():
                    lp = pipeline.ar.replay(conditions, segment=track.segment, temperature=AR_TEMPERATURE)
                    lp.sum().backward()
                g = emb.grad
                assert g is not None and g.abs().sum() > 0, "no grad reached embed_tokens"
                log(f"{task} grad smoke: |grad|_1={g.abs().sum().item():.3e} OK")
            finally:
                emb.grad = None
                emb.requires_grad_(False)

        results[task] = ("PASS" if ok else "FAIL",
                         f"median|dlogp|={d.median().item():.2e} max={d.max().item():.2e} text={txt[:60]!r}")
    except Exception as e:  # noqa: BLE001
        log(f"{task} FAILED: {e}\n{traceback.format_exc()}")
        results[task] = ("FAIL", f"{type(e).__name__}: {e}")


def greedy_cross_check(pipeline, results: dict) -> None:
    """Our decode_text (T<=0) vs pristine generate_text(do_sample=False): same ids."""
    from unirl.models.bagel import rl_ops
    from unirl.models.bagel.ar import BagelARStep

    try:
        log("=== greedy cross-check (decode_text vs pristine generate_text) ===")
        bundle = pipeline.bundle
        bagel = bundle.model
        device = torch.device(bundle.device)
        prompt = "What is the capital of France?"
        ntk = bundle.new_token_ids
        n_max = 32

        # Pristine path: vendored context build + generate_text (greedy).
        inf = bundle.inferencer
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            ctx_v = inf.update_context_text(prompt, inf.init_gen_context())
            gi = bagel.prepare_start_tokens(ctx_v["kv_lens"], ctx_v["ropes"], ntk)
            vend = bagel.generate_text(
                past_key_values=ctx_v["past_key_values"],
                max_length=n_max,
                do_sample=False,
                end_token_id=ntk["eos_token_id"],
                **gi,
            )
        vend_ids = [int(t) for t in vend[:, 0].tolist()][1:]  # drop the BOS input token

        # Our path: same RAW split → rl_ops prefill + greedy decode_text.
        ids = [ntk["bos_token_id"]] + bundle.tokenizer.encode(prompt) + [ntk["eos_token_id"]]
        step = BagelARStep(temperature=0.0)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            ctx = rl_ops.init_und_context(bagel)
            ctx = rl_ops.prefill_text_split(bagel, ctx, text_ids=torch.tensor(ids, dtype=torch.long), device=device)
            ours, _ = rl_ops.decode_text(
                bagel, ctx,
                start_token_id=int(ntk["bos_token_id"]),
                sample_fn=step.step,
                max_new_tokens=n_max,
                stop_ids=[int(ntk["eos_token_id"])],
                device=device,
            )
        # Vendored seq excludes the stop token (break before append); ours includes it.
        ours_cmp = ours[:-1] if ours and ours[-1] == int(ntk["eos_token_id"]) else ours
        match = ours_cmp == vend_ids[: len(ours_cmp)]
        log(f"greedy ids ours[:8]={ours[:8]} vendored[:8]={vend_ids[:8]} match={match}")
        results["greedy-xcheck"] = ("PASS" if match else "FAIL", f"ours={ours[:12]} vend={vend_ids[:12]}")
    except Exception as e:  # noqa: BLE001
        log(f"greedy cross-check FAILED: {e}\n{traceback.format_exc()}")
        results["greedy-xcheck"] = ("FAIL", f"{type(e).__name__}: {e}")


def run_image_mode(pipeline, task: str, results: dict) -> None:
    from unirl.models.bagel.conditions import BagelDiffusionConditions
    from unirl.types.sampling import get_diffusion_params

    prompt = {
        "t2i": "A photorealistic red panda sitting on a mossy rock in a misty forest.",
        "it2i": "Change the background to a golden sunset sky.",
    }[task]
    try:
        log(f"=== {task} ===")
        req = image_req(task, prompt, with_image=task == "it2i", pipeline=pipeline)
        params = get_diffusion_params(req.sampling_params)

        torch.manual_seed(1234)
        t0 = time.time()
        with torch.no_grad():
            resp = pipeline.generate(req)
        track = resp.tracks["image"]
        px = track.decoded.pixels
        outp = os.path.join(OUT_DIR, f"{task}.png")
        track.decoded.to_list()[0].to_pil().save(outp)
        log(f"{task} output ({time.time() - t0:.1f}s): pixels {tuple(px.shape)} "
            f"min {px.min().item():.3f} max {px.max().item():.3f} -> {outp}")
        ok = px.ndim == 4 and px.shape[0] == 1 and px.shape[1] == 3

        # Determinism: identical second run → bit-equal trajectory latents.
        detail = ""
        if task == "it2i":
            req2 = image_req(task, prompt, with_image=True, pipeline=pipeline)
            torch.manual_seed(1234)
            with torch.no_grad():
                resp2 = pipeline.generate(req2)
            same = torch.equal(track.segment.latents, resp2.tracks["image"].segment.latents)
            log(f"{task} determinism (two identical runs, same seed): {'bit-equal' if same else 'MISMATCH'}")
            ok = ok and same
            detail += f"deterministic={same} "

        # Replay ratio: recompute SDE log-probs over the stored fp32 trajectory.
        conditions = BagelDiffusionConditions.from_dict(track.conditions)
        with torch.no_grad():
            rep = pipeline.diffusion.replay(conditions, segment=track.segment, params=params)
        d = (rep.log_probs.to(track.segment.sde_logp.device) - track.segment.sde_logp).abs()
        log(f"{task} replay ratio: max|new_logp - sde_logp| = {d.max().item():.3e}")
        ok = ok and d.max().item() < 1e-4
        detail += f"max|dlogp|={d.max().item():.2e} -> {outp}"
        results[task] = ("PASS" if ok else "FAIL", detail)
    except Exception as e:  # noqa: BLE001
        log(f"{task} FAILED: {e}\n{traceback.format_exc()}")
        results[task] = ("FAIL", f"{type(e).__name__}: {e}")


def run() -> int:
    log(f"torch {torch.__version__}; GPUs {torch.cuda.device_count()}; modes {MODES}")
    log(f"steps={STEPS} hw={HW} max_tokens={MAX_TOKENS} grad_smoke={GRAD_SMOKE}")
    need_vit = any(m in MODES for m in ("i2t", "it2t", "it2i"))
    try:
        pipeline = build_pipeline(enable_vit=need_vit)
    except Exception as e:  # noqa: BLE001
        log(f"MODEL LOAD FAILED: {e}\n{traceback.format_exc()}")
        return 2

    results: dict = {}
    for task in ("t2t", "i2t", "it2t"):
        if task in MODES:
            run_text_mode(pipeline, task, results)
    if "t2t" in MODES:
        greedy_cross_check(pipeline, results)
    for task in ("t2i", "it2i"):
        if task in MODES:
            run_image_mode(pipeline, task, results)

    log("================ SUMMARY ================")
    for m in ("t2t", "i2t", "it2t", "greedy-xcheck", "t2i", "it2i"):
        if m in results:
            status, detail = results[m]
            log(f"{m:13s}: {status:5s} | {detail}")
    log("========================================")
    n_pass = sum(1 for v in results.values() if v[0] == "PASS")
    log(f"{n_pass}/{len(results)} checks PASS")
    return 0 if results and n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(run())
