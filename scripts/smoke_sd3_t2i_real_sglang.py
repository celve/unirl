"""Real-runtime SD3 smoke for the new-protocol SGLang rollout engine.

Boots :class:`diffusionrl.rollout.engine.sglang.engine.SGLangRolloutEngine`
against a local SD3.5-medium checkpoint and exercises four end-to-end
scenarios:

  1. **basic_ode** — ODE-mode generate, asserts shape/dtype of every
     ``RolloutResp`` slot (rollout_traces, decoded, conditions).
  2. **init_latents_passthrough** — SDE mode with caller-supplied
     ``request_conditions['initial_latents']``; asserts the engine
     forwards the tensor verbatim to SGLang and it lands at
     ``trajectory_latents[:, 0]``.
  3. **sde_replay_logp** — SDE mode with ``cfg.logprob_source='replay'``;
     asserts BOTH ``LatentSegment.sde_logp`` AND ``sde_indices`` stay
     ``None``. Replay-mode rollout traces are produced for inference / eval only
     and are intentionally NOT consumable by trainer-side replay.
  4. **sde_native_logp** — SDE mode with ``cfg.logprob_source='native'``;
     asserts ``LatentSegment.sde_logp`` is populated as ``[B, num_steps]``
     and ``sde_indices`` carries the per-step transition index list.
  5. **trainer_replay** — feeds the scenario-4 native-mode ``RolloutResp``
     into a trainer-side :class:`SD3DiffusionStage.replay` (boot a real
     :class:`SD3Bundle` after shutting the SGLang engine down to free GPU
     memory) and asserts the returned ``ReplayResult.log_probs`` has the
     expected ``[B, num_inference_steps]`` shape with finite values. This
     is the round-trip proof that native-mode rollout traces are consumable by
     training code.

Each scenario writes one PNG per sample (4) to
``<out_dir>/scenario_<n>/sid_<i>.png`` for visual inspection.

Run on a single H20:

    python scripts/smoke_sd3_t2i_real_sglang.py

Single-GPU only; no Ray, no actor stack, no reward / training. The plumbing
smoke at ``scripts/smoke_sd3_t2i_replay_sglang.py`` covers the no-runtime
path; this one is the first integration test against a live ``DiffGenerator``.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional

import torch

# Imports below run at module import time. SGLang must be importable; if not,
# the smoke aborts with a clear message before booting the DiffGenerator.
try:
    import sglang  # noqa: F401
except ImportError as exc:
    print(f"[FAIL] sglang import failed: {exc}", flush=True)
    sys.exit(1)

from diffusionrl.models_new.sd3.config import SD3PipelineConfig
from diffusionrl.rollout.engine.sglang.config import SGLangEngineConfig
from diffusionrl.rollout.engine.sglang.engine import SGLangRolloutEngine
from diffusionrl.sde.kernels import FlowSDEStrategy
from diffusionrl.types.conditions.image import ImageLatentCondition
from diffusionrl.types.primitives import Texts
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutResp
from diffusionrl.types.sampling import SamplingParams, SDEConfig

PROMPTS = [
    "a red cat sitting on a green hill at sunset",
    "a blue dragon flying over a futuristic city at night",
]
SAMPLES_PER_PROMPT = 2  # K=2 expansion → 4 samples total
HEIGHT = 512
WIDTH = 512
NUM_INFERENCE_STEPS = 4
GUIDANCE_SCALE = 4.5
ETA = 0.7
SHIFT = 3.0
SD3_LATENT_CHANNELS = 16  # SD3 transformer in_channels
SD3_VAE_SPATIAL_RATIO = 8  # VAE downsamples 8x
SD3_TEXT_HIDDEN = 4096  # T5-XXL hidden dim
LATENT_H = HEIGHT // SD3_VAE_SPATIAL_RATIO
LATENT_W = WIDTH // SD3_VAE_SPATIAL_RATIO


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_cfg(seed: int) -> SGLangEngineConfig:
    sampling = SamplingParams(
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        height=HEIGHT,
        width=WIDTH,
        num_frames=1,
        seed=seed,
        num_samples_per_prompt=SAMPLES_PER_PROMPT,
        sde_config=SDEConfig(eta=ETA),
        sampler_kwargs={},
    )
    return SGLangEngineConfig(
        sampling=sampling,
        model_family="sd3",
        populate_conditions=True,
        init_same_noise=False,
        logprob_source="replay",  # scenario 4 mutates this to "native"
        num_gpus=1,
        tp_size=1,
        local_mode=True,
        disable_autocast=False,
    )


def _make_req(
    *,
    sde_indices: Optional[List[int]],
    seed: int,
    initial_latents: Optional[torch.Tensor] = None,
) -> RolloutReq:
    expanded_prompts: List[str] = []
    sample_ids: List[str] = []
    group_ids: List[str] = []
    for gi, prompt in enumerate(PROMPTS):
        for si in range(SAMPLES_PER_PROMPT):
            expanded_prompts.append(prompt)
            sample_ids.append(f"s{gi}_{si}")
            group_ids.append(f"g{gi}")

    request_conditions = {}
    if initial_latents is not None:
        if int(initial_latents.shape[0]) != len(sample_ids):
            raise ValueError(
                f"initial_latents.shape[0]={int(initial_latents.shape[0])} != sample count {len(sample_ids)}"
            )
        request_conditions["initial_latents"] = ImageLatentCondition(latents=initial_latents)

    diffusion_params = {
        "height": HEIGHT,
        "width": WIDTH,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "eta": ETA,
        "seed": seed,
        "sde_indices": list(sde_indices) if sde_indices is not None else None,
        "num_samples_per_prompt": SAMPLES_PER_PROMPT,
    }

    return RolloutReq(
        sample_ids=sample_ids,
        group_ids=group_ids,
        primitives={"text": Texts(texts=expanded_prompts)},
        request_conditions=request_conditions,
        stage_params={"diffusion": diffusion_params},
    )


def _write_previews(resp: RolloutResp, scenario_dir: Path) -> None:
    """Save each decoded sample as a PNG under ``scenario_dir/sid_<i>.png``."""
    from torchvision.transforms.functional import to_pil_image

    decoded = resp.decoded.get("image")
    if decoded is None or decoded.pixels is None:
        print(f"  [warn] no decoded image to write under {scenario_dir}", flush=True)
        return
    scenario_dir.mkdir(parents=True, exist_ok=True)
    pixels = decoded.pixels.detach().cpu().float().clamp(0.0, 1.0)
    for i in range(int(pixels.shape[0])):
        out_path = scenario_dir / f"{resp.sample_ids[i]}.png"
        to_pil_image(pixels[i]).save(out_path)
    print(f"  wrote {int(pixels.shape[0])} previews to {scenario_dir}", flush=True)


def _check_basic_shape(resp: RolloutResp, *, batch: int) -> None:
    assert list(resp.sample_ids) == [f"s{gi}_{si}" for gi in range(len(PROMPTS)) for si in range(SAMPLES_PER_PROMPT)], (
        f"sample_ids round-trip mismatch: {resp.sample_ids}"
    )
    seg = resp.rollout_traces.get("image")
    assert seg is not None, "resp.rollout_traces['image'] missing"
    expected_latents = (batch, NUM_INFERENCE_STEPS + 1, SD3_LATENT_CHANNELS, LATENT_H, LATENT_W)
    assert tuple(seg.latents.shape) == expected_latents, (
        f"latents shape {tuple(seg.latents.shape)} != expected {expected_latents}"
    )
    assert seg.sigmas is not None and tuple(seg.sigmas.shape) == (NUM_INFERENCE_STEPS + 1,), (
        f"sigmas shape {tuple(seg.sigmas.shape) if seg.sigmas is not None else None}"
    )
    decoded = resp.decoded.get("image")
    assert decoded is not None and decoded.pixels is not None, "decoded.pixels missing"
    expected_decoded = (batch, 3, HEIGHT, WIDTH)
    assert tuple(decoded.pixels.shape) == expected_decoded, (
        f"decoded.pixels shape {tuple(decoded.pixels.shape)} != expected {expected_decoded}"
    )
    assert decoded.pixels.dtype == torch.float32, f"decoded.pixels dtype {decoded.pixels.dtype}"
    pmin = float(decoded.pixels.min().item())
    pmax = float(decoded.pixels.max().item())
    assert 0.0 <= pmin and pmax <= 1.0, f"decoded.pixels out of [0,1]: [{pmin}, {pmax}]"
    text_cond = resp.conditions.get("text")
    assert text_cond is not None and text_cond.embeds is not None, "conditions['text'].embeds missing"
    assert int(text_cond.embeds.shape[0]) == batch, f"text embeds batch {int(text_cond.embeds.shape[0])} != {batch}"
    assert int(text_cond.embeds.shape[-1]) == SD3_TEXT_HIDDEN, (
        f"text embeds hidden {int(text_cond.embeds.shape[-1])} != {SD3_TEXT_HIDDEN}"
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def scenario_1_basic_ode(engine: SGLangRolloutEngine, out_dir: Path, seed: int) -> None:
    name = "scenario_1_basic_ode"
    print(f"\n[run] {name}", flush=True)
    req = _make_req(sde_indices=None, seed=seed)
    t0 = time.perf_counter()
    resp = engine.generate(req)
    dt = time.perf_counter() - t0
    print(f"  generate({len(req.sample_ids)} samples, ODE) took {dt:.1f}s", flush=True)

    _check_basic_shape(resp, batch=len(req.sample_ids))
    seg = resp.rollout_traces["image"]
    assert seg.sde_logp is None, "ODE mode must not pack sde_logp"
    assert seg.sde_indices is None, "ODE mode must not pack sde_indices"

    _write_previews(resp, out_dir / name)
    print(f"[PASS] {name}", flush=True)


def scenario_2_init_latents_passthrough(
    engine: SGLangRolloutEngine,
    out_dir: Path,
    seed: int,
    device: torch.device,
) -> None:
    name = "scenario_2_init_latents_passthrough"
    print(f"\n[run] {name}", flush=True)
    batch = len(PROMPTS) * SAMPLES_PER_PROMPT
    gen = torch.Generator(device=device).manual_seed(0)
    fixed_x_T = torch.randn(
        batch,
        SD3_LATENT_CHANNELS,
        LATENT_H,
        LATENT_W,
        generator=gen,
        device=device,
        dtype=torch.float32,
    )
    req = _make_req(
        sde_indices=list(range(NUM_INFERENCE_STEPS)),
        seed=seed,
        initial_latents=fixed_x_T,
    )
    t0 = time.perf_counter()
    resp = engine.generate(req)
    dt = time.perf_counter() - t0
    print(f"  generate(SDE + initial_latents) took {dt:.1f}s", flush=True)

    _check_basic_shape(resp, batch=batch)
    seg = resp.rollout_traces["image"]
    landed = seg.latents[:, 0].detach().cpu().float()
    expected = fixed_x_T.detach().cpu().float()
    if not torch.allclose(landed, expected, atol=1e-5):
        max_diff = (landed - expected).abs().max().item()
        raise AssertionError(
            f"initial_latents pass-through broken: "
            f"resp.rollout_traces['image'].latents[:, 0] does not match the supplied tensor; "
            f"max abs diff = {max_diff:.3e}. The engine must forward "
            f"req.request_conditions['initial_latents'].latents to SGLang's "
            f"initial_noise kwarg verbatim."
        )

    _write_previews(resp, out_dir / name)
    print(f"[PASS] {name}", flush=True)


def scenario_3_sde_replay_logp(
    engine: SGLangRolloutEngine,
    out_dir: Path,
    seed: int,
) -> None:
    name = "scenario_3_sde_replay_logp"
    print(f"\n[run] {name}", flush=True)
    engine.cfg.logprob_source = "replay"  # default; explicit for clarity
    req = _make_req(sde_indices=list(range(NUM_INFERENCE_STEPS)), seed=seed)
    t0 = time.perf_counter()
    resp = engine.generate(req)
    dt = time.perf_counter() - t0
    print(f"  generate(SDE replay) took {dt:.1f}s", flush=True)

    _check_basic_shape(resp, batch=len(req.sample_ids))
    seg = resp.rollout_traces["image"]
    # Replay-mode rollout traces are NOT consumable by trainer-side replay —
    # ``sde_logp`` and ``sde_indices`` travel together. Use native mode
    # (scenario_4) when the rollout needs to feed training.
    assert seg.sde_logp is None, "replay mode must leave sde_logp None"
    assert seg.sde_indices is None, "replay mode must leave sde_indices None"

    _write_previews(resp, out_dir / name)
    print(f"[PASS] {name}", flush=True)


def scenario_4_sde_native_logp(
    engine: SGLangRolloutEngine,
    out_dir: Path,
    seed: int,
) -> RolloutResp:
    name = "scenario_4_sde_native_logp"
    print(f"\n[run] {name}", flush=True)
    engine.cfg.logprob_source = "native"
    req = _make_req(sde_indices=list(range(NUM_INFERENCE_STEPS)), seed=seed)
    t0 = time.perf_counter()
    resp = engine.generate(req)
    dt = time.perf_counter() - t0
    print(f"  generate(SDE native) took {dt:.1f}s", flush=True)

    _check_basic_shape(resp, batch=len(req.sample_ids))
    seg = resp.rollout_traces["image"]
    assert seg.sde_logp is not None, "native mode must populate sde_logp"
    expected_shape = (len(req.sample_ids), NUM_INFERENCE_STEPS)
    assert tuple(seg.sde_logp.shape) == expected_shape, (
        f"sde_logp shape {tuple(seg.sde_logp.shape)} != expected {expected_shape}"
    )
    assert seg.sde_indices is not None, "native mode must populate sde_indices"
    assert torch.equal(seg.sde_indices, torch.arange(NUM_INFERENCE_STEPS, dtype=torch.long)), (
        f"sde_indices {seg.sde_indices.tolist()} != arange({NUM_INFERENCE_STEPS})"
    )
    assert torch.isfinite(seg.sde_logp).all(), "sde_logp contains non-finite values"

    _write_previews(resp, out_dir / name)
    print(f"[PASS] {name}", flush=True)
    return resp


def scenario_5_trainer_replay(
    native_resp: RolloutResp,
    model_path: Path,
    device: torch.device,
) -> None:
    """Round-trip: feed scenario_4's native-mode ``RolloutResp`` into SD3DiffusionStage.replay.

    Trainer-side replay requires ``segment.sde_indices`` non-None, which
    is populated only when ``cfg.logprob_source='native'`` — so the input
    here is the scenario-4 response, not scenario-3's replay-mode one
    (whose segment is intentionally not training-consumable).

    Requires the SGLang engine to be shut down first (frees ~16 GB on the
    GPU; replay then loads the SD3Bundle locally). Asserts the stage
    consumes ``LatentSegment.sde_indices`` / ``latents`` / ``sigmas`` and
    ``RolloutResp.conditions`` via ``SD3Conditions.from_dict`` without a
    typed-container round-trip on the rollout side. ``log_probs`` shape
    must be ``[B, num_inference_steps]`` and all finite.
    """
    name = "scenario_5_trainer_replay"
    print(f"\n[run] {name}", flush=True)

    # Import here so that the heavy SD3 trainer-side modules don't load
    # before the engine boots (scenarios 1-4 don't need them).
    from diffusionrl.models_new.sd3.bundle import SD3Bundle
    from diffusionrl.models_new.sd3.conditions import SD3Conditions
    from diffusionrl.models_new.sd3.config import SD3PipelineConfig
    from diffusionrl.models_new.sd3.diffusion import (
        SD3DiffusionParams,
        SD3DiffusionStage,
        SD3DiffusionStep,
    )
    from diffusionrl.sde.kernels import FlowSDEStrategy

    seg = native_resp.rollout_traces["image"]
    batch = int(seg.latents.shape[0])
    if seg.sde_indices is None:
        raise AssertionError(
            "scenario_5: native_resp.rollout_traces['image'].sde_indices is None — "
            "scenario_4 must run in native-logprob mode to populate sde_indices."
        )

    print(f"  loading SD3Bundle from {model_path} on {device}", flush=True)
    t0 = time.perf_counter()
    bundle = SD3Bundle.from_config(
        SD3PipelineConfig(
            pretrained_model_ckpt_path=str(model_path),
            device=device,
        )
    )
    print(f"  bundle loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    step = SD3DiffusionStep()
    strategy = FlowSDEStrategy()
    strategy.init_schedule(seg.sigmas.to(device))
    stage = SD3DiffusionStage(
        model=bundle,
        step=step,
        strategy=strategy,
        autocast_precision="bf16",
        trajectory_precision="fp16",
        logprob_precision="fp32",
        vae_scale_factor=SD3_VAE_SPATIAL_RATIO,
        latent_channels=SD3_LATENT_CHANNELS,
    )

    # Move segment tensors to the trainer device (the rollout engine returned
    # everything on CPU). `latents_at` reads .device off `latents`, so this
    # is the single place that matters; sigmas / indices get moved alongside.
    def _to_dev(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return t.to(device) if t is not None else None

    seg_dev = type(seg)(
        latents=_to_dev(seg.latents),
        sigmas=_to_dev(seg.sigmas),
        indices=_to_dev(seg.indices),
        sde_logp=_to_dev(seg.sde_logp),
        sde_indices=_to_dev(seg.sde_indices),
        sample_indices=_to_dev(seg.sample_indices),
    )

    # Move text + negative_text conditions to the trainer device before
    # SD3Conditions.from_dict — SD3DiffusionStep.predict_noise will index
    # them with sigma on `device` so a CPU/GPU mismatch would error.
    cond_dict_dev = {}
    for slot, cond in native_resp.conditions.items():
        cond_dict_dev[slot] = type(cond)(
            embeds=_to_dev(cond.embeds),
            pooled=_to_dev(cond.pooled),
            attn_mask=_to_dev(cond.attn_mask),
        )
    typed_conds = SD3Conditions.from_dict(cond_dict_dev)

    params = SD3DiffusionParams(
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        height=HEIGHT,
        width=WIDTH,
        seed=42,
        sde_indices=list(range(NUM_INFERENCE_STEPS)),
        eta=ETA,
        samples_per_prompt=SAMPLES_PER_PROMPT,
    )

    print("  running stage.replay(...)", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        result = stage.replay(typed_conds, segment=seg_dev, params=params)
    print(f"  replay took {time.perf_counter() - t0:.1f}s", flush=True)

    if result.log_probs is None:
        raise AssertionError("scenario_5: ReplayResult.log_probs is None")
    expected_shape = (batch, NUM_INFERENCE_STEPS)
    if tuple(result.log_probs.shape) != expected_shape:
        raise AssertionError(
            f"scenario_5: log_probs shape {tuple(result.log_probs.shape)} != expected {expected_shape}"
        )
    if not torch.isfinite(result.log_probs).all():
        raise AssertionError(
            "scenario_5: replay log_probs contain non-finite values; "
            "either the segment latents lost precision in CPU↔GPU round-trip "
            "or the strategy returned NaN."
        )
    print(
        f"  log_probs shape={tuple(result.log_probs.shape)} "
        f"mean={float(result.log_probs.mean().item()):.4f} "
        f"min={float(result.log_probs.min().item()):.4f} "
        f"max={float(result.log_probs.max().item()):.4f}",
        flush=True,
    )
    print(f"[PASS] {name}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="/root/diffusionrl/models/local/stable-diffusion-3.5-medium",
        help="Path to a SD3.5-medium diffusers checkpoint (must contain model_index.json).",
    )
    parser.add_argument(
        "--out-dir",
        default="/tmp/lin254_real_sd3_smoke",
        help="Directory for per-scenario PNG previews.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Pre-flight
    model_path = Path(args.model_path)
    if not (model_path / "model_index.json").is_file():
        print(
            f"[FAIL] model_index.json not found under {model_path}. "
            f"Copy SD3.5-medium first (see scripts/smoke_sd3_t2i_real_sglang.py docstring).",
            flush=True,
        )
        return 1
    if not torch.cuda.is_available():
        print("[FAIL] torch.cuda.is_available() is False; smoke needs a GPU.", flush=True)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    cfg = _build_cfg(args.seed)
    model_config = SD3PipelineConfig(pretrained_model_ckpt_path=str(model_path))
    strategy = FlowSDEStrategy()

    print(f"=== boot DiffGenerator + SD3.5-medium from {model_path} ===", flush=True)
    boot_t0 = time.perf_counter()
    engine = SGLangRolloutEngine(
        cfg,
        device=device,
        strategy=strategy,
        rank=0,
        model_config=model_config,
    )
    print(f"  boot took {time.perf_counter() - boot_t0:.1f}s", flush=True)

    failed: List[str] = []
    # Captured scenario_4 native-mode response for scenario_5 (trainer-side
    # replay needs ``segment.sde_indices`` populated, which only native mode
    # produces).
    native_resp_for_trainer: Optional[RolloutResp] = None
    try:
        try:
            scenario_1_basic_ode(engine, out_dir, args.seed)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] scenario_1_basic_ode: {exc}", flush=True)
            traceback.print_exc()
            failed.append("scenario_1_basic_ode")
        try:
            scenario_2_init_latents_passthrough(engine, out_dir, args.seed, device)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] scenario_2_init_latents_passthrough: {exc}", flush=True)
            traceback.print_exc()
            failed.append("scenario_2_init_latents_passthrough")
        try:
            scenario_3_sde_replay_logp(engine, out_dir, args.seed)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] scenario_3_sde_replay_logp: {exc}", flush=True)
            traceback.print_exc()
            failed.append("scenario_3_sde_replay_logp")
        try:
            native_resp_for_trainer = scenario_4_sde_native_logp(engine, out_dir, args.seed)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] scenario_4_sde_native_logp: {exc}", flush=True)
            traceback.print_exc()
            failed.append("scenario_4_sde_native_logp")
    finally:
        print("\n=== shutdown ===", flush=True)
        try:
            engine.shutdown()
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] engine.shutdown raised: {exc}", flush=True)
        # Make sure the engine's GPU allocations are released before scenario_5
        # tries to load the SD3Bundle on the same device.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Scenario 5 runs OUTSIDE the engine-owning try/finally: it needs the
    # engine GPU memory freed first. Skipped if scenario_4 didn't produce a
    # native-mode resp.
    if native_resp_for_trainer is not None:
        try:
            scenario_5_trainer_replay(native_resp_for_trainer, model_path, device)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] scenario_5_trainer_replay: {exc}", flush=True)
            traceback.print_exc()
            failed.append("scenario_5_trainer_replay")
    else:
        print("[skip] scenario_5_trainer_replay (scenario_4 did not produce a resp)", flush=True)

    if failed:
        print(f"\nSCENARIOS FAILED: {failed}", flush=True)
        return 3
    print("\nALL SCENARIOS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
