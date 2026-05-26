"""End-to-end smoke for :class:`ComposedRolloutEngine`.

What this validates on a real GPU pod (no training, no Ray, no reward
pipeline — pure rollout):

1. Hydra compose of ``+experiment=smoke_composed_pe`` produces a sane cfg.
2. ``build(cfg.rollout.engine, model_config=...)`` brings up both children:
   the SGLang LLM subprocess (Qwen3-1.7B) and the SGLang diffusion subprocess
   (SD3.5-medium). At ctor time, only the LLM is awake (``sleep_diffusion_on_start``).
3. ``engine.generate(req)`` over P=2 prompts, with
   ``sampling_params=ComposedSamplingParams(ar=ARSamplingParams(samples_per_prompt=3,...),
   diffusion=DiffusionSamplingParams(num_samples_per_prompt=2,...))``, returns a 2-track
   :class:`RolloutResp` with the right cardinalities (6 LLM samples,
   12 diffusion samples), correct lineage (parent_track="llm"), and a
   :class:`LatentSegment` populated on the diffusion track.
4. ``resp.tracks["diffusion"].decoded.pixels`` carries 12 rows — the
   diffusion VAE actually decoded the latents and the primitive flowed
   through the composed engine to the surface.
5. Weight-sync routing: ``engine.init_weights_update_group(stage_ids=[0])``
   reaches the diffusion child only; ``[1]`` reaches the LLM child only.
   Validated via in-script monkey-patched call counters — no real distributed
   init.
6. ``engine.shutdown()`` exits cleanly; the host should show no orphan
   sglang subprocesses afterwards (``pgrep -af sglang``).

Run on a pod with at least one H20 (or 80GB) GPU::

    cd ~/diffusionrl && source .venv/bin/activate
    PRETRAINED_MODEL=/root/diffusionrl/models/local/stable-diffusion-3.5-medium \
    LLM_MODEL=/root/diffusionrl/models/local/Qwen3-1.7B \
      python scripts/smoke_composed_pe_engine.py \
      2>&1 | tee /mnt/gz/logs/smoke-composed-pe-$(date +%m%d).log

Acceptance: script exits 0 with the line ``SMOKE OK`` and ``nvidia-smi`` after
the script returns shows zero residual GPU processes.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from diffusionrl.types.sampling import ARSamplingParams, ComposedSamplingParams, DiffusionSamplingParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("smoke_composed_pe")


# ---------------------------------------------------------------------------
# Phase 0 — Hydra compose
# ---------------------------------------------------------------------------


def _compose_cfg() -> "Any":
    """Compose ``train.yaml`` with ``+experiment=smoke_composed_pe``.

    Resolves interpolations and returns the ``DictConfig`` ready for
    ``build()`` / ``materialize()``.
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from diffusionrl.config import register_all_configs

    register_all_configs()
    repo_root = Path(__file__).resolve().parent.parent
    conf_dir = repo_root / "conf"
    with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        cfg = compose(config_name="train", overrides=["+experiment=smoke_composed_pe"])
    OmegaConf.resolve(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Phase 1 — Engine construction
# ---------------------------------------------------------------------------


def _build_engine(cfg: "Any") -> "Any":
    """Construct the :class:`ComposedRolloutEngine` directly (no Ray).

    Matches the production wiring in ``RolloutActor`` (see
    ``diffusionrl/ray/rollout_actor.py:128``): ``build(cfg.rollout.engine,
    device=..., strategy=..., rank=..., model_config=materialize(cfg.model))``.
    The composed engine forwards the four runtime kwargs to each child.
    """
    import torch

    from diffusionrl.config import build, materialize

    require_cuda = torch.cuda.is_available()
    if not require_cuda:
        raise RuntimeError(
            "smoke_composed_pe_engine: CUDA is required for SGLang subprocesses. "
            "Run this on a GPU pod, not on a CPU host."
        )
    device = torch.device("cuda:0")
    strategy = build(cfg.sampling.sde_strategy)
    model_config = materialize(cfg.model)

    logger.info(
        "Building ComposedRolloutEngine (LLM=%s, diffusion=%s)",
        cfg.rollout.engine.children.llm.pretrained_model_ckpt_path,
        model_config.pretrained_model_ckpt_path,
    )
    engine = build(
        cfg.rollout.engine,
        device=device,
        strategy=strategy,
        rank=0,
        model_config=model_config,
    )
    logger.info("Engine built. Children: %s", sorted(engine._child_by_name.keys()))
    return engine


# ---------------------------------------------------------------------------
# Phase 2 — Build a 2-prompt RolloutReq with PE knobs (P=2, N=3, M=2)
# ---------------------------------------------------------------------------


def _build_req() -> "Any":
    from diffusionrl.types.primitives import Texts
    from diffusionrl.types.rollout_req import RolloutReq

    prompts = [
        "a cat sitting on a tiled roof at sunset",
        "a single red apple on a wooden table",
    ]
    return RolloutReq(
        sample_ids=[f"p{i}" for i in range(len(prompts))],
        group_ids=[f"g{i}" for i in range(len(prompts))],
        primitives={"text": Texts(texts=prompts)},
        request_conditions={},
        sampling_params=ComposedSamplingParams(
            ar=ARSamplingParams(samples_per_prompt=3, max_new_tokens=64),
            diffusion=DiffusionSamplingParams(
                num_samples_per_prompt=2,
                height=256,
                width=256,
                # 20 steps is enough for visually-recognizable SD3.5 output at
                # 256x256 (default is 28). 4 steps was used in the original
                # plumbing-only smoke and produced unrecognizable noise.
                num_inference_steps=20,
                guidance_scale=7.5,
                eta=1.0,
                seed=0,
            ),
        ),
        sigmas=None,
    )


# ---------------------------------------------------------------------------
# Phase 3 — Resp-shape assertions
# ---------------------------------------------------------------------------


def _assert_resp_shape(resp: "Any", *, p: int, n: int, m: int) -> None:
    """Verify the 2-track response cardinalities + lineage.

    Cardinalities: LLM has P*N samples (one per PE candidate), diffusion has
    P*N*M samples (M images per PE). Lineage: diffusion's parent_track is
    "llm"; LLM's parent_track is None (root). Diffusion's segment is a
    LatentSegment; the decoded primitive (Optional[Images] post-LIN-284
    Step 7) must carry P*N*M pixel rows.
    """
    from diffusionrl.types.segments.latent import LatentSegment

    pn = p * n
    pnm = pn * m

    assert set(resp.tracks.keys()) == {"llm", "diffusion"}, (
        f"expected tracks={{'llm','diffusion'}}, got {sorted(resp.tracks.keys())}"
    )

    llm_track = resp.tracks["llm"]
    diff_track = resp.tracks["diffusion"]

    assert llm_track.parent_track is None, f"LLM root track must have parent_track=None, got {llm_track.parent_track!r}"
    assert diff_track.parent_track == "llm", f"diffusion parent_track must be 'llm', got {diff_track.parent_track!r}"

    assert len(llm_track.sample_ids) == pn, f"LLM track: expected {pn} samples (P*N), got {len(llm_track.sample_ids)}"
    assert len(diff_track.sample_ids) == pnm, (
        f"diffusion track: expected {pnm} samples (P*N*M), got {len(diff_track.sample_ids)}"
    )

    assert isinstance(diff_track.segment, LatentSegment), (
        f"diffusion segment must be LatentSegment, got {type(diff_track.segment).__name__}"
    )

    # Post-flatten: ``decoded`` is ``Optional[Decoded]`` (a primitive, not a
    # dict). For the diffusion track we expect the SGLang sgl-diffusion
    # response builder to populate it with ``Images``.
    diff_decoded = diff_track.decoded
    assert diff_decoded is not None, "diffusion track has no decoded primitive"
    pixels = getattr(diff_decoded, "pixels", None)
    assert pixels is not None, "diffusion decoded.pixels is None"
    assert int(pixels.shape[0]) == pnm, f"diffusion pixels first dim: expected {pnm}, got {int(pixels.shape[0])}"
    logger.info("Resp shape OK: LLM=%d, diff=%d (P=%d, N=%d, M=%d)", pn, pnm, p, n, m)


# ---------------------------------------------------------------------------
# Phase 4 — Weight-sync routing test (no real RPC, just call delivery)
# ---------------------------------------------------------------------------


def _assert_weight_sync_routing(engine: "Any") -> None:
    """Patch counters on each child's ``init_weights_update_group`` and assert
    routing delivers exactly one call per ``stage_ids=[s]`` invocation, to
    the right child.

    Skips the real RPC (which would require a live trainer-side rank to
    handshake). The contract this verifies is the parent's stage_ids →
    child mapping, which is the only piece composed_pe adds — children
    handle the actual RPC the same way they do under any other parent.
    """
    counts = {"llm": 0, "diffusion": 0}

    def _patch(child: "Any", name: str) -> None:
        # Replace bound method with a no-op counter.
        def _wrapped(**kw: Any) -> None:
            counts[name] += 1
            return None

        child.init_weights_update_group = _wrapped

    _patch(engine._llm, "llm")
    _patch(engine._diffusion, "diffusion")

    # stage_ids=[0] → diffusion only
    engine.init_weights_update_group(
        master_address="127.0.0.1",
        master_port=29500,
        rank_offset=0,
        world_size=1,
        group_name="g_diff_smoke",
        stage_ids=[0],
    )
    assert counts == {"llm": 0, "diffusion": 1}, f"stage_ids=[0] routing failed: {counts!r}"

    # stage_ids=[1] → llm only
    engine.init_weights_update_group(
        master_address="127.0.0.1",
        master_port=29501,
        rank_offset=0,
        world_size=1,
        group_name="g_llm_smoke",
        stage_ids=[1],
    )
    assert counts == {"llm": 1, "diffusion": 1}, f"stage_ids=[1] routing failed: {counts!r}"

    # No stage_ids → must raise (no implicit broadcast)
    raised = False
    try:
        engine.init_weights_update_group(
            master_address="127.0.0.1",
            master_port=29502,
            rank_offset=0,
            world_size=1,
            group_name="g_implicit",
            stage_ids=None,
        )
    except ValueError:
        raised = True
    assert raised, "stage_ids=None must raise ValueError (no implicit broadcast)"
    assert counts == {"llm": 1, "diffusion": 1}, f"stage_ids=None must not deliver any call: {counts!r}"

    logger.info("Weight-sync routing OK: stage 0 → diffusion, stage 1 → llm, None → raise")


# ---------------------------------------------------------------------------
# Optional artifact dump (env-gated via SMOKE_OUTPUT_DIR)
# ---------------------------------------------------------------------------


def _dump_artifacts(req: "Any", resp: "Any", *, out_dir: "Path") -> None:
    """Persist the PE texts + decoded images for inspection / archival.

    Lays out::

        <out_dir>/
            request.json     # original prompts + sampling_params
            pe_texts.json    # 6 PE candidates from the LLM track
            images/<sid>.png # 12 PNGs, one per diffusion sample_id
            META.txt         # shape + pixel-range summary

    Pixel range: detected at runtime — SD3.5 via SGLang's sgl-diffusion
    pipeline can return ``Images.pixels`` in either ``[0, 1]`` (post-VAE
    normalized) or ``[-1, 1]`` (raw VAE-tanh out) depending on which post-
    processing path the pipeline takes. We log the observed min/max/mean
    and pick the mapping by sign of min: ``min < -0.05`` → assume
    ``[-1, 1]`` and remap ``(x+1)/2``; else clamp+scale as-is.
    """
    import json

    import torch
    from PIL import Image as PILImage

    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)

    # 1. Request snapshot — useful to know what produced these artifacts.
    import dataclasses

    request_dump = {
        "sample_ids": list(req.sample_ids),
        "prompts": list(req.primitives["text"].texts),
        "sampling_params": dataclasses.asdict(req.sampling_params) if req.sampling_params is not None else None,
    }
    (out_dir / "request.json").write_text(json.dumps(request_dump, indent=2))

    # 2. PE texts — the LLM track's decoded ``Texts`` primitive.
    llm_track = resp.tracks["llm"]
    pe_texts = llm_track.decoded.texts
    pe_dump = [
        {
            "sample_id": llm_track.sample_ids[i],
            "parent_id": (llm_track.parent_ids[i] if llm_track.parent_ids else None),
            "text": pe_texts[i],
        }
        for i in range(len(llm_track.sample_ids))
    ]
    (out_dir / "pe_texts.json").write_text(json.dumps(pe_dump, indent=2, ensure_ascii=False))

    # 3. Diffusion images — one PNG per sample, with range auto-detection.
    diff_track = resp.tracks["diffusion"]
    pixels_raw = diff_track.decoded.pixels  # [B, C, H, W], float
    pixels_raw = pixels_raw.detach().to("cpu", dtype=torch.float32)

    p_min = float(pixels_raw.min())
    p_max = float(pixels_raw.max())
    p_mean = float(pixels_raw.mean())
    p_std = float(pixels_raw.std())
    sys.stderr.write(f">>> pixel stats: min={p_min:.3f}, max={p_max:.3f}, mean={p_mean:.3f}, std={p_std:.3f}\n")
    sys.stderr.flush()

    if p_min < -0.05:
        pixels = (pixels_raw + 1.0) / 2.0
        range_note = "[-1, 1] -> [0, 1]"
    else:
        pixels = pixels_raw
        range_note = "already [0, 1]"
    sys.stderr.write(f">>> pixel range mapping: {range_note}\n")
    sys.stderr.flush()
    pixels = pixels.clamp(0.0, 1.0)
    pixels = (pixels * 255.0).round().to(torch.uint8)  # [B, C, H, W] uint8
    pixels = pixels.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]

    for i, sid in enumerate(diff_track.sample_ids):
        # Hierarchical IDs contain ``/`` — flatten for filenames.
        fname = sid.replace("/", "_") + ".png"
        arr = pixels[i].numpy()
        PILImage.fromarray(arr, mode="RGB").save(img_dir / fname)

    # 4. Summary file the README can reference.
    meta_lines = [
        f"P=2, N=3, M=2 → {len(llm_track.sample_ids)} PE candidates, {len(diff_track.sample_ids)} images",
        f"image shape: {tuple(diff_track.decoded.pixels.shape)}",
        f"image dtype: {diff_track.decoded.pixels.dtype}",
        f"pixel range raw: min={p_min:.3f}, max={p_max:.3f}, mean={p_mean:.3f}, std={p_std:.3f}",
        f"pixel range mapping: {range_note}",
        f"diffusion parent_track: {diff_track.parent_track!r}",
    ]
    (out_dir / "META.txt").write_text("\n".join(meta_lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    cfg = _compose_cfg()
    engine = _build_engine(cfg)
    try:
        req = _build_req()
        logger.info("Generating: P=2, N=3, M=2 → 6 PE candidates, 12 images")
        try:
            resp = engine.generate(req)
        except Exception as exc:
            import traceback

            sys.stderr.write(f"\n>>> engine.generate RAISED: {type(exc).__name__}: {exc}\n")
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            raise
        sys.stderr.write(
            f"\n>>> generate returned: tracks={sorted(resp.tracks.keys())}, "
            f"llm.n={len(resp.tracks.get('llm').sample_ids) if 'llm' in resp.tracks else 'NA'}, "
            f"diff.n={len(resp.tracks.get('diffusion').sample_ids) if 'diffusion' in resp.tracks else 'NA'}\n"
        )
        sys.stderr.flush()
        _assert_resp_shape(resp, p=2, n=3, m=2)
        sys.stderr.write(">>> _assert_resp_shape OK\n")
        sys.stderr.flush()
        _assert_weight_sync_routing(engine)
        sys.stderr.write(">>> _assert_weight_sync_routing OK\n")
        sys.stderr.flush()
        # Optional artifact dump — env-gated so the smoke is still cheap by default.
        out_dir = os.environ.get("SMOKE_OUTPUT_DIR")
        if out_dir:
            _dump_artifacts(req, resp, out_dir=Path(out_dir))
            sys.stderr.write(f">>> dumped artifacts to {out_dir}\n")
            sys.stderr.flush()
    except Exception as exc:
        import traceback

        sys.stderr.write(f"\n>>> main RAISED: {type(exc).__name__}: {exc}\n")
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
    finally:
        logger.info("engine.shutdown()")
        try:
            engine.shutdown()
        except Exception as exc:
            logger.warning("shutdown raised: %s", exc)

    print("SMOKE OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
