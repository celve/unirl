#!/usr/bin/env python3
"""Runtime oracle for the Sample → Sample rollout-engine refactor (LIN-454).

Exercises the PURE Sample-assembly logic the converted engines rely on — the
path-id lineage, the part-extraction routing, ``assemble_sample`` fill, the σ
round-trip verifier, and the OD-2 noise-key derivation — with fabricated
Samples + ``SimpleNamespace`` fakes (no GPU, no backend, no vllm/sglang). It
does NOT boot an engine; it guards the conversion's structural contracts so a
broken fork / fill / route surfaces here instead of mid-rollout.

Only needs the unirl runtime types (torch) — the imported ``vllm_omni.utils``
helpers pull in no engine runtime. Run in the engine venv:

    python scripts/check_sample_roundtrip.py

Standalone ``main() -> int``; exits non-zero on the first failed contract.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Callable, Tuple

import torch

from unirl.rollout.engine.sigma_verify import verify_engine_used_sigmas
from unirl.rollout.engine.vllm_omni.utils import (
    ar_gen_part,
    assemble_sample,
    diffusion_gen_part,
    image_input_part,
    texts_from_sample,
)
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _expect_raises(fn: Callable[[], object], exc: type, msg: str) -> None:
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"{msg}: expected {exc.__name__} but none was raised")


def _two_stage_sample() -> Tuple[Sample, Part, Part, Part]:
    """``[input(P=2), ar(P*N, N=1), image(P*N*M, M=1)]`` — the HI3 t2i shape."""
    inp = Part.input(["p0", "p1"], primitive=Texts(texts=["a cat", "a dog"]))
    ar_shell = inp.fork(1, sampling_params=ARSamplingParams())
    img_shell = ar_shell.fork(
        1, sampling_params=DiffusionSamplingParams(num_inference_steps=4, height=256, width=256)
    )
    return Sample(parts=[inp, ar_shell, img_shell]), inp, ar_shell, img_shell


def check_part_extraction() -> None:
    """The part-extraction helpers route by ``sampling_params`` type / primitive."""
    sample, _inp, ar_shell, img_shell = _two_stage_sample()
    _check(ar_gen_part(sample) is ar_shell, "ar_gen_part must return the AR shell")
    _check(diffusion_gen_part(sample) is img_shell, "diffusion_gen_part must return the image shell")
    _check(image_input_part(sample) is None, "image_input_part must be None (no Images input Part present)")
    _check(list(texts_from_sample(sample).texts) == ["a cat", "a dog"], "texts_from_sample returns input prompts")

    # A prompt/gen count mismatch must fail fast (mis-forked Sample).
    mismatched = Sample(
        parts=[
            Part.input(["p0"], primitive=Texts(texts=["x", "y"])),
            Part.input(["p0"], primitive=Texts(texts=["x", "y"])).fork(1, sampling_params=ARSamplingParams()),
        ]
    )
    _expect_raises(lambda: texts_from_sample(mismatched), ValueError, "prompt/gen count mismatch must raise")


def check_assemble_sample() -> None:
    """``assemble_sample`` fills the right gen Parts (by params type), preserves ids."""
    sample, inp, _ar_shell, _img_shell = _two_stage_sample()
    ar_seg, img_seg = SimpleNamespace(tag="ar_seg"), SimpleNamespace(tag="img_seg")
    ar_dec, img_dec = Texts(texts=["pe0", "pe1"]), SimpleNamespace(tag="img_dec")
    cond = {"text": SimpleNamespace(tag="cond")}

    out = assemble_sample(
        sample,
        segments_for_track={"ar": ar_seg, "image": img_seg},
        decoded_for_track={"ar": ar_dec, "image": img_dec},
        conditions=cond,
    )
    _check(len(out.parts) == 3, "assemble_sample must preserve the part count")
    out_inp, out_ar, out_img = out.parts
    _check(out_inp is inp, "the input Part passes through unchanged")
    _check(out_ar.segment is ar_seg and out_ar.primitive is ar_dec, "ar Part filled by the 'ar' track")
    _check(out_img.segment is img_seg and out_img.primitive is img_dec, "image Part filled by the 'image' track")
    _check(out_ar.conditions == cond and out_img.conditions == cond, "conditions replicate onto every filled Part")
    _check(list(out_ar.sample_ids) == ["p0/0", "p1/0"], "ar path ids preserved")
    _check(list(out_img.sample_ids) == ["p0/0/0", "p1/0/0"], "image path ids preserved")
    _check(isinstance(out_ar.sampling_params, ARSamplingParams), "ar sampling_params preserved through fill")
    _check(isinstance(out_img.sampling_params, DiffusionSamplingParams), "image sampling_params preserved through fill")

    # Single-DiT shape ``[input, image]``: only the diffusion Part is targeted.
    inp_s = Part.input(["p0", "p1"], primitive=Texts(texts=["a", "b"]))
    img_s = inp_s.fork(1, sampling_params=DiffusionSamplingParams(num_inference_steps=4, height=256, width=256))
    single = Sample(parts=[inp_s, img_s])
    out2 = assemble_sample(
        single, segments_for_track={"image": img_seg}, decoded_for_track={"image": img_dec}, conditions={}
    )
    _check(out2.parts[-1].segment is img_seg, "single-DiT: image Part filled")


def check_sigma_roundtrip() -> None:
    """``verify_engine_used_sigmas`` passes on echo == sent, raises on drift / missing."""
    sig = torch.linspace(1.0, 0.0, steps=5)
    verify_engine_used_sigmas(sig.clone(), expected=sig, engine_name="oracle")  # match → no raise

    perturbed = sig.clone()
    perturbed[1] += 0.1
    _expect_raises(
        lambda: verify_engine_used_sigmas(perturbed, expected=sig, engine_name="oracle"),
        RuntimeError,
        "perturbed σ echo must raise",
    )
    _expect_raises(
        lambda: verify_engine_used_sigmas(None, expected=sig, engine_name="oracle"),
        RuntimeError,
        "missing σ echo must raise",
    )
    verify_engine_used_sigmas(None, expected=None, engine_name="oracle")  # expected None → skip, no raise


def check_noise_key_lineage() -> None:
    """OD-2: the x_T noise key is the gen Part's lineage — group id (shared) vs
    sample id (per-sample), both derived from the path-id fork."""
    inp = Part.input(["p0"], primitive=Texts(texts=["x"]))
    ar = inp.fork(2, sampling_params=ARSamplingParams())
    img = ar.fork(2, sampling_params=DiffusionSamplingParams(num_inference_steps=4, height=256, width=256))
    _check(
        list(img.sample_ids) == ["p0/0/0", "p0/0/1", "p0/1/0", "p0/1/1"],
        "fork builds group-by-parent contiguous path ids",
    )
    _check(len(set(img.sample_ids)) == 4, "per-sample noise key (sample_ids) is unique → distinct x_T")
    _check(
        img.group_ids == ["p0/0", "p0/0", "p0/1", "p0/1"],
        "group noise key (group_ids) = parent id → siblings share x_T",
    )


def check_split_concat_roundtrip() -> None:
    """``Sample.split`` → ``Sample.concat`` round-trips ids (chunked_engine_generate core)."""
    sample, *_ = _two_stage_sample()
    groups = sample.split()
    _check(len(groups) == 2, "split yields one Sample per root group")
    merged = Sample.concat(groups)
    _check(
        [list(p.sample_ids) for p in merged.parts] == [list(p.sample_ids) for p in sample.parts],
        "split→concat round-trips the per-part sample ids in order",
    )


def check_noise_recipe_from_sample() -> None:
    """``NoiseRecipe.from_sample`` keys the x_T on the gen Part's lineage
    (sample_ids, or group_ids under ``init_same_noise``) and reads seed + shape off
    its ``DiffusionSamplingParams`` — parity with the engines' ``_resolve_initial_noise``."""
    inp = Part.input(["p0", "p1"], primitive=Texts(texts=["a", "b"]))
    img = inp.fork(
        2,
        sampling_params=DiffusionSamplingParams(
            num_inference_steps=4, height=256, width=256, seed=7,
            init_noise_latent_shape=[16, 32, 32], init_same_noise=False,
        ),
    )
    recipe = NoiseRecipe.from_sample(Sample(parts=[inp, img]))
    _check(recipe.noise_group_ids == list(img.sample_ids), "from_sample keys per-sample ids (init_same_noise=False)")
    _check(recipe.base_seed == 7, "from_sample reads seed off the gen params")
    _check(tuple(recipe.latent_shape or ()) == (16, 32, 32), "from_sample reads init_noise_latent_shape")
    _check(recipe.initial_latents is None, "no img2img segment latents → initial_latents None")

    img_shared = inp.fork(
        2,
        sampling_params=DiffusionSamplingParams(
            num_inference_steps=4, height=256, width=256, seed=7,
            init_noise_latent_shape=[16, 32, 32], init_same_noise=True,
        ),
    )
    shared = NoiseRecipe.from_sample(Sample(parts=[inp, img_shared]))
    _check(
        shared.noise_group_ids == list(img_shared.group_ids),
        "from_sample keys group ids when init_same_noise=True (siblings share x_T)",
    )


def check_generate_fills_frontier() -> None:
    """The pipeline contract: ``generate`` fills the pre-forked frontier and returns
    ``[input, filled-gen]`` — ids + sampling_params preserved, conditions left empty
    (the trainside re-encode path). Mirrors the Sample reconstruction in
    ``SD3Pipeline.generate`` / ``Qwen3Pipeline.generate`` with fake payloads."""
    inp = Part.input(["p0", "p1"], primitive=Texts(texts=["a", "b"]))
    shell = inp.fork(1, sampling_params=DiffusionSamplingParams(num_inference_steps=4, height=256, width=256))
    sample = Sample(parts=[inp, shell])
    seg, dec = SimpleNamespace(tag="latent_seg"), SimpleNamespace(tag="images")
    filled = sample.parts[-1].fill(segment=seg, primitive=dec)
    out = Sample(parts=[*sample.parts[:-1], filled])
    _check(len(out.parts) == 2, "generate returns [input, gen]")
    _check(out.parts[0] is inp, "input Part passes through unchanged")
    gen = out.parts[-1]
    _check(gen.segment is seg and gen.primitive is dec, "frontier filled with segment + decoded")
    _check(list(gen.sample_ids) == ["p0/0", "p1/0"], "frontier path ids preserved")
    _check(gen.conditions == {}, "conditions left empty (replay re-encodes)")
    _check(isinstance(gen.sampling_params, DiffusionSamplingParams), "sampling_params preserved through fill")


_CHECKS: Tuple[Callable[[], None], ...] = (
    check_part_extraction,
    check_assemble_sample,
    check_sigma_roundtrip,
    check_noise_key_lineage,
    check_split_concat_roundtrip,
    check_noise_recipe_from_sample,
    check_generate_fills_frontier,
)


def main() -> int:
    failures = []
    for check in _CHECKS:
        try:
            check()
        except Exception as exc:  # noqa: BLE001 — the oracle reports, doesn't crash
            failures.append(f"{check.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"  ok  {check.__name__}")
    if failures:
        print("check-sample-roundtrip: FAILED", file=sys.stderr)
        for f in failures:
            print(f"  FAIL {f}", file=sys.stderr)
        return 1
    print(f"check-sample-roundtrip: {len(_CHECKS)} contracts hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
