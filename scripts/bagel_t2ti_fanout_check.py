#!/usr/bin/env python3
"""Bagel t2ti N×M fan-out lineage CPU check — no GPU, no model, no flash_attn.

Drives ``BagelPipeline._generate_t2ti`` with FAKE stages (AR autoregress, think-
context build, diffuse+decode) to verify the fan-out bookkeeping the method owns:

    P prompts ──N captions/prompt──▶ P*N AR captions   (root "ar", group by prompt)
              ──M images/caption───▶ P*N*M images       ("image", parent_track="ar")

Asserts: group-by-parent contiguous lineage on both tracks, the
``image s ← caption a=s//M ← prompt a//N`` index map into the diffusion contexts,
distinct per-image x_T ids, and that ``propagate_rewards`` / ``compute_advantages``
accept the produced tracks. The real AR + diffusion numerics are covered by
``bagel_ar_cpu_check.py`` and the GPU smoke; this isolates the fan-out indexing.

Run:  python3 scripts/bagel_t2ti_fanout_check.py
"""
from __future__ import annotations

import sys
import types

# Flash-free + vendor-free: poison flash_attn and stub the vendored inferencer
# (the only vendor symbol _generate_t2ti imports) BEFORE importing unirl.
sys.modules.setdefault("flash_attn", None)
_vendor = types.ModuleType("unirl.models.bagel.vendor")
_inferencer = types.ModuleType("unirl.models.bagel.vendor.inferencer")
_inferencer.GEN_THINK_SYSTEM_PROMPT = "PLAN."
_vendor.inferencer = _inferencer
sys.modules["unirl.models.bagel.vendor"] = _vendor
sys.modules["unirl.models.bagel.vendor.inferencer"] = _inferencer

import torch

from unirl.models.bagel.conditions import BagelDiffusionConditions
from unirl.models.bagel.diffusion import BagelDiffusionParams
from unirl.models.bagel.pipeline import BagelPipeline
from unirl.types.primitives import Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, _track_with_field
from unirl.types.sampling import ARSamplingParams, ComposedSamplingParams
from unirl.types.segments.text import TextSegment

BOS, EOS = 1, 2


def log(m: str) -> None:
    print(f"[bagel-t2ti-fanout] {m}", flush=True)


class FakeTok:
    def encode(self, s):  # noqa: ANN001
        return [3, 4, 5]

    def decode(self, ids):  # noqa: ANN001
        return f"think={list(ids)}"


class FakeBundle:
    new_token_ids = {"bos_token_id": BOS, "eos_token_id": EOS}
    tokenizer = FakeTok()
    device = torch.device("cpu")


class FakeAR:
    """Returns a varlen TextSegment with one distinct response token per sample."""

    def __init__(self) -> None:
        self.calls = []

    def autoregress(self, conditions, *, sampling_params, **kw):  # noqa: ANN001
        self.calls.append(conditions.batch_size)
        k = conditions.batch_size
        toks = [torch.tensor([100 + i, EOS], dtype=torch.long) for i in range(k)]
        lps = [torch.tensor([-0.1, -0.1], dtype=torch.float32) for _ in range(k)]
        return TextSegment.pack(tokens=toks, log_probs=lps)


def build_pipe(captured: dict) -> BagelPipeline:
    pipe = object.__new__(BagelPipeline)  # skip __init__ (no model load)
    pipe.bundle = FakeBundle()
    pipe.ar = FakeAR()

    def fake_detok(segment):  # noqa: ANN001
        cu = [int(c) for c in segment.cu_seqlens.tolist()]
        return Texts(texts=[f"T{i}" for i in range(len(cu) - 1)])

    def fake_build_think(system_prompt, prompt, think_text):  # noqa: ANN001
        return ("CTX", prompt, think_text)

    def fake_diffuse(contexts, *, params, req, image_shape):  # noqa: ANN001
        captured["contexts"] = list(contexts)
        captured["diff_sample_ids"] = list(req.sample_ids)
        captured["diff_parent_ids"] = list(req.group_ids)
        captured["diff_noise_ids"] = list(req.init_noise_group_ids)
        k = len(contexts)
        conds = BagelDiffusionConditions(
            gen_contexts=list(contexts),
            cfg_text_contexts=[None] * k,
            cfg_img_contexts=[None] * k,
            image_shapes=[image_shape] * k,
        )
        return None, conds, None  # segment/decoded irrelevant to lineage

    pipe._detokenize = fake_detok
    pipe._build_think_contexts = fake_build_think
    pipe._diffuse_and_decode = fake_diffuse
    return pipe


def run(P: int, N: int, M: int) -> None:
    prompts = [f"prompt{p}" for p in range(P)]
    sample_ids = [f"p{p}" for p in range(P)]
    captured: dict = {}
    pipe = build_pipe(captured)
    req = RolloutReq(
        sample_ids=list(sample_ids),
        group_ids=[f"g{p}" for p in range(P)],
        primitives={"text": Texts(texts=prompts)},
        sampling_params=ComposedSamplingParams(
            ar=ARSamplingParams(samples_per_prompt=N, temperature=0.7, max_new_tokens=4),
            diffusion=BagelDiffusionParams(
                samples_per_prompt=M, height=64, width=64, num_inference_steps=4, eta=1.0
            ),
        ),
        sigmas=torch.linspace(1.0, 0.0, 5),
    )
    resp = pipe._generate_t2ti(req)
    ar, img = resp.tracks["ar"], resp.tracks["image"]

    # AR ran on exactly P*N captions.
    assert pipe.ar.calls == [P * N], pipe.ar.calls
    exp_ar_sids = [f"{sid}/a{j}" for sid in sample_ids for j in range(N)]
    exp_ar_pids = [sid for sid in sample_ids for _ in range(N)]
    assert ar.sample_ids == exp_ar_sids, ar.sample_ids
    assert ar.parent_ids == exp_ar_pids, ar.parent_ids
    assert ar.parent_track is None

    # Image fan-out (parent_track="ar", M per caption, group-by-parent order).
    exp_img_sids = [f"{asid}/i{k}" for asid in exp_ar_sids for k in range(M)]
    exp_img_pids = [asid for asid in exp_ar_sids for _ in range(M)]
    assert img.sample_ids == exp_img_sids, img.sample_ids
    assert img.parent_ids == exp_img_pids, img.parent_ids
    assert img.parent_track == "ar"
    assert captured["diff_sample_ids"] == exp_img_sids
    assert captured["diff_parent_ids"] == exp_img_pids
    assert captured["diff_noise_ids"] == [f"r0:{s}" for s in exp_img_sids]

    # Index map: contexts[s] ← prompt (s//M)//N, caption T{s//M}.
    assert len(captured["contexts"]) == P * N * M
    for s in range(P * N * M):
        a = s // M
        _, ctx_prompt, ctx_think = captured["contexts"][s]
        assert ctx_prompt == prompts[a // N], (s, ctx_prompt)
        assert ctx_think == f"T{a}", (s, ctx_think)

    # Reward propagation (image → ar mean) + GRPO grouping accept the lineage.
    img_r = _track_with_field(img, "rewards", torch.arange(P * N * M, dtype=torch.float32))
    resp2 = RolloutResp(tracks={"ar": ar, "image": img_r})
    prop = resp2.propagate_rewards(op="mean")
    ar_rewards = prop.tracks["ar"].rewards
    assert ar_rewards is not None and list(ar_rewards.shape) == [P * N], ar_rewards
    # group-by-parent contiguity: these must not raise.
    prop.tracks["ar"].compute_advantages(normalize=True)
    img_r.compute_advantages(normalize=True)
    log(f"PASS  P={P} N={N} M={M}  ar={len(ar.sample_ids)} img={len(img.sample_ids)}")


def main() -> int:
    import unirl.models.bagel  # noqa: F401  flash-free import must survive

    log("PASS  flash-free import")
    run(2, 2, 2)
    run(1, 1, 1)  # flat 1:1 reproduces
    run(3, 4, 2)  # asymmetric P,N,M
    log("ALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
