"""Unit tests for :class:`PEPipeline` — composed LLM-expand + diffusion-generate
with sampling-param-driven N×M fan-out.

Tests use CPU-only stub Pipelines (the ``Pipeline`` Protocol requires
only ``.bundle`` and ``.generate(req) -> RolloutResp``) so no model
weights or GPU are needed. The stub Pipelines record their input
``RolloutReq`` for later inspection and return canned ``RolloutResp``
objects sized 1:1 to their request — mimicking the real Qwen3/SD3
pipelines, which neither expand nor drop samples. PE owns the fan-out
(prompt ×N for the LLM, each rewrite ×M for the diffusion child) and the
lineage; these tests verify that in isolation.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import pytest
import torch
from omegaconf import OmegaConf

from diffusionrl.models.pe import PEBundle, PEPipeline
from diffusionrl.types.conditions import ImageLatentCondition, TextEmbedCondition, TextTokenCondition
from diffusionrl.types.primitives import Texts
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutResp, RolloutTrack
from diffusionrl.types.sampling import (
    ARSamplingParams,
    ComposedSamplingParams,
    DiffusionSamplingParams,
)

# ---------------------------------------------------------------------------
# Stub Pipeline machinery
# ---------------------------------------------------------------------------


class _StubBundle:
    """Empty stand-in for any ``Bundle`` (the Protocol is empty)."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.pretrained_path = None  # diffusion child's schedule-policy fallback reads this


class _StubStage:
    """Minimal stage stand-in exposing a trainable ``nn.Module``."""

    def __init__(self) -> None:
        self._model = torch.nn.Linear(2, 2)

    def trainable_module(self):
        return self._model


class _RecordingLLMPipeline:
    """Stub LLM pipeline: 1:1 over its request, records input, returns canned text.

    ``rewritten_texts`` must match the LLM child's request sample count
    (i.e. P*N) — mirroring a real LLM pipeline that emits one rewrite per
    (already-replicated) input sample.
    """

    def __init__(self, rewritten_texts: list[str]) -> None:
        self.bundle = _StubBundle("llm")
        self.ar = _StubStage()  # trainable AR stage, for pe.ar accessor
        self._rewritten = rewritten_texts
        self.last_req: Optional[RolloutReq] = None

    def generate(self, req: RolloutReq) -> RolloutResp:
        self.last_req = req
        n = len(req.sample_ids)
        if len(self._rewritten) != n:
            raise AssertionError(f"stub LLM was configured for {len(self._rewritten)} rewrites but got {n} samples")
        prompt_cond = TextTokenCondition(
            input_ids=torch.zeros(n, 4, dtype=torch.long),
            attention_mask=torch.ones(n, 4, dtype=torch.long),
        )
        return RolloutResp(
            tracks={
                "ar": RolloutTrack(
                    sample_ids=list(req.sample_ids),
                    parent_ids=list(req.group_ids),
                    conditions={"prompt": prompt_cond},
                    segment=None,  # TextSegment not exercised here
                    decoded=Texts(texts=list(self._rewritten)),
                ),
            }
        )


class _RecordingDiffusionPipeline:
    """Stub diffusion pipeline: 1:1 over its request, records input, canned conditions."""

    def __init__(self) -> None:
        self.bundle = _StubBundle("diffusion")
        self.shift = 3.0  # diffusion child's schedule-policy fallback reads this
        self.diffusion = _StubStage()  # trainable diffusion stage, for pe.diffusion accessor
        self.last_req: Optional[RolloutReq] = None

    def generate(self, req: RolloutReq) -> RolloutResp:
        self.last_req = req
        n = len(req.sample_ids)
        text_cond = TextEmbedCondition(
            embeds=torch.zeros(n, 4, 8),
            pooled=torch.zeros(n, 8),
        )
        return RolloutResp(
            tracks={
                "image": RolloutTrack(
                    sample_ids=list(req.sample_ids),
                    parent_ids=list(req.group_ids),
                    conditions={"text": text_cond},
                    segment=None,  # LatentSegment not exercised here
                    decoded=None,  # Images output not exercised here
                ),
            }
        )


def _composed_sampling(*, n_rewrites: int = 1, n_images: int = 1) -> ComposedSamplingParams:
    return ComposedSamplingParams(
        diffusion=DiffusionSamplingParams(height=512, width=512, num_inference_steps=4, samples_per_prompt=n_images),
        ar=ARSamplingParams(max_new_tokens=16, temperature=0.7, samples_per_prompt=n_rewrites),
    )


def _make_parent_req(
    *,
    n: int = 2,
    with_sigmas: bool = True,
    with_request_conditions: bool = True,
    extra_primitives: Optional[dict] = None,
    sampling_params: Optional[ComposedSamplingParams] = None,
    stage_config: Optional[dict] = None,
) -> RolloutReq:
    primitives = {"text": Texts(texts=[f"raw prompt {i}" for i in range(n)])}
    if extra_primitives:
        primitives.update(extra_primitives)
    request_conditions = {}
    if with_request_conditions:
        request_conditions["initial_latents"] = ImageLatentCondition(
            latents=torch.zeros(n, 4, 8, 8),
        )
    sigmas = torch.linspace(1.0, 0.0, 5) if with_sigmas else None
    if sampling_params is None:
        sampling_params = _composed_sampling()
    if stage_config is None:
        stage_config = {"chat": {"system_instruction": "rewrite"}}
    return RolloutReq(
        sample_ids=[f"s{i}" for i in range(n)],
        group_ids=[f"g{i // 2}" for i in range(n)],
        primitives=primitives,
        request_conditions=request_conditions,
        sampling_params=sampling_params,
        stage_config=stage_config,
        sigmas=sigmas,
    )


# ---------------------------------------------------------------------------
# __init__ + bundle wiring
# ---------------------------------------------------------------------------


def test_init_exposes_composed_bundle() -> None:
    llm = _RecordingLLMPipeline(rewritten_texts=["a", "b"])
    diff = _RecordingDiffusionPipeline()

    pe = PEPipeline(diffusion_pipeline=diff, llm_pipeline=llm)

    assert isinstance(pe.bundle, PEBundle)
    assert pe.bundle.diffusion is diff.bundle
    assert pe.bundle.llm is llm.bundle
    assert pe.diffusion_pipeline is diff
    assert pe.llm_pipeline is llm


def test_stage_accessors_and_schedule_policy() -> None:
    """PEPipeline exposes the child stages + a σ-schedule for a trainside engine."""
    from diffusionrl.sde.runtime import FlowMatchSchedulePolicy

    llm = _RecordingLLMPipeline(rewritten_texts=["a", "b"])
    diff = _RecordingDiffusionPipeline()
    pe = PEPipeline(diffusion_pipeline=diff, llm_pipeline=llm)

    # Stage accessors delegate to the children — a trainside engine reaches the
    # trainable modules via pe.diffusion / pe.ar (stage_attrs=["diffusion","ar"]).
    assert pe.diffusion is diff.diffusion
    assert pe.ar is llm.ar
    assert pe.diffusion.trainable_module() is diff.diffusion.trainable_module()

    # Schedule policy reaches through to the diffusion child (the composed
    # PEBundle has no shift / pretrained_path of its own).
    policy = pe.build_schedule_policy()
    assert isinstance(policy, FlowMatchSchedulePolicy)


# ---------------------------------------------------------------------------
# generate(): N×M fan-out + lineage  (the load-bearing test)
# ---------------------------------------------------------------------------


def test_generate_branches_by_samples_per_prompt() -> None:
    """P=2 prompts, N=3 rewrites/prompt, M=4 images/rewrite → ar=6, diffusion=24."""
    P, N, M = 2, 3, 4
    llm = _RecordingLLMPipeline(rewritten_texts=[f"rw{i}" for i in range(P * N)])
    diff = _RecordingDiffusionPipeline()
    pe = PEPipeline(diffusion_pipeline=diff, llm_pipeline=llm)

    parent = _make_parent_req(n=P, sampling_params=_composed_sampling(n_rewrites=N, n_images=M))
    resp = pe.generate(parent)

    ar = resp.tracks["ar"]
    di = resp.tracks["diffusion"]

    # ── sample counts fan out by the branch factors ──
    assert set(resp.tracks) == {"ar", "diffusion"}
    assert len(ar.sample_ids) == P * N == 6
    assert len(di.sample_ids) == P * N * M == 24

    # ── lineage: ar is the root grouped by prompt; diffusion groups by rewrite ──
    assert ar.parent_track is None
    assert ar.parent_ids == [sid for sid in parent.sample_ids for _ in range(N)]
    assert di.parent_track == "ar"
    assert di.parent_ids == [sid for sid in ar.sample_ids for _ in range(M)]

    # ── GRPO-groupable: every group has >1 member (N on ar, M on diffusion) ──
    assert set(Counter(ar.group_ids).values()) == {N}
    assert set(Counter(di.group_ids).values()) == {M}

    # ── PE replicated the inputs for the 1:1 children (prompt-major order) ──
    assert llm.last_req is not None and diff.last_req is not None
    assert len(llm.last_req.sample_ids) == P * N
    assert llm.last_req.primitives["text"].texts == [t for t in parent.primitives["text"].texts for _ in range(N)]
    assert llm.last_req.group_ids == ar.parent_ids  # sub-req group_ids carry the lineage parents
    assert len(diff.last_req.sample_ids) == P * N * M
    assert diff.last_req.primitives["text"].texts == [t for t in ar.decoded.texts for _ in range(M)]
    assert diff.last_req.group_ids == di.parent_ids

    # merged child payloads land on the right tracks
    assert "prompt" in ar.conditions
    assert ar.decoded.texts == [f"rw{i}" for i in range(P * N)]
    assert "text" in di.conditions


def test_generate_default_sampling_is_one_to_one() -> None:
    """Default sampling (N=M=1) still wraps with lineage: ar=P, diffusion=P."""
    P = 2
    llm = _RecordingLLMPipeline(rewritten_texts=["r0", "r1"])
    diff = _RecordingDiffusionPipeline()
    pe = PEPipeline(diffusion_pipeline=diff, llm_pipeline=llm)

    resp = pe.generate(_make_parent_req(n=P))

    assert len(resp.tracks["ar"].sample_ids) == P
    assert len(resp.tracks["diffusion"].sample_ids) == P
    # Even at N=M=1 the tracks carry lineage (root "ar" + child "diffusion").
    assert resp.tracks["ar"].parent_track is None
    assert resp.tracks["diffusion"].parent_track == "ar"


# ---------------------------------------------------------------------------
# generate(): LLM child slice
# ---------------------------------------------------------------------------


def test_generate_llm_child_receives_replicated_text_only() -> None:
    N = 2
    llm = _RecordingLLMPipeline(rewritten_texts=["x0", "x1", "y0", "y1"])  # P*N = 4
    diff = _RecordingDiffusionPipeline()
    pe = PEPipeline(diffusion_pipeline=diff, llm_pipeline=llm)

    parent = _make_parent_req(
        n=2,
        sampling_params=_composed_sampling(n_rewrites=N, n_images=1),
        extra_primitives={"negative_text": Texts(texts=["neg0", "neg1"])},
    )
    pe.generate(parent)

    assert llm.last_req is not None
    # primitives: only text forwarded to the LLM, replicated ×N (prompt-major)
    assert set(llm.last_req.primitives.keys()) == {"text"}
    assert llm.last_req.primitives["text"].texts == ["raw prompt 0", "raw prompt 0", "raw prompt 1", "raw prompt 1"]
    # sampling_params: AR params only; stage_config: chat only
    assert isinstance(llm.last_req.sampling_params, ARSamplingParams)
    assert set(llm.last_req.stage_config.keys()) == {"chat"}
    # sigmas and request_conditions stripped from LLM child
    assert llm.last_req.sigmas is None
    assert llm.last_req.request_conditions == {}


# ---------------------------------------------------------------------------
# generate(): diffusion child slice
# ---------------------------------------------------------------------------


def test_generate_diffusion_child_receives_rewritten_text_and_sigmas() -> None:
    M = 2
    llm = _RecordingLLMPipeline(rewritten_texts=["rewritten 0", "rewritten 1"])  # N=1 → P*N=2
    diff = _RecordingDiffusionPipeline()
    pe = PEPipeline(diffusion_pipeline=diff, llm_pipeline=llm)

    parent = _make_parent_req(n=2, sampling_params=_composed_sampling(n_rewrites=1, n_images=M))
    pe.generate(parent)

    assert diff.last_req is not None
    # primitives["text"] = each rewrite replicated ×M (rewrite-major)
    assert diff.last_req.primitives["text"].texts == ["rewritten 0", "rewritten 0", "rewritten 1", "rewritten 1"]
    # sampling_params: diffusion params only
    assert isinstance(diff.last_req.sampling_params, DiffusionSamplingParams)
    # sigmas and request_conditions forwarded verbatim
    assert diff.last_req.sigmas is not None
    torch.testing.assert_close(diff.last_req.sigmas, parent.sigmas)
    assert "initial_latents" in diff.last_req.request_conditions


def test_generate_diffusion_child_drops_non_text_primitives() -> None:
    """Under branching, non-text per-sample primitives are NOT forwarded.

    (Matches ComposedRolloutEngine; SD3's empty-negative default applies.)
    """
    llm = _RecordingLLMPipeline(rewritten_texts=["rewritten 0", "rewritten 1"])
    diff = _RecordingDiffusionPipeline()
    pe = PEPipeline(diffusion_pipeline=diff, llm_pipeline=llm)

    parent = _make_parent_req(extra_primitives={"negative_text": Texts(texts=["neg0", "neg1"])})
    pe.generate(parent)

    assert diff.last_req is not None
    assert set(diff.last_req.primitives.keys()) == {"text"}
    assert "negative_text" not in diff.last_req.primitives


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_generate_rejects_non_texts_primitive() -> None:
    """primitives['text'] must be a Texts instance."""
    llm = _RecordingLLMPipeline(rewritten_texts=["r0"])
    diff = _RecordingDiffusionPipeline()
    pe = PEPipeline(diffusion_pipeline=diff, llm_pipeline=llm)

    parent = RolloutReq(
        sample_ids=["s0"],
        group_ids=["g0"],
        primitives={"text": "raw"},  # str, not Texts — invalid
        request_conditions={},
        sampling_params=_composed_sampling(),
        sigmas=None,
    )
    with pytest.raises(TypeError, match=r"primitives\['text'\] must be a Texts"):
        pe.generate(parent)


def test_generate_rejects_sample_count_mismatch_from_llm() -> None:
    """The LLM child must be 1:1 over its (P*N) request."""
    llm = _RecordingLLMPipeline(rewritten_texts=["r0"])  # only 1 rewrite
    diff = _RecordingDiffusionPipeline()
    pe = PEPipeline(diffusion_pipeline=diff, llm_pipeline=llm)

    parent = _make_parent_req(n=2)  # P*N = 2 expected
    # The stub raises AssertionError internally when configured-for != got;
    # PE raises RuntimeError if the count slips through. Either surfaces.
    with pytest.raises((AssertionError, RuntimeError)):
        pe.generate(parent)


def test_generate_rejects_missing_decoded_text() -> None:
    """If the LLM child's "ar" track has no decoded Texts, PE raises a clear error."""

    class _EmptyLLM:
        def __init__(self) -> None:
            self.bundle = _StubBundle("llm")

        def generate(self, req: RolloutReq) -> RolloutResp:
            return RolloutResp(
                tracks={
                    "ar": RolloutTrack(
                        sample_ids=list(req.sample_ids),
                        parent_ids=list(req.group_ids),
                        conditions={},
                        segment=None,
                        decoded=None,  # no Texts on the LLM track
                    ),
                }
            )

    pe = PEPipeline(diffusion_pipeline=_RecordingDiffusionPipeline(), llm_pipeline=_EmptyLLM())
    parent = _make_parent_req()
    with pytest.raises(RuntimeError, match=r"tracks\['ar'\]\.decoded"):
        pe.generate(parent)


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------


def test_from_config_dispatches_each_child_via_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """``PEPipeline.from_config`` calls ``build()`` on each child DictConfig."""
    llm = _RecordingLLMPipeline(rewritten_texts=["x"])
    diff = _RecordingDiffusionPipeline()

    seen: list = []

    def fake_build(cfg, **deps):
        seen.append(dict(cfg))
        # Return diffusion for the first call, llm for the second — matches
        # PEPipeline.from_config's call order.
        return diff if len(seen) == 1 else llm

    # Patch the build symbol that PE imports from
    # ``diffusionrl.config.instantiate``. The from_config method does
    # ``from diffusionrl.config.instantiate import build`` at module import,
    # so we patch the rebinding inside the pipeline module.
    import diffusionrl.models.pe.pipeline as pe_pipeline_mod

    monkeypatch.setattr(pe_pipeline_mod, "build", fake_build)

    cfg = OmegaConf.create(
        {
            "diffusion": {"_target_": "tests.unused.Diffusion.from_config", "ckpt": "/fake/diff"},
            "llm": {"_target_": "tests.unused.LLM.from_config", "ckpt": "/fake/llm"},
        }
    )

    pe = PEPipeline.from_config(cfg)

    assert pe.diffusion_pipeline is diff
    assert pe.llm_pipeline is llm
    assert len(seen) == 2
    # First call was diffusion cfg, second was llm cfg
    assert seen[0]["ckpt"] == "/fake/diff"
    assert seen[1]["ckpt"] == "/fake/llm"
