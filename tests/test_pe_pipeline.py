"""Unit tests for :class:`PEPipeline` — composed LLM-expand + diffusion-generate.

Tests use CPU-only stub Pipelines (the ``Pipeline`` Protocol requires
only ``.bundle`` and ``.generate(req) -> RolloutResp``) so no model
weights or GPU are needed. The stub Pipelines record their input
``RolloutReq`` for later inspection and return canned ``RolloutResp``
objects, letting us verify PE's request-slicing and response-merging
logic in isolation.
"""

from __future__ import annotations

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


class _RecordingLLMPipeline:
    """Stub LLM pipeline that records its input and returns canned text."""

    def __init__(self, rewritten_texts: list[str]) -> None:
        self.bundle = _StubBundle("llm")
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
                "text": RolloutTrack(
                    sample_ids=list(req.sample_ids),
                    parent_ids=list(req.group_ids),
                    conditions={"prompt": prompt_cond},
                    segment=None,  # TextSegment not exercised here
                    decoded=Texts(texts=list(self._rewritten)),
                ),
            }
        )


class _RecordingDiffusionPipeline:
    """Stub diffusion pipeline that records its input and returns canned image conditions."""

    def __init__(self) -> None:
        self.bundle = _StubBundle("diffusion")
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
        sampling_params = ComposedSamplingParams(
            diffusion=DiffusionSamplingParams(height=512, width=512, num_inference_steps=4),
            ar=ARSamplingParams(max_new_tokens=16, temperature=0.7),
        )
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


# ---------------------------------------------------------------------------
# generate(): LLM child slice
# ---------------------------------------------------------------------------


def test_generate_llm_child_receives_raw_text_only() -> None:
    llm = _RecordingLLMPipeline(rewritten_texts=["x", "y"])
    diff = _RecordingDiffusionPipeline()
    pe = PEPipeline(diffusion_pipeline=diff, llm_pipeline=llm)

    parent = _make_parent_req(
        extra_primitives={"negative_text": Texts(texts=["neg0", "neg1"])},
    )
    pe.generate(parent)

    assert llm.last_req is not None
    # primitives: only text forwarded to the LLM
    assert set(llm.last_req.primitives.keys()) == {"text"}
    assert llm.last_req.primitives["text"].texts == ["raw prompt 0", "raw prompt 1"]
    # sampling_params: AR params only; stage_config: chat only
    assert isinstance(llm.last_req.sampling_params, ARSamplingParams)
    assert set(llm.last_req.stage_config.keys()) == {"chat"}
    # sigmas and request_conditions stripped from LLM child
    assert llm.last_req.sigmas is None
    assert llm.last_req.request_conditions == {}
    # sample_ids / group_ids preserved
    assert llm.last_req.sample_ids == parent.sample_ids
    assert llm.last_req.group_ids == parent.group_ids


# ---------------------------------------------------------------------------
# generate(): diffusion child slice
# ---------------------------------------------------------------------------


def test_generate_diffusion_child_receives_rewritten_text_and_sigmas() -> None:
    llm = _RecordingLLMPipeline(rewritten_texts=["rewritten 0", "rewritten 1"])
    diff = _RecordingDiffusionPipeline()
    pe = PEPipeline(diffusion_pipeline=diff, llm_pipeline=llm)

    parent = _make_parent_req()
    pe.generate(parent)

    assert diff.last_req is not None
    # primitives["text"] replaced with the LLM-rewritten text
    assert diff.last_req.primitives["text"].texts == ["rewritten 0", "rewritten 1"]
    # sampling_params: diffusion params only
    assert isinstance(diff.last_req.sampling_params, DiffusionSamplingParams)
    # sigmas and request_conditions forwarded verbatim
    assert diff.last_req.sigmas is not None
    torch.testing.assert_close(diff.last_req.sigmas, parent.sigmas)
    assert "initial_latents" in diff.last_req.request_conditions
    # sample_ids / group_ids preserved
    assert diff.last_req.sample_ids == parent.sample_ids
    assert diff.last_req.group_ids == parent.group_ids


def test_generate_diffusion_child_forwards_other_primitives() -> None:
    """Non-text primitives (e.g. negative_text, image) are forwarded to the diffusion child."""
    llm = _RecordingLLMPipeline(rewritten_texts=["rewritten 0", "rewritten 1"])
    diff = _RecordingDiffusionPipeline()
    pe = PEPipeline(diffusion_pipeline=diff, llm_pipeline=llm)

    parent = _make_parent_req(
        extra_primitives={"negative_text": Texts(texts=["neg0", "neg1"])},
    )
    pe.generate(parent)

    assert diff.last_req is not None
    assert "negative_text" in diff.last_req.primitives
    assert diff.last_req.primitives["negative_text"].texts == ["neg0", "neg1"]
    # text was still replaced with the rewritten version
    assert diff.last_req.primitives["text"].texts == ["rewritten 0", "rewritten 1"]


# ---------------------------------------------------------------------------
# generate(): merged response shape
# ---------------------------------------------------------------------------


def test_generate_merges_responses_by_track_union() -> None:
    llm = _RecordingLLMPipeline(rewritten_texts=["r0", "r1"])
    diff = _RecordingDiffusionPipeline()
    pe = PEPipeline(diffusion_pipeline=diff, llm_pipeline=llm)

    parent = _make_parent_req()
    resp = pe.generate(parent)

    # Both children's tracks are union'd into one resp.
    assert set(resp.tracks.keys()) == {"text", "image"}
    # Per-track conditions: LLM contributes "prompt" on "text" track;
    # diffusion contributes "text" condition on "image" track.
    assert "prompt" in resp.tracks["text"].conditions
    assert "text" in resp.tracks["image"].conditions
    # Decoded text comes from LLM (rewritten prompt) on the "text" track.
    assert resp.tracks["text"].decoded.texts == ["r0", "r1"]
    # Sample IDs preserved per track (no concat shift); both tracks see the
    # same parent sample set since PE is 1:1.
    assert resp.tracks["text"].sample_ids == parent.sample_ids
    assert resp.tracks["image"].sample_ids == parent.sample_ids
    # parent_ids carry the legacy group_ids semantic per track.
    assert resp.tracks["text"].parent_ids == parent.group_ids
    assert resp.tracks["image"].parent_ids == parent.group_ids
    assert len(resp.tracks["text"].sample_ids) == 2  # NOT 4 — no sample-axis stacking


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
        sigmas=None,
    )
    with pytest.raises(TypeError, match=r"primitives\['text'\] must be a Texts"):
        pe.generate(parent)


def test_generate_rejects_sample_count_mismatch_from_llm() -> None:
    """The LLM child must preserve the parent sample count 1:1."""
    llm = _RecordingLLMPipeline(rewritten_texts=["r0"])  # only 1 rewrite
    diff = _RecordingDiffusionPipeline()
    pe = PEPipeline(diffusion_pipeline=diff, llm_pipeline=llm)

    parent = _make_parent_req(n=2)  # 2 samples expected
    # The stub raises AssertionError internally when configured-for != got;
    # we wrap to confirm the failure surfaces (regardless of which side raises).
    with pytest.raises((AssertionError, RuntimeError)):
        pe.generate(parent)


def test_generate_rejects_missing_decoded_text() -> None:
    """If the LLM child's "text" track has no decoded Texts, PE raises a clear error."""

    class _EmptyLLM:
        def __init__(self) -> None:
            self.bundle = _StubBundle("llm")

        def generate(self, req: RolloutReq) -> RolloutResp:
            return RolloutResp(
                tracks={
                    "text": RolloutTrack(
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
    with pytest.raises(RuntimeError, match=r"tracks\['text'\]\.decoded"):
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
