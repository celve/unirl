"""Integration test: text reward pipeline with MCExactMatchRewardScorer."""

from __future__ import annotations

import os
import sys
import types

import torch

# Stub the scorers package to avoid heavy-dep imports (tqdm, paddleocr).
_SCORERS_PKG = "diffusionrl.reward.scorers"
if _SCORERS_PKG not in sys.modules:
    import diffusionrl.reward

    _stub = types.ModuleType(_SCORERS_PKG)
    _scorers_dir = os.path.join(os.path.dirname(diffusionrl.reward.__file__), "scorers")
    _stub.__path__ = [_scorers_dir]
    _stub.__package__ = _SCORERS_PKG
    sys.modules[_SCORERS_PKG] = _stub

from diffusionrl.reward.base import InProcessRewardExecutor  # noqa: E402
from diffusionrl.reward.pipeline import RewardPipeline  # noqa: E402
from diffusionrl.reward.scorers.mc_exact_match import MCExactMatchRewardScorer, MCExactMatchSpec  # noqa: E402
from diffusionrl.reward.service import RewardService  # noqa: E402
from diffusionrl.types.primitives import Texts  # noqa: E402
from diffusionrl.types.rollout_req import RolloutReq  # noqa: E402
from diffusionrl.types.rollout_resp import RolloutTrack  # noqa: E402
from diffusionrl.types.segments.text import TextSegment  # noqa: E402


def _make_text_track(texts: list[str]) -> RolloutTrack:
    n = len(texts)
    per_sample_tokens = [torch.tensor([i], dtype=torch.long) for i in range(n)]
    per_sample_logprobs = [torch.tensor([0.0]) for _ in range(n)]
    segment = TextSegment.pack(
        tokens=per_sample_tokens,
        log_probs=per_sample_logprobs,
        sample_indices=torch.arange(n, dtype=torch.long),
    )
    return RolloutTrack(
        sample_ids=[f"s{i}" for i in range(n)],
        parent_ids=[f"g{i}" for i in range(n)],
        conditions={},
        segment=segment,
        decoded=Texts(texts=texts),
    )


def _make_req(prompts: list[str], metadata: list[dict]) -> RolloutReq:
    return RolloutReq(
        sample_ids=[f"s{i}" for i in range(len(prompts))],
        group_ids=[f"g{i}" for i in range(len(prompts))],
        primitives={"text": Texts(texts=prompts)},
        metadata=metadata,
    )


class TestTextsProperty:
    def test_extracts_texts_from_generated(self):
        from diffusionrl.types.reward import RewardRequest

        req = RewardRequest(generated={"text": Texts(texts=["hello", "world"])})
        assert req.texts == ["hello", "world"]

    def test_returns_none_when_missing(self):
        from diffusionrl.types.reward import RewardRequest

        req = RewardRequest(generated={"image": Texts(texts=[])})
        assert req.texts is None


class TestTextRewardPipeline:
    def test_score_and_attach_mc(self):
        scorer = MCExactMatchRewardScorer(config=MCExactMatchSpec(weight=1.0), base_device="cpu")
        executor = InProcessRewardExecutor(scorer, weight=1.0)
        service = RewardService(executors=[executor], aggregation_method="mean")
        pipeline = RewardPipeline(service)
        assert pipeline.preferred_input_kind == "text"

        track = _make_text_track(["C", "A", "B"])
        req = _make_req(
            prompts=["q1", "q2", "q3"],
            metadata=[
                {"answer": "C"},
                {"answer": "B"},
                {"answer": "B"},
            ],
        )

        pipeline.score_and_attach(req=req, track=track)

        assert track.rewards is not None
        rewards = track.rewards.tolist()
        assert rewards == [1.0, 0.0, 1.0]
