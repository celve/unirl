"""Unit tests for the response-builder in :mod:`sglang_llm.engine`.

Targets the pure :func:`build_rollout_resp` helper — no live sglang runtime
needed. Pins:

- ``RolloutResp.tracks["text"].decoded`` Texts ordering and content.
- ``RolloutResp.tracks["text"].segment`` TextSegment packed-varlen shape:
  ``cu_seqlens``, ``sample_indices``, per-sample token row recovery via
  ``tokens[cu[i]:cu[i+1]]``.
- ``sample_ids`` / ``group_ids`` echo and the ``n > 1`` mangling rule
  (``f"{sid}#{k}"``).
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
import torch

from diffusionrl.rollout.engine.sglang_llm.engine import (
    _parse_one_response,
    _strip_thinking_tags,
    build_rollout_resp,
)
from diffusionrl.types.primitives import Texts
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.rollout_resp import RolloutResp


def _canned_candidate(
    text: str,
    token_ids: List[int],
    logprobs: List[float],
    finish_reason: str = "stop",
) -> Dict[str, Any]:
    return {
        "text": text,
        "meta_info": {
            "output_token_logprobs": [[float(lp), int(tid)] for lp, tid in zip(logprobs, token_ids)],
            "finish_reason": finish_reason,
        },
    }


def _make_req(prompts: List[str], *, with_ids: bool = True) -> RolloutReq:
    n = len(prompts)
    if with_ids:
        return RolloutReq(
            sample_ids=[f"s{i}" for i in range(n)],
            group_ids=["g0"] * n,
            primitives={"text": Texts(texts=list(prompts))},
        )
    return RolloutReq(
        primitives={"text": Texts(texts=list(prompts))},
    )


# ---------------------------------------------------------------------------
# build_rollout_resp
# ---------------------------------------------------------------------------


def test_build_resp_single_prompt_n1() -> None:
    prompts = ["hello"]
    raw = [
        {
            "text": "world",
            "content": "world",
            "token_ids": [100, 101, 102],
            "logprobs": [-0.1, -0.2, -0.3],
        }
    ]
    req = _make_req(prompts)

    resp = build_rollout_resp(req, prompts, raw, n_per_prompt=1)
    assert isinstance(resp, RolloutResp)
    track = resp.tracks["text"]
    assert track.decoded.texts == ["world"]
    assert track.sample_ids == ["s0"]
    assert track.group_ids == ["g0"]

    seg = track.segment
    assert seg.batch_size == 1
    assert seg.tokens.tolist() == [100, 101, 102]
    assert seg.log_probs.tolist() == [pytest.approx(-0.1), pytest.approx(-0.2), pytest.approx(-0.3)]
    cu = seg.cu_seqlens
    assert cu.tolist() == [0, 3]
    assert seg.sample_indices.tolist() == [0]


def test_build_resp_n_greater_than_one_remaps_sample_ids() -> None:
    """For n > 1, sample_ids mangle as 'sid#k' but group_ids stay shared."""
    prompts = ["A", "B"]
    n = 2
    raw = [
        # prompt 0 candidate 0
        {"text": "a0", "content": "a0", "token_ids": [1, 2], "logprobs": [-0.1, -0.2]},
        # prompt 0 candidate 1
        {"text": "a1", "content": "a1", "token_ids": [3], "logprobs": [-0.3]},
        # prompt 1 candidate 0
        {"text": "b0", "content": "b0", "token_ids": [4, 5, 6, 7], "logprobs": [-0.4, -0.5, -0.6, -0.7]},
        # prompt 1 candidate 1
        {"text": "b1", "content": "b1", "token_ids": [8, 9], "logprobs": [-0.8, -0.9]},
    ]
    req = _make_req(prompts)
    resp = build_rollout_resp(req, prompts, raw, n_per_prompt=n)

    track = resp.tracks["text"]
    assert track.decoded.texts == ["a0", "a1", "b0", "b1"]
    assert track.sample_ids == ["s0#0", "s0#1", "s1#0", "s1#1"]
    assert track.group_ids == ["g0"] * 4

    seg = track.segment
    assert seg.batch_size == 4
    assert seg.cu_seqlens.tolist() == [0, 2, 3, 7, 9]
    assert seg.tokens.tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert seg.sample_indices.tolist() == [0, 1, 2, 3]


def test_build_resp_per_sample_row_recovery() -> None:
    """Per-sample tokens reconstructable via tokens[cu[i]:cu[i+1]]."""
    prompts = ["P0", "P1", "P2"]
    raw = [
        {"text": "x", "content": "x", "token_ids": [10], "logprobs": [-0.1]},
        {"text": "yy", "content": "yy", "token_ids": [20, 21], "logprobs": [-0.2, -0.3]},
        {"text": "zzz", "content": "zzz", "token_ids": [30, 31, 32], "logprobs": [-0.4, -0.5, -0.6]},
    ]
    req = _make_req(prompts)
    resp = build_rollout_resp(req, prompts, raw, n_per_prompt=1)
    seg = resp.tracks["text"].segment
    cu = seg.cu_seqlens.tolist()
    expected = [[10], [20, 21], [30, 31, 32]]
    for i in range(3):
        assert seg.tokens[cu[i] : cu[i + 1]].tolist() == expected[i]


def test_build_resp_no_request_ids_synthesizes_them() -> None:
    prompts = ["p0", "p1"]
    raw = [
        {"text": "a", "content": "a", "token_ids": [1], "logprobs": [-0.1]},
        {"text": "b", "content": "b", "token_ids": [2], "logprobs": [-0.2]},
    ]
    req = _make_req(prompts, with_ids=False)
    resp = build_rollout_resp(req, prompts, raw, n_per_prompt=1)
    track = resp.tracks["text"]
    assert track.sample_ids == ["s0", "s1"]
    assert track.group_ids == ["s0", "s1"]


def test_build_resp_empty_text_uses_text_fallback() -> None:
    """When ``content`` is empty, fall back to ``text``; never None."""
    prompts = ["p"]
    raw = [
        {"text": "non-empty", "content": "", "token_ids": [1], "logprobs": [-0.1]},
    ]
    req = _make_req(prompts)
    resp = build_rollout_resp(req, prompts, raw, n_per_prompt=1)
    assert resp.tracks["text"].decoded.texts == ["non-empty"]


def test_build_resp_wrong_candidate_count_rejected() -> None:
    prompts = ["a", "b"]
    raw = [{"text": "x", "content": "x", "token_ids": [1], "logprobs": [-0.1]}]
    req = _make_req(prompts)
    with pytest.raises(ValueError, match="expected 2 candidates"):
        build_rollout_resp(req, prompts, raw, n_per_prompt=1)


def test_build_resp_packs_prompt_token_ids_into_conditions() -> None:
    """When raw candidates carry ``prompt_token_ids``, the track's
    ``conditions['prompt']`` is a right-padded TextTokenCondition that
    :meth:`Qwen3ARStage.replay` consumes at train time.
    """
    from diffusionrl.types.conditions import TextTokenCondition

    prompts = ["A", "B"]
    raw = [
        {
            "text": "a",
            "content": "a",
            "token_ids": [1],
            "logprobs": [-0.1],
            "prompt_token_ids": [100, 101, 102],  # 3 tokens
        },
        {
            "text": "b",
            "content": "b",
            "token_ids": [2],
            "logprobs": [-0.2],
            "prompt_token_ids": [200, 201, 202, 203, 204],  # 5 tokens
        },
    ]
    req = _make_req(prompts)
    resp = build_rollout_resp(req, prompts, raw, n_per_prompt=1, pad_token_id=999)

    track = resp.tracks["text"]
    assert "prompt" in track.conditions
    prompt_cond = track.conditions["prompt"]
    assert isinstance(prompt_cond, TextTokenCondition)
    # Right-padded to in-batch max (5) with pad_token_id=999.
    assert prompt_cond.input_ids.tolist() == [
        [100, 101, 102, 999, 999],
        [200, 201, 202, 203, 204],
    ]
    assert prompt_cond.attention_mask.tolist() == [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
    ]


def test_build_resp_no_prompt_token_ids_yields_empty_conditions() -> None:
    """When raw candidates omit ``prompt_token_ids`` (legacy/test fixtures),
    the track's ``conditions`` stays empty — preserves prior behavior so
    ComposedRolloutEngine + single-track AR pipelines that don't need the
    prompt-token-id round-trip aren't disturbed.
    """
    prompts = ["A"]
    raw = [
        {"text": "a", "content": "a", "token_ids": [1], "logprobs": [-0.1]},
    ]
    req = _make_req(prompts)
    resp = build_rollout_resp(req, prompts, raw, n_per_prompt=1)
    assert resp.tracks["text"].conditions == {}


# ---------------------------------------------------------------------------
# _parse_one_response (sglang JSON -> per-candidate dict)
# ---------------------------------------------------------------------------


def test_parse_one_response_dict_input() -> None:
    resp = _canned_candidate("hi", [10, 11], [-0.1, -0.2])
    out = _parse_one_response(resp, prompt="p", known_prompt_token_ids=[1, 2, 3])
    assert len(out) == 1
    r = out[0]
    assert r["text"] == "hi"
    assert r["content"] == "hi"
    assert r["reasoning_content"] == ""
    assert r["token_ids"] == [10, 11]
    assert r["logprobs"] == [pytest.approx(-0.1), pytest.approx(-0.2)]
    assert r["prompt_token_ids"] == [1, 2, 3]
    assert r["finish_reason"] == "stop"
    assert r["prompt"] == "p"


def test_parse_one_response_list_input() -> None:
    resp = [
        _canned_candidate("a", [1], [-0.1]),
        _canned_candidate("b", [2], [-0.2]),
    ]
    out = _parse_one_response(resp, prompt="p", known_prompt_token_ids=[7])
    assert [r["text"] for r in out] == ["a", "b"]
    assert [r["token_ids"] for r in out] == [[1], [2]]


def test_parse_one_response_finish_reason_dict() -> None:
    cand = _canned_candidate("x", [1], [-0.1])
    cand["meta_info"]["finish_reason"] = {"type": "length", "matched": 0}
    out = _parse_one_response(cand, prompt="p")
    assert out[0]["finish_reason"] == "length"


def test_parse_one_response_falls_back_to_output_token_ids() -> None:
    """If output_token_logprobs is empty but output_token_ids is set, use it."""
    cand = {
        "text": "x",
        "meta_info": {
            "output_token_logprobs": [],
            "output_token_ids": [42, 43, 44],
            "finish_reason": "stop",
        },
    }
    out = _parse_one_response(cand, prompt="p")
    assert out[0]["token_ids"] == [42, 43, 44]
    assert out[0]["logprobs"] == []


def test_parse_one_response_rejects_garbage_type() -> None:
    with pytest.raises(RuntimeError, match="Unexpected sglang response type"):
        _parse_one_response(42, prompt="p")


# ---------------------------------------------------------------------------
# _strip_thinking_tags
# ---------------------------------------------------------------------------


def test_strip_thinking_closed() -> None:
    content, reasoning = _strip_thinking_tags("<think>let me consider</think>answer")
    assert content == "answer"
    assert reasoning == "let me consider"


def test_strip_thinking_multiple_closed() -> None:
    content, reasoning = _strip_thinking_tags("<think>step1</think>middle<think>step2</think>final")
    assert content == "middlefinal"
    assert reasoning == "step1\nstep2"


def test_strip_thinking_unclosed() -> None:
    content, reasoning = _strip_thinking_tags("answer<think>cut off mid-thought")
    assert content == "answer"
    assert reasoning == "cut off mid-thought"


def test_strip_thinking_none() -> None:
    content, reasoning = _strip_thinking_tags("just an answer")
    assert content == "just an answer"
    assert reasoning == ""


# ---------------------------------------------------------------------------
# End-to-end: parse + build a typed resp
# ---------------------------------------------------------------------------


def test_end_to_end_parse_and_build() -> None:
    """Drive a canned /generate JSON through the full response pipeline."""
    prompts = ["P"]
    raw_response = _canned_candidate("hello world", [100, 101, 102], [-0.1, -0.2, -0.3])
    parsed = _parse_one_response(raw_response, prompt="P", known_prompt_token_ids=[1, 2])
    req = _make_req(prompts)
    resp = build_rollout_resp(req, prompts, parsed, n_per_prompt=1)

    track = resp.tracks["text"]
    assert track.decoded.texts == ["hello world"]
    seg = track.segment
    assert torch.equal(seg.tokens, torch.tensor([100, 101, 102], dtype=torch.long))
    assert seg.cu_seqlens.tolist() == [0, 3]
