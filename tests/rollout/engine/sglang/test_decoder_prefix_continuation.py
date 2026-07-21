"""CPU contract for SGLang's decoder-prefix continuation primitive."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, List

import torch

from unirl.rollout.engine.sglang.adapters.text import TextLMAdapter
from unirl.rollout.engine.sglang.engine import SGLangRolloutEngine
from unirl.rollout.engine.sglang.utils.conditions import pack_prompt_condition
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams
from unirl.types.segments.text import TextSegment


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 99

    @staticmethod
    def encode(text: str, *, add_special_tokens: bool) -> List[int]:
        assert text == "<answer>"
        assert add_special_tokens is False
        return [7, 8]


@dataclass
class _Raw:
    text: str
    token_ids: List[int]
    logprobs: List[float]
    finish_reason: str = "stop"


class _Backend:
    def __init__(self) -> None:
        self.requests: List[dict[str, Any]] = []

    def generate(self, requests: List[dict[str, Any]]) -> List[_Raw]:
        self.requests = requests
        return [
            _Raw(text="alpha</answer>", token_ids=[40, 41, 42], logprobs=[-0.4, -0.5, -0.6]),
            _Raw(text="beta</answer>", token_ids=[50, 51], logprobs=[-0.7, -0.8]),
        ]


def _engine() -> SGLangRolloutEngine:
    tokenizer = _Tokenizer()
    adapter = object.__new__(TextLMAdapter)
    adapter._tokenizer = tokenizer

    engine = object.__new__(SGLangRolloutEngine)
    engine.cfg = SimpleNamespace(
        temperature=0.7,
        max_new_tokens=512,
        top_p=0.9,
        system_instruction=None,
    )
    engine.adapter = adapter
    engine._tokenizer = tokenizer
    engine._backend = _Backend()
    engine._weight_sync = SimpleNamespace(active_adapter="repair-lora")
    engine._weight_version = 3
    return engine


def _prior_generation() -> Sample:
    params = ARSamplingParams(
        samples_per_prompt=1,
        temperature=1.0,
        max_new_tokens=128,
        top_p=1.0,
        top_k=0,
    )
    root = Part.input(
        ["p0", "p1"],
        primitive=Texts(texts=["q0", "q1"]),
        control={
            "ar": {
                "stop": ["<tool_response>"],
                "no_stop_trim": False,
                "return_logprob": True,
            }
        },
    )
    request = Sample.request(root).fork(1, sampling_params=params)
    response = TextSegment.pack(
        tokens=[torch.tensor([20, 99]), torch.tensor([30, 31, 99, 99])],
        log_probs=[torch.tensor([-0.1, -0.2]), torch.tensor([-0.3, -0.4, -0.5, -0.6])],
    )
    prompt = pack_prompt_condition([[1, 2], [3, 4, 5]], pad_token_id=0)
    prior = request.parts[-1].fill(
        segment=response,
        primitive=Texts(texts=["reasoning-a", "reasoning-b"]),
        conditions={"prompt": prompt},
    )
    return request.with_parts([root, prior])


def _real_prompt_ids(part: Part, row: int) -> List[int]:
    prompt = part.conditions["prompt"]
    return prompt.input_ids[row][prompt.attention_mask[row].bool()].tolist()


def test_decoder_prefix_continuation_uses_exact_tokens_and_samples_suffix_only():
    engine = _engine()
    prior = _prior_generation()
    repair_params = ARSamplingParams(
        samples_per_prompt=1,
        temperature=0.2,
        max_new_tokens=64,
        top_p=0.8,
        top_k=0,
    )

    continued = engine.continue_generation(
        prior,
        prefix="<answer>",
        sampling_params=repair_params,
    )

    # Prompt = exact previous prompt + previous sampled response (terminal EOS
    # removed) + injected prefix. It never passes through the chat template.
    assert [request["input_ids"] for request in engine._backend.requests] == [
        [1, 2, 20, 7, 8],
        [3, 4, 5, 30, 31, 7, 8],
    ]
    for request in engine._backend.requests:
        assert request["sampling_params"]["stop"] == ["</answer>"]
        assert request["sampling_params"]["no_stop_trim"] is True
        assert request["sampling_params"]["n"] == 1
        assert request["sampling_params"]["max_new_tokens"] == 64
        assert request["lora_path"] == "repair-lora"

    assert len(continued.parts) == len(prior.parts) + 1
    frontier = continued.parts[-1]
    assert _real_prompt_ids(frontier, 0) == [1, 2, 20, 7, 8]
    assert _real_prompt_ids(frontier, 1) == [3, 4, 5, 30, 31, 7, 8]

    # Injected prefix belongs only to conditioning. The policy segment and old
    # log-probs are exactly the newly sampled suffix returned by SGLang.
    assert frontier.segment.tokens.tolist() == [40, 41, 42, 50, 51]
    assert torch.allclose(
        frontier.segment.log_probs,
        torch.tensor([-0.4, -0.5, -0.6, -0.7, -0.8]),
    )
    assert frontier.segment.lengths.tolist() == [3, 2]

    # Scoring reads the final generated primitive, so expose a complete tagged
    # response even though the opener itself was decoder conditioning.
    assert frontier.primitive.texts == [
        "<answer>alpha</answer>",
        "<answer>beta</answer>",
    ]
    assert frontier.weight_version == 3
    assert frontier.metadata == [
        {"answer_injected": True, "format_repair": True, "decoder_prefix": "<answer>"},
        {"answer_injected": True, "format_repair": True, "decoder_prefix": "<answer>"},
    ]

    # The source generation is preserved byte-for-byte as its own policy Part.
    assert continued.parts[-2] is prior.parts[-1]
    assert continued.parts[-2].segment.tokens.tolist() == [20, 99, 30, 31, 99, 99]
