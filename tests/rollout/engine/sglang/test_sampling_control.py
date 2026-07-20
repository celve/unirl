"""CPU contracts for request-scoped SGLang sampling controls."""

from __future__ import annotations

from types import SimpleNamespace

from unirl.rollout.engine.sglang.adapters.text import TextLMAdapter
from unirl.rollout.engine.sglang.backends.native import payload_to_generate_kwargs
from unirl.rollout.engine.sglang.utils.sampling import resolve_sampling
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams

_CFG = SimpleNamespace(
    temperature=0.7,
    max_new_tokens=512,
    top_p=0.9,
    system_instruction=None,
)


def _request(ar_control: dict) -> Sample:
    root = Part.input(
        ["prompt-0"],
        primitive=Texts(texts=["question"]),
        control={"ar": ar_control},
    )
    return Sample.request(root).fork(
        1,
        sampling_params=ARSamplingParams(
            samples_per_prompt=1,
            temperature=1.0,
            max_new_tokens=8192,
            top_p=1.0,
            top_k=0,
        ),
    )


def test_closed_tool_boundary_reaches_sglang_sampling_params_and_native_kwargs():
    resolved = resolve_sampling(
        _CFG,
        _request({"stop": ["</tool_call>"], "no_stop_trim": True}),
    )

    assert resolved.block["stop"] == ["</tool_call>"]
    assert resolved.block["no_stop_trim"] is True

    adapter = object.__new__(TextLMAdapter)
    payload = adapter.base_payload(resolved)
    kwargs = payload_to_generate_kwargs({"input_ids": [1, 2], **payload})
    assert kwargs["sampling_params"]["stop"] == ["</tool_call>"]
    assert kwargs["sampling_params"]["no_stop_trim"] is True


def test_no_stop_trim_is_absent_without_explicit_request_control():
    resolved = resolve_sampling(_CFG, _request({"stop": ["</tool_call>"]}))

    assert "no_stop_trim" not in resolved.block
