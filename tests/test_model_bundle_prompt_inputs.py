from __future__ import annotations

from types import MethodType

import pytest
import torch

from diffusionrl.models.mochi import MochiModelBundle
from diffusionrl.models.sd3 import SD3ModelBundle


class _FakeMochiTextEncoder:
    def encode_prompt(
        self,
        prompt: list[str],
        negative_prompt: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del negative_prompt
        batch_size = len(prompt)
        return (
            torch.zeros(batch_size, 3, 4),
            torch.ones(batch_size, 3, dtype=torch.long),
        )


def test_mochi_encode_inputs_keeps_attention_mask_semantics() -> None:
    bundle = object.__new__(MochiModelBundle)
    bundle._text_encoder = _FakeMochiTextEncoder()

    encoded = bundle.encode_inputs(["prompt A", "prompt B"])

    assert encoded["prompt_embeds"].shape == (2, 3, 4)
    assert torch.equal(encoded["encoder_attention_mask"], torch.ones(2, 3, dtype=torch.long))
    assert "pooled_prompt_embeds" not in encoded


@pytest.mark.parametrize(
    ("negative_prompt", "expected_negative_call", "expected_negative_value"),
    [
        (None, ["", ""], 0.0),
        ("avoid", ["avoid", "avoid"], -1.0),
    ],
)
def test_sd3_encode_inputs_builds_negative_prompt_branch(
    negative_prompt: str | None,
    expected_negative_call: list[str],
    expected_negative_value: float,
) -> None:
    bundle = object.__new__(SD3ModelBundle)
    calls: list[list[str]] = []

    def encode_prompt(self: SD3ModelBundle, prompts: list[str], **_: object) -> tuple[torch.Tensor, torch.Tensor]:
        del self
        calls.append(list(prompts))
        values = [expected_negative_value if prompt == expected_negative_call[0] else 1.0 for prompt in prompts]
        prompt_embeds = torch.tensor(values, dtype=torch.float32).reshape(len(prompts), 1, 1)
        pooled_prompt_embeds = torch.tensor(values, dtype=torch.float32).reshape(len(prompts), 1)
        return prompt_embeds, pooled_prompt_embeds

    bundle.encode_prompt = MethodType(encode_prompt, bundle)

    encoded = bundle.encode_inputs(["prompt A", "prompt B"], negative_prompt=negative_prompt)

    assert calls == [["prompt A", "prompt B"], expected_negative_call]
    assert torch.equal(encoded["prompt_embeds"], torch.ones(2, 1, 1))
    assert torch.equal(encoded["pooled_prompt_embeds"], torch.ones(2, 1))
    assert torch.equal(encoded["negative_prompt_embeds"], torch.full((2, 1, 1), expected_negative_value))
    assert torch.equal(encoded["negative_pooled_prompt_embeds"], torch.full((2, 1), expected_negative_value))
