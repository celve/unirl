from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from diffusionrl.models.wan21 import WAN21ModelBundle, WANTextEncoderWrapper


class _FakeWANTextEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[str] | None]] = []

    def encode_prompt(
        self,
        prompt: list[str],
        negative_prompt: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append((list(prompt), None if negative_prompt is None else list(negative_prompt)))
        batch_size = len(prompt)
        return (
            torch.zeros(batch_size, 2, 3),
            torch.ones(batch_size, 2, dtype=torch.long),
        )


def _wan_bundle_with_text_encoder(text_encoder: _FakeWANTextEncoder) -> WAN21ModelBundle:
    bundle = object.__new__(WAN21ModelBundle)
    bundle._text_encoder = text_encoder
    return bundle


def test_wan_encode_inputs_broadcasts_scalar_negative_prompt() -> None:
    text_encoder = _FakeWANTextEncoder()
    bundle = _wan_bundle_with_text_encoder(text_encoder)

    encoded = bundle.encode_inputs(["prompt A", "prompt B"], negative_prompt="")

    assert encoded["prompt_embeds"].shape[0] == 2
    assert encoded["negative_prompt_embeds"].shape[0] == 2
    assert "encoder_attention_mask" not in encoded
    assert text_encoder.calls == [
        (["prompt A", "prompt B"], None),
        (["", ""], None),
    ]


def test_wan_encode_inputs_rejects_mismatched_negative_prompt_batch() -> None:
    text_encoder = _FakeWANTextEncoder()
    bundle = _wan_bundle_with_text_encoder(text_encoder)

    with pytest.raises(ValueError, match="negative_prompt batch size"):
        bundle.encode_inputs(["prompt A", "prompt B"], negative_prompt=["only one"])


class _FakeTokenizerOutput:
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask


class _FakeTokenizer:
    def __call__(self, prompt: list[str], **_: object) -> _FakeTokenizerOutput:
        batch_size = len(prompt)
        input_ids = torch.arange(batch_size * 4, dtype=torch.long).reshape(batch_size, 4)
        attention_mask = torch.tensor(
            [
                [1, 1, 0, 0],
                [1, 1, 1, 0],
            ][:batch_size],
            dtype=torch.long,
        )
        return _FakeTokenizerOutput(input_ids=input_ids, attention_mask=attention_mask)


class _FakeEncoderOutput:
    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


class _FakeEncoder:
    def __call__(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> _FakeEncoderOutput:
        del attention_mask
        hidden = torch.arange(input_ids.numel() * 3, dtype=torch.float32).reshape(*input_ids.shape, 3) + 1.0
        return _FakeEncoderOutput(last_hidden_state=hidden)


def test_wan_text_encoder_zeroes_padding_embeddings() -> None:
    text_encoder = WANTextEncoderWrapper(
        encoder=_FakeEncoder(),
        tokenizer=_FakeTokenizer(),
        device="cpu",
        dtype=torch.float32,
        max_length=4,
    )

    prompt_embeds, attention_mask = text_encoder.encode_prompt(["prompt A", "prompt B"])

    assert torch.equal(
        attention_mask,
        torch.tensor(
            [
                [1, 1, 0, 0],
                [1, 1, 1, 0],
            ],
            dtype=torch.long,
        ),
    )
    assert torch.all(prompt_embeds[0, 2:] == 0)
    assert torch.all(prompt_embeds[1, 3:] == 0)
    assert torch.all(prompt_embeds[0, :2] != 0)
    assert torch.all(prompt_embeds[1, :3] != 0)
