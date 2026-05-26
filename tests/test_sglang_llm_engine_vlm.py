"""Unit tests for VLM (vision-language) support in the SGLang LLM engine.

Targets:
- ``_pil_to_base64`` round-trip encoding.
- ``_apply_chat_template`` multimodal content-parts path and fallback.
- ``generate()`` image extraction validation (no live server needed).
- Config ``image_token`` field behavior.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import PIL.Image
import pytest
import torch

from diffusionrl.rollout.engine.sglang_llm.config import SGLangLLMEngineConfig
from diffusionrl.rollout.engine.sglang_llm.engine import _pil_to_base64
from diffusionrl.types.primitives import Images, Texts
from diffusionrl.types.rollout_req import RolloutReq

# ---------------------------------------------------------------------------
# Config: image_token field
# ---------------------------------------------------------------------------


def test_config_image_token_default_none() -> None:
    cfg = SGLangLLMEngineConfig(pretrained_model_ckpt_path="/tmp/m")
    assert cfg.image_token is None


def test_config_image_token_set() -> None:
    cfg = SGLangLLMEngineConfig(
        pretrained_model_ckpt_path="/tmp/m",
        image_token="<|vision_start|><|image_pad|><|vision_end|>",
    )
    assert cfg.image_token == "<|vision_start|><|image_pad|><|vision_end|>"


# ---------------------------------------------------------------------------
# _pil_to_base64
# ---------------------------------------------------------------------------


def _make_pil_image(width: int = 32, height: int = 32) -> PIL.Image.Image:
    return PIL.Image.fromarray(torch.randint(0, 255, (height, width, 3), dtype=torch.uint8).numpy())


def test_pil_to_base64_returns_data_uri() -> None:
    img = _make_pil_image()
    uri = _pil_to_base64(img)
    assert uri.startswith("data:image/png;base64,")


def test_pil_to_base64_round_trip() -> None:
    img = _make_pil_image(16, 16)
    uri = _pil_to_base64(img)

    b64_data = uri.split(",", 1)[1]
    decoded = base64.b64decode(b64_data)
    recovered = PIL.Image.open(io.BytesIO(decoded))
    assert recovered.size == img.size
    assert recovered.mode == img.mode


def test_pil_to_base64_different_sizes() -> None:
    for w, h in [(1, 1), (64, 64), (224, 224)]:
        img = _make_pil_image(w, h)
        uri = _pil_to_base64(img)
        assert isinstance(uri, str)
        assert len(uri) > 30


# ---------------------------------------------------------------------------
# _apply_chat_template — multimodal paths
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer mock that records the messages it receives."""

    def __init__(self, *, supports_structured: bool = True):
        self._supports_structured = supports_structured
        self.last_messages: Optional[List[Dict[str, Any]]] = None

    def apply_chat_template(
        self,
        messages: List[Dict[str, Any]],
        add_generation_prompt: bool = True,
        tokenize: bool = True,
    ) -> List[int]:
        self.last_messages = messages
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list) and not self._supports_structured:
                raise TypeError("This tokenizer does not support structured content")
        return [1, 2, 3, 4, 5]

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        return "<preview>"


def _make_engine_shell(
    *,
    image_token: Optional[str] = None,
    supports_structured: bool = True,
) -> Any:
    """Build a minimal SGLangLLMRolloutEngine-shaped object for template tests.

    Does NOT launch a real server — only populates the fields that
    ``_apply_chat_template`` reads.
    """
    from diffusionrl.rollout.engine.sglang_llm.engine import SGLangLLMRolloutEngine

    obj = object.__new__(SGLangLLMRolloutEngine)
    obj.cfg = SGLangLLMEngineConfig(
        pretrained_model_ckpt_path="/tmp/m",
        image_token=image_token,
    )
    obj._tokenizer = _FakeTokenizer(supports_structured=supports_structured)
    obj._chat_template_logged = False
    return obj


def test_chat_template_text_only() -> None:
    engine = _make_engine_shell()
    ids = engine._apply_chat_template("hello world")
    assert ids == [1, 2, 3, 4, 5]
    msg = engine._tokenizer.last_messages[-1]
    assert msg["content"] == "hello world"


def test_chat_template_multimodal_structured() -> None:
    engine = _make_engine_shell(
        image_token="<|vision_start|><|image_pad|><|vision_end|>",
    )
    ids = engine._apply_chat_template("describe this", has_image=True)
    assert ids is not None
    msg = engine._tokenizer.last_messages[-1]
    assert isinstance(msg["content"], list)
    types = [part["type"] for part in msg["content"]]
    assert types == ["image", "text"]
    assert msg["content"][1]["text"] == "describe this"


def test_chat_template_multimodal_fallback() -> None:
    engine = _make_engine_shell(
        image_token="<image>",
        supports_structured=False,
    )
    ids = engine._apply_chat_template("describe this", has_image=True)
    assert ids is not None
    msg = engine._tokenizer.last_messages[-1]
    assert isinstance(msg["content"], str)
    assert msg["content"].startswith("<image>")
    assert "describe this" in msg["content"]


def test_chat_template_no_image_token_ignores_flag() -> None:
    engine = _make_engine_shell(image_token=None)
    ids = engine._apply_chat_template("hello", has_image=False)
    assert ids is not None
    msg = engine._tokenizer.last_messages[-1]
    assert msg["content"] == "hello"


def test_chat_template_with_system_instruction() -> None:
    engine = _make_engine_shell(
        image_token="<|vision_start|><|image_pad|><|vision_end|>",
    )
    ids = engine._apply_chat_template(
        "describe this",
        system_instruction="You are helpful.",
        has_image=True,
    )
    assert ids is not None
    msgs = engine._tokenizer.last_messages
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "You are helpful."
    assert msgs[1]["role"] == "user"
    assert isinstance(msgs[1]["content"], list)


# ---------------------------------------------------------------------------
# generate() — image extraction validation
# ---------------------------------------------------------------------------


def _make_req_with_images(
    prompts: List[str],
    *,
    include_images: bool = True,
    image_batch_mismatch: bool = False,
) -> RolloutReq:
    n = len(prompts)
    primitives: Dict[str, Any] = {"text": Texts(texts=list(prompts))}
    if include_images:
        img_count = n + 1 if image_batch_mismatch else n
        pixels = torch.rand(img_count, 3, 32, 32)
        primitives["image"] = Images(pixels=pixels)
    return RolloutReq(
        sample_ids=[f"s{i}" for i in range(n)],
        group_ids=["g0"] * n,
        primitives=primitives,
        stage_config={},
    )


def test_generate_rejects_images_without_image_token() -> None:
    """Images in req but image_token=None should raise."""
    engine = _make_engine_shell(image_token=None)
    engine._http_client = MagicMock()
    req = _make_req_with_images(["hello"], include_images=True)

    with pytest.raises(ValueError, match="image_token is None"):
        engine.generate(req)


def test_generate_rejects_image_batch_mismatch() -> None:
    """Image batch size != prompt count should raise."""
    engine = _make_engine_shell(image_token="<image>")
    engine._http_client = MagicMock()
    req = _make_req_with_images(["hello"], include_images=True, image_batch_mismatch=True)

    with pytest.raises(ValueError, match="image batch"):
        engine.generate(req)


def test_generate_accepts_text_only_with_image_token_set() -> None:
    """image_token configured but no images in req should NOT raise.

    VLMs can operate text-only even when multimodal-capable.
    The engine should proceed to HTTP (which we mock to avoid a live server).
    """
    engine = _make_engine_shell(image_token="<image>")
    engine._http_client = MagicMock()

    call_log: List[Any] = []

    async def _fake_generate_text_async(prompts, sampling_params, images=None):
        call_log.append({"prompts": prompts, "images": images})
        return [
            {
                "text": "response",
                "content": "response",
                "token_ids": [1, 2],
                "logprobs": [-0.1, -0.2],
                "prompt_token_ids": [10, 11],
            }
        ]

    engine._generate_text_async = _fake_generate_text_async
    engine._run_async_gather = lambda prompts, sampling, images=None: (
        __import__("asyncio")
        .new_event_loop()
        .run_until_complete(_fake_generate_text_async(prompts, sampling, images=images))
    )

    req = _make_req_with_images(["hello"], include_images=False)
    resp = engine.generate(req)
    assert resp is not None
    assert call_log[0]["images"] is None
