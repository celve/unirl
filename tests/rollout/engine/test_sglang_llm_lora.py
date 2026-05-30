"""Unit tests for SGLangLLMRolloutEngine native LoRA-pool sync.

Covers the opt-in ``set_lora_from_tensors`` path (HTTP
``/load_lora_adapter_from_tensors``), per-request ``lora_path`` activation in
generation, and adapter invalidation on weight release. Mirrors the stub style
of ``test_sglang_llm_memory_lifecycle.py`` — no live SRT server.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Dict, List, Tuple

import pytest

from diffusionrl.rollout.engine.sglang_llm.engine import SGLangLLMRolloutEngine


class _AliveProcess:
    def is_alive(self) -> bool:
        return True


def _engine_stub(*, offloaded: bool = False, lora_loaded: bool = False) -> SGLangLLMRolloutEngine:
    engine = object.__new__(SGLangLLMRolloutEngine)
    engine._server_process = _AliveProcess()
    engine._is_offloaded = offloaded
    engine._weights_onloaded_for_sync = False
    engine._base_url = "http://unused"
    engine._lora_loaded = lora_loaded
    engine._lora_adapter_name = "default" if lora_loaded else None
    engine._posts: List[Tuple[str, Dict[str, Any]]] = []
    engine._post_errors: Dict[str, Exception] = {}
    engine._flushes = 0

    def post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        engine._posts.append((path, dict(payload)))
        err = engine._post_errors.get(path)
        if err is not None:
            raise err
        return {"success": True}

    def flush_cache() -> None:
        engine._flushes += 1

    engine._post = post
    engine._flush_cache = flush_cache
    return engine


@pytest.fixture
def fake_serializer(monkeypatch):
    """Stub ``sglang.srt.utils.MultiprocessingSerializer`` so the test runs
    without a real (CUDA-backed) sglang install and without serializing real
    tensors."""
    captured: Dict[str, Any] = {}

    class _Serializer:
        @staticmethod
        def serialize(obj: Any, output_str: bool = False) -> str:
            captured["obj"] = obj
            captured["output_str"] = output_str
            return "SERIALIZED_SENTINEL"

    mod = types.ModuleType("sglang.srt.utils")
    mod.MultiprocessingSerializer = _Serializer
    for parent in ("sglang", "sglang.srt"):
        if parent not in sys.modules:
            monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", mod)
    return captured


# ---------------------------------------------------------------------------
# set_lora_from_tensors
# ---------------------------------------------------------------------------


def test_engine_overrides_base_set_lora_from_tensors() -> None:
    from diffusionrl.rollout.engine.base import BaseRolloutEngine

    assert hasattr(SGLangLLMRolloutEngine, "set_lora_from_tensors")
    assert SGLangLLMRolloutEngine.set_lora_from_tensors is not BaseRolloutEngine.set_lora_from_tensors


def test_set_lora_from_tensors_unloads_then_loads(fake_serializer) -> None:
    engine = _engine_stub()
    peft = {"r": 8, "lora_alpha": 16, "target_modules": ["q_proj", "v_proj"]}
    tensors = {"model.layers.0.self_attn.q_proj.lora_A.weight": object()}

    engine.set_lora_from_tensors("default", tensors, peft_config=peft)

    assert [p for p, _ in engine._posts] == [
        "/unload_lora_adapter",
        "/load_lora_adapter_from_tensors",
    ]
    unload_payload = engine._posts[0][1]
    load_payload = engine._posts[1][1]
    assert unload_payload == {"lora_name": "default"}
    assert load_payload["lora_name"] == "default"
    assert load_payload["config_dict"] == peft
    assert load_payload["serialized_tensors"] == "SERIALIZED_SENTINEL"
    # Serialized as an output string (HTTP transport), and the exact tensor
    # bag is passed through unchanged (no key adaptation for SRT).
    assert fake_serializer["output_str"] is True
    assert fake_serializer["obj"] is tensors
    assert engine._lora_loaded is True
    assert engine._lora_adapter_name == "default"


def test_set_lora_from_tensors_none_peft_config_becomes_empty(fake_serializer) -> None:
    engine = _engine_stub()
    engine.set_lora_from_tensors("default", {"k": object()})
    assert engine._posts[1][1]["config_dict"] == {}


def test_set_lora_from_tensors_swallows_unload_failure(fake_serializer) -> None:
    engine = _engine_stub()
    # First push: nothing to unload — SRT returns 400 -> _post raises.
    engine._post_errors["/unload_lora_adapter"] = RuntimeError("not loaded")

    engine.set_lora_from_tensors(
        "default",
        {"k": object()},
        peft_config={"r": 4, "lora_alpha": 8, "target_modules": ["q_proj"]},
    )

    assert [p for p, _ in engine._posts] == [
        "/unload_lora_adapter",
        "/load_lora_adapter_from_tensors",
    ]
    assert engine._lora_loaded is True


def test_set_lora_from_tensors_propagates_load_failure(fake_serializer) -> None:
    engine = _engine_stub()
    engine._post_errors["/load_lora_adapter_from_tensors"] = RuntimeError("rank too big")
    with pytest.raises(RuntimeError, match="rank too big"):
        engine.set_lora_from_tensors("default", {"k": object()})
    assert engine._lora_loaded is False


def test_check_update_response_surfaces_error_message() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        SGLangLLMRolloutEngine._check_update_response(
            {"success": False, "error_message": "boom"}, "set_lora_from_tensors"
        )


# ---------------------------------------------------------------------------
# Adapter invalidation on weight release
# ---------------------------------------------------------------------------


def test_full_sleep_clears_lora_loaded() -> None:
    engine = _engine_stub(offloaded=False, lora_loaded=True)
    engine.sleep()
    assert engine._lora_loaded is False


def test_weights_tag_sleep_clears_lora_loaded() -> None:
    engine = _engine_stub(offloaded=False, lora_loaded=True)
    engine.sleep(tags=["weights"])
    assert engine._lora_loaded is False


def test_kv_cache_only_sleep_keeps_lora_loaded() -> None:
    engine = _engine_stub(offloaded=False, lora_loaded=True)
    engine.sleep(tags=["kv_cache"])
    assert engine._lora_loaded is True


def test_onload_weights_keeps_lora_loaded() -> None:
    engine = _engine_stub(offloaded=True, lora_loaded=True)
    engine.onload_weights()
    assert engine._lora_loaded is True


# ---------------------------------------------------------------------------
# Per-request activation in generation
# ---------------------------------------------------------------------------


def _generate_stub(*, lora_loaded: bool) -> Tuple[SGLangLLMRolloutEngine, Dict[str, Any]]:
    engine = _engine_stub(lora_loaded=lora_loaded)
    engine._http_client = object()  # only checked for non-None
    engine._concurrency = 2
    engine._tokenizer = None
    engine._parse_response_logged_first = True

    captured: Dict[str, Any] = {}

    def apply_chat_template(prompt, system_instruction=None, has_image=False):
        return [1, 2, 3]

    async def fake_apost(path, payload, max_retries=60):
        captured["path"] = path
        captured["payload"] = dict(payload)
        return {"text": "hi", "meta_info": {"output_token_logprobs": [[-0.1, 5]]}}

    engine._apply_chat_template = apply_chat_template
    engine._apost = fake_apost
    return engine, captured


def _sampling() -> Dict[str, Any]:
    return {
        "n": 1,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 8,
        "return_logprob": True,
    }


def test_generate_injects_lora_path_when_loaded() -> None:
    engine, captured = _generate_stub(lora_loaded=True)
    engine._run_async_gather(["a prompt"], _sampling())
    assert captured["payload"].get("lora_path") == "default"


def test_generate_omits_lora_path_when_not_loaded() -> None:
    engine, captured = _generate_stub(lora_loaded=False)
    engine._run_async_gather(["a prompt"], _sampling())
    assert "lora_path" not in captured["payload"]
