from __future__ import annotations

from typing import Any, Dict, List, Tuple

from diffusionrl.rollout.engine.sglang_llm.engine import SGLangLLMRolloutEngine


class _AliveProcess:
    def is_alive(self) -> bool:
        return True


def _engine_stub(*, offloaded: bool) -> SGLangLLMRolloutEngine:
    engine = object.__new__(SGLangLLMRolloutEngine)
    engine._server_process = _AliveProcess()
    engine._is_offloaded = offloaded
    engine._weights_onloaded_for_sync = False
    engine._base_url = "http://unused"
    engine._posts: List[Tuple[str, Dict[str, Any]]] = []
    engine._flushes = 0

    def post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        engine._posts.append((path, dict(payload)))
        return {}

    def flush_cache() -> None:
        engine._flushes += 1

    engine._post = post
    engine._flush_cache = flush_cache
    return engine


def test_onload_weights_resumes_only_weights() -> None:
    engine = _engine_stub(offloaded=True)

    engine.onload_weights()

    assert engine._posts == [("/resume_memory_occupation", {"tags": ["weights"]})]
    assert engine.is_offloaded
    assert engine._weights_onloaded_for_sync


def test_sleep_after_weight_onload_releases_only_weights_without_flush() -> None:
    engine = _engine_stub(offloaded=True)
    engine.onload_weights()
    engine._posts.clear()

    engine.sleep()

    assert engine._posts == [("/release_memory_occupation", {"tags": ["weights"]})]
    assert engine._flushes == 0
    assert engine.is_offloaded
    assert not engine._weights_onloaded_for_sync


def test_full_wake_after_weight_onload_resumes_remaining_memory_types() -> None:
    engine = _engine_stub(offloaded=True)
    engine.onload_weights()
    engine._posts.clear()

    engine.wake_up()

    assert engine._posts == [("/resume_memory_occupation", {"tags": ["kv_cache", "cuda_graph"]})]
    assert not engine.is_offloaded
    assert not engine._weights_onloaded_for_sync


def test_full_sleep_flushes_and_releases_all_memory_types() -> None:
    engine = _engine_stub(offloaded=False)

    engine.sleep()

    assert engine._posts == [("/release_memory_occupation", {})]
    assert engine._flushes == 1
    assert engine.is_offloaded
