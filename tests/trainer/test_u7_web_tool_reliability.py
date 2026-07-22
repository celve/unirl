"""Hermetic CPU tests for U7 web-tool transport reliability."""

from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

import pytest
import requests

pytest.importorskip("torch")

from unirl.rollout.loop import ToolEnvironment  # noqa: E402
from unirl.rollout.loop.tools.polaris import (  # noqa: E402
    POLARIS_JINA_URL,
    POLARIS_SERPER_URL,
    application_error,
    full_jitter_delay,
    get_gateway_semaphore,
    response_error,
)
from unirl.rollout.loop.tools.search import SearchTool  # noqa: E402
from unirl.rollout.loop.tools.tool import Tool, ToolExecutionResult  # noqa: E402
from unirl.rollout.loop.tools.visit import VisitTool  # noqa: E402
from unirl.types.primitives import Texts  # noqa: E402
from unirl.types.sample import Part, Sample  # noqa: E402
from unirl.types.sampling import ARSamplingParams  # noqa: E402


class _Resp:
    def __init__(self, payload=None, text="", status_code=200, headers=None, json_error=False):
        self._payload = payload if payload is not None else {}
        self.text = text
        self.status_code = status_code
        self.headers = headers or {}
        self._json_error = json_error

    def json(self):
        if self._json_error:
            raise ValueError("malformed")
        return self._payload


@pytest.fixture(autouse=True)
def _clean_web_env(monkeypatch):
    for name in (
        "POLARIS_APP_ID",
        "POLARIS_APP_KEY",
        "POLARIS_PROVIDER_TIMEOUT",
        "SERPER_APP_ID",
        "SERPER_APP_KEY",
        "JINA_APP_ID",
        "JINA_APP_KEY",
        "SERPER_AUTH",
        "SERPER_KEY_ID",
        "SERPER_URL",
        "JINA_PROVIDER",
        "JINA_API_KEYS",
        "SEARCH_PROVIDER",
        "SUMMARY_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def _polaris_env(monkeypatch):
    monkeypatch.setenv("POLARIS_APP_ID", "app")
    monkeypatch.setenv("POLARIS_APP_KEY", "key")
    monkeypatch.setenv("SERPER_AUTH", "polaris")
    monkeypatch.setenv("JINA_PROVIDER", "jina_ai")


@pytest.mark.parametrize("status", [408, 425, 429, 500, 503])
def test_transient_http_status_classification(status):
    assert response_error(_Resp(status_code=status)).category == "transient"


@pytest.mark.parametrize(("status", "category"), [(401, "auth"), (403, "auth"), (422, "permanent")])
def test_non_retryable_http_status_classification(status, category):
    assert response_error(_Resp(status_code=status)).category == category


def test_application_status_family_classification():
    assert application_error(42901).category == "transient"
    assert application_error(42206).category == "permanent"
    assert application_error(40301).category == "auth"


def test_retry_jitter_does_not_advance_global_python_rng():
    random.seed(1234)
    expected = random.random()
    random.seed(1234)
    full_jitter_delay(2, base_s=0.5, cap_s=8.0)
    assert random.random() == expected


def test_gateway_capacity_mismatch_fails_fast():
    gateway = "test://u7-capacity-mismatch"
    get_gateway_semaphore(2, gateway=gateway)
    with pytest.raises(ValueError, match="same gateway_max_in_flight"):
        get_gateway_semaphore(3, gateway=gateway)


def test_search_recovers_application_429_with_safe_diagnostics(monkeypatch):
    import unirl.rollout.loop.tools.search as mod

    _polaris_env(monkeypatch)
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)
    calls = 0

    def fake_post(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _Resp({"code": 429})
        return _Resp({"organic": [{"title": "T", "link": "https://x", "snippet": "S"}]})

    monkeypatch.setattr(mod.requests, "post", fake_post)
    secret_query = "TOP_SECRET_QUERY"
    result = SearchTool(retry_base_s=0).execute_with_info({"query": secret_query})
    assert result.text == f"Results for {secret_query!r}:\n1. [T](https://x)\nS"
    assert calls == 2
    assert result.diagnostics["request_count"] == 2
    assert result.diagnostics["retry_count"] == 1
    assert result.diagnostics["success_count"] == 1
    assert result.diagnostics["recovered_transient_count"] == 1
    assert result.diagnostics["status_counts"]["app_429"] == 1
    assert secret_query not in str(result.diagnostics)


@pytest.mark.parametrize("status", ["success", 0, "20000"])
def test_search_accepts_benign_optional_status(monkeypatch, status):
    import unirl.rollout.loop.tools.search as mod

    _polaris_env(monkeypatch)
    monkeypatch.setattr(
        mod.requests,
        "post",
        lambda *a, **k: _Resp(
            {
                "code": 200,
                "status": status,
                "data": {"organic": [{"title": "T", "link": "https://x", "snippet": "S"}]},
            }
        ),
    )
    result = SearchTool().execute_with_info({"query": "q"})
    assert result.diagnostics["success_count"] == 1
    assert result.diagnostics["transient_exhausted_count"] == 0


def test_bad_polaris_timeout_is_auth_configuration_failure(monkeypatch):
    _polaris_env(monkeypatch)
    monkeypatch.setenv("POLARIS_PROVIDER_TIMEOUT", "bad")
    result = SearchTool().execute_with_info({"query": "q"})
    assert result.diagnostics["request_count"] == 0
    assert result.diagnostics["auth_error_count"] == 1
    assert result.diagnostics["permanent_error_count"] == 0


def test_search_retry_after_is_capped_and_exhaustion_is_actionable(monkeypatch):
    import unirl.rollout.loop.tools.search as mod

    sleeps = []
    monkeypatch.setattr(mod.time, "sleep", sleeps.append)
    calls = 0

    def throttled(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _Resp(status_code=429, headers={"Retry-After": "99"})

    monkeypatch.setattr(mod.requests, "post", throttled)
    result = SearchTool(max_retries=4, retry_base_s=0).execute_with_info({"query": "q"})
    assert calls == 4
    assert sleeps == [30.0, 30.0, 30.0]
    assert "temporarily unavailable" in result.text
    assert "Do not repeat the identical request" in result.text
    assert result.diagnostics["retry_count"] == 3
    assert result.diagnostics["transient_exhausted_count"] == 1
    assert result.diagnostics["status_counts"] == {"http_429": 4}


def test_search_success_cache_is_single_flight(monkeypatch):
    import unirl.rollout.loop.tools.search as mod

    calls = 0
    lock = threading.Lock()

    def slow_success(*args, **kwargs):
        nonlocal calls
        with lock:
            calls += 1
        time.sleep(0.05)
        return _Resp({"organic": [{"title": "T", "link": "https://x", "snippet": "S"}]})

    monkeypatch.setattr(mod.requests, "post", slow_success)
    tool = SearchTool()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: tool.execute_with_info({"query": "same"}), range(8)))
    assert calls == 1
    assert len({result.text for result in results}) == 1
    assert sum(result.diagnostics["request_count"] for result in results) == 1
    assert sum(result.diagnostics["cache_hit_count"] for result in results) == 7


def test_search_cache_normalizes_only_whitespace(monkeypatch):
    import unirl.rollout.loop.tools.search as mod

    calls = 0

    def success(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _Resp({"organic": []})

    monkeypatch.setattr(mod.requests, "post", success)
    tool = SearchTool()
    tool.execute_with_info({"query": "  apple   inc "})
    cached = tool.execute_with_info({"query": "apple inc"})
    assert calls == 1
    assert cached.diagnostics["cache_hit_count"] == 1


def test_search_does_not_cache_permanent_failures(monkeypatch):
    import unirl.rollout.loop.tools.search as mod

    calls = 0

    def unprocessable(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _Resp(status_code=422)

    monkeypatch.setattr(mod.requests, "post", unprocessable)
    tool = SearchTool()
    first = tool.execute_with_info({"query": "same"})
    second = tool.execute_with_info({"query": "same"})
    assert calls == 2
    assert first.diagnostics["permanent_error_count"] == 1
    assert second.diagnostics["cache_hit_count"] == 0


def test_jina_malformed_envelope_retries_then_caches_only_14k(monkeypatch):
    import unirl.rollout.loop.tools.visit as mod

    _polaris_env(monkeypatch)
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)
    calls = 0

    def flaky(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _Resp(json_error=True)
        return _Resp({"code": 200, "data": {"content": "X" * 15000}})

    monkeypatch.setattr(mod.requests, "post", flaky)
    tool = VisitTool(max_content_chars=20000, retry_base_s=0)
    first = tool.execute_with_info({"url": "https://example.test/a", "goal": "g1"})
    second = tool.execute_with_info({"url": "https://example.test/a", "goal": "g2"})
    assert calls == 2
    assert first.text.endswith("X" * 14000)
    assert first.diagnostics["retry_count"] == 1
    assert first.diagnostics["recovered_transient_count"] == 1
    assert first.diagnostics["status_counts"]["malformed_jina_envelope"] == 1
    assert second.diagnostics["request_count"] == 0
    assert second.diagnostics["cache_hit_count"] == 1


@pytest.mark.parametrize("status", ["success", 0, "20000"])
def test_jina_accepts_benign_optional_status(monkeypatch, status):
    import unirl.rollout.loop.tools.visit as mod

    _polaris_env(monkeypatch)
    monkeypatch.setattr(
        mod.requests,
        "post",
        lambda *a, **k: _Resp(
            {"code": 200, "status": status, "data": {"content": "PAGE"}}
        ),
    )
    result = VisitTool().execute_with_info({"url": "https://example.test", "goal": "g"})
    assert result.diagnostics["success_count"] == 1
    assert result.diagnostics["transient_exhausted_count"] == 0


def test_visit_cache_ignores_url_fragment(monkeypatch):
    import unirl.rollout.loop.tools.visit as mod

    _polaris_env(monkeypatch)
    calls = 0

    def success(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _Resp({"code": 200, "data": {"content": "PAGE"}})

    monkeypatch.setattr(mod.requests, "post", success)
    tool = VisitTool()
    tool.execute_with_info({"url": "https://example.test/a#one", "goal": "g1"})
    cached = tool.execute_with_info({"url": "https://example.test/a#two", "goal": "g2"})
    assert calls == 1
    assert cached.diagnostics["cache_hit_count"] == 1


def test_jina_empty_envelope_exhausts_but_422_is_permanent(monkeypatch):
    import unirl.rollout.loop.tools.visit as mod

    _polaris_env(monkeypatch)
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)
    calls = 0

    def empty(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _Resp({"code": 200, "data": {"content": ""}})

    monkeypatch.setattr(mod.requests, "post", empty)
    transient = VisitTool(retry_base_s=0).execute_with_info(
        {"url": "https://example.test/empty", "goal": "g"}
    )
    assert calls == 4
    assert "Do not repeat the identical request" in transient.text
    assert transient.diagnostics["transient_exhausted_count"] == 1

    calls = 0
    monkeypatch.setattr(mod.requests, "post", lambda *a, **k: _Resp({"code": 422}))
    permanent = VisitTool(retry_base_s=0).execute_with_info(
        {"url": "https://example.test/permanent", "goal": "g"}
    )
    assert "could not be accessed" in permanent.text
    assert permanent.diagnostics["request_count"] == 1
    assert permanent.diagnostics["permanent_error_count"] == 1


def test_search_and_visit_share_combined_polaris_capacity_two(monkeypatch):
    _polaris_env(monkeypatch)
    active = 0
    peak = 0
    lock = threading.Lock()

    def provider(url, *args, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        if url == POLARIS_SERPER_URL:
            return _Resp({"organic": []})
        assert url == POLARIS_JINA_URL
        return _Resp({"code": 200, "data": {"content": "PAGE"}})

    monkeypatch.setattr(requests, "post", provider)
    search = SearchTool()
    visit = VisitTool()
    jobs = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i in range(4):
            jobs.append(pool.submit(search.execute, {"query": f"q{i}"}))
            jobs.append(pool.submit(visit.execute, {"url": f"https://x/{i}", "goal": "g"}))
        [job.result() for job in jobs]
    assert peak == 2


class _DiagnosticTool(Tool):
    name = "diagnostic"

    def json_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "parameters": {"type": "object"}},
        }

    def execute(self, arguments: Dict[str, Any]) -> str:
        return "ok"

    def execute_with_info(self, arguments: Dict[str, Any]) -> ToolExecutionResult:
        return ToolExecutionResult(
            text="ok",
            diagnostics={
                "tool": "spoofed",
                "provider": "polaris",
                "request_count": 1,
                "success_count": 1,
                "cache_hit_count": 0,
                "retry_count": 0,
                "recovered_transient_count": 0,
                "transient_exhausted_count": 0,
                "permanent_error_count": 0,
                "auth_error_count": 0,
                "status_counts": {
                    "http_200": 1,
                    "https://secret/path": 9,
                    "TOP_SECRET_QUERY": 7,
                },
                "url": "https://secret/path",
                "query": "secret query",
                "body": "secret body",
            },
        )


def _turn(body: str) -> Sample:
    sample = Sample.request(Part.input(["r0"], primitive=Texts(texts=["prompt"])))
    sample = sample.fork(1, sampling_params=ARSamplingParams(samples_per_prompt=1))
    return sample.with_filled_frontier(primitive=Texts(texts=[body]))


def test_tool_environment_emits_row_aligned_allowlisted_diagnostics():
    call = '<tool_call>{"name":"diagnostic","arguments":{}}</tool_call>'
    _, _, info = ToolEnvironment([_DiagnosticTool()]).step(_turn(call))
    assert len(info["tool_diagnostics"]) == len(info["results"]) == 1
    diagnostics = info["tool_diagnostics"][0]
    assert diagnostics["tool"] == "diagnostic"
    assert diagnostics["provider"] == "polaris"
    assert diagnostics["status_counts"] == {"http_200": 1}
    assert set(diagnostics) == {
        "tool",
        "provider",
        "request_count",
        "success_count",
        "cache_hit_count",
        "retry_count",
        "recovered_transient_count",
        "transient_exhausted_count",
        "permanent_error_count",
        "auth_error_count",
        "status_counts",
    }
    assert info["tool_diagnostics"] == [diagnostics]


def test_tool_environment_uses_none_for_rows_without_tool_calls():
    _, _, info = ToolEnvironment([_DiagnosticTool()]).step(_turn("<answer>done</answer>"))
    assert info["tool_diagnostics"] == [None]
