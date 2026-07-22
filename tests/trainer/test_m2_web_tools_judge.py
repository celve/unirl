"""CPU tests for the M2 deep-research web tools + LLM-judge (LIN-519).

No network: HTTP is monkeypatched. Validates each tool's ``json_schema`` and
argument handling, the Serper/Jina response parsing, and the judge's
correct/incorrect verdict parsing.
"""

from __future__ import annotations

import pytest
import requests

pytest.importorskip("torch")  # the unirl types import torch at module load
pytest.importorskip("requests")

from unirl.reward.local.llm_judge import LLMJudgeRewardScorer, LLMJudgeSpec  # noqa: E402
from unirl.rollout.loop.tools import SearchTool, VisitTool  # noqa: E402
from unirl.types.primitives import Texts  # noqa: E402
from unirl.types.reward import RewardRequest  # noqa: E402


class _Resp:
    def __init__(self, payload=None, text="", status_code=200):
        self._payload = payload or {}
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):  # noqa: D401
        if self.status_code >= 400:
            response = requests.Response()
            response.status_code = self.status_code
            raise requests.HTTPError(f"HTTP {self.status_code}", response=response)

    def json(self):
        return self._payload


def _chat(content):
    return _Resp({"choices": [{"message": {"content": content}}]})


@pytest.fixture(autouse=True)
def _isolate_web_tool_environment(monkeypatch):
    """Keep developer/pod credentials from changing these hermetic tests."""
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
        "JINA_READER_URL",
        "JINA_API_KEYS",
        "SEARCH_PROVIDER",
        "SUMMARY_URL",
    ):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# SearchTool
# --------------------------------------------------------------------------- #


def test_search_schema_and_validation():
    tool = SearchTool()
    assert tool.name == "search"
    fn = tool.json_schema()["function"]
    assert fn["name"] == "search"
    assert fn["parameters"]["required"] == ["query"]
    with pytest.raises(ValueError):
        tool.execute({})
    with pytest.raises(ValueError):
        tool.execute({"query": []})


def test_search_parses_serper(monkeypatch):
    import unirl.rollout.loop.tools.search as mod

    def fake_post(url, json=None, headers=None, timeout=None):
        assert json["q"] == "who won"
        return _Resp({"organic": [{"title": "T", "link": "http://x", "snippet": "S"}]})

    monkeypatch.setattr(mod.requests, "post", fake_post)
    out = SearchTool().execute({"query": ["who won"]})
    assert "T" in out and "http://x" in out and "S" in out


@pytest.mark.parametrize("provider", ["serper", "jina_ai"])
def test_polaris_headers_use_common_app_pair(monkeypatch, provider):
    from unirl.rollout.loop.tools.polaris import polaris_headers

    monkeypatch.setenv("POLARIS_APP_ID", "test-app")
    monkeypatch.setenv("POLARIS_APP_KEY", "test-key")
    monkeypatch.setenv("POLARIS_PROVIDER_TIMEOUT", "60")
    assert polaris_headers(provider) == {
        "Authorization": f"Bearer test-app:test-key?provider={provider}&timeout=60",
        "Content-Type": "application/json",
    }


def test_polaris_common_pair_wins_over_stale_provider_pair(monkeypatch):
    from unirl.rollout.loop.tools.polaris import polaris_headers

    monkeypatch.setenv("POLARIS_APP_ID", "current-app")
    monkeypatch.setenv("POLARIS_APP_KEY", "current-key")
    monkeypatch.setenv("SERPER_APP_ID", "stale-app")
    monkeypatch.setenv("SERPER_APP_KEY", "stale-key")
    header = polaris_headers("serper")["Authorization"]
    assert "current-app:current-key" in header
    assert "stale" not in header


def test_polaris_rejects_half_pair_without_leaking_secret(monkeypatch):
    from unirl.rollout.loop.tools.polaris import polaris_headers

    secret = "must-not-appear"
    monkeypatch.setenv("POLARIS_APP_ID", secret)
    with pytest.raises(RuntimeError) as caught:
        polaris_headers("serper")
    assert secret not in str(caught.value)


def test_polaris_rejects_bad_provider_timeout(monkeypatch):
    from unirl.rollout.loop.tools.polaris import polaris_headers

    monkeypatch.setenv("POLARIS_APP_ID", "app")
    monkeypatch.setenv("POLARIS_APP_KEY", "key")
    monkeypatch.setenv("POLARIS_PROVIDER_TIMEOUT", "not-a-number")
    with pytest.raises(RuntimeError, match="positive integer"):
        polaris_headers("jina_ai")


def test_search_uses_polaris_gateway_contract(monkeypatch):
    import unirl.rollout.loop.tools.search as mod
    from unirl.rollout.loop.tools.polaris import POLARIS_SERPER_URL

    monkeypatch.setenv("POLARIS_APP_ID", "search-app")
    monkeypatch.setenv("POLARIS_APP_KEY", "search-key")
    monkeypatch.setenv("SERPER_AUTH", "polaris")
    monkeypatch.setenv("SERPER_URL", "https://stale.example/credential-sink")
    monkeypatch.setenv("SERPER_KEY_ID", "stale-legacy-key")

    def fake_post(url, json=None, headers=None, timeout=None):
        assert url == POLARIS_SERPER_URL
        assert json == {"q": "who won", "num": 10}
        assert headers["Authorization"] == (
            "Bearer search-app:search-key?provider=serper&timeout=60"
        )
        assert "stale-legacy-key" not in str(headers)
        assert timeout == 65.0
        return _Resp({"organic": [{"title": "T", "link": "http://x", "snippet": "S"}]})

    monkeypatch.setattr(mod.requests, "post", fake_post)
    assert "T" in SearchTool().execute({"query": "who won"})


def test_search_error_is_text(monkeypatch):
    import unirl.rollout.loop.tools.search as mod

    monkeypatch.setattr(mod.requests, "post", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("net down")))
    out = SearchTool().execute({"query": "x"})
    assert out.startswith("[search] error")  # surfaced to the model, not raised


def test_search_error_redacts_exception_and_fails_fast(monkeypatch):
    import unirl.rollout.loop.tools.search as mod

    secret = "Bearer app:super-secret?provider=serper"
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError(f"proxy exposed Authorization: {secret}")

    monkeypatch.setattr(mod.requests, "post", fail)
    out = SearchTool(max_retries=3).execute({"query": "x"})
    assert calls == 1
    assert secret not in out
    assert "super-secret" not in out


def test_search_retries_transient_timeout(monkeypatch):
    import unirl.rollout.loop.tools.search as mod

    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)
    calls = 0

    def flaky(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise requests.Timeout("temporary")
        return _Resp({"organic": [{"title": "T", "link": "http://x", "snippet": "S"}]})

    monkeypatch.setattr(mod.requests, "post", flaky)
    assert "T" in SearchTool(max_retries=3).execute({"query": "x"})
    assert calls == 2


def test_search_does_not_retry_permanent_http_error(monkeypatch):
    import unirl.rollout.loop.tools.search as mod

    calls = 0

    def forbidden(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _Resp(status_code=403)

    monkeypatch.setattr(mod.requests, "post", forbidden)
    out = SearchTool(max_retries=3).execute({"query": "x"})
    assert calls == 1
    assert out.startswith("[search] error")
    assert "403" not in out


# --------------------------------------------------------------------------- #
# VisitTool
# --------------------------------------------------------------------------- #


def test_visit_schema_and_raw_fallback(monkeypatch):
    import unirl.rollout.loop.tools.visit as mod

    monkeypatch.delenv("SUMMARY_URL", raising=False)
    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: _Resp(text="PAGE BODY"))
    tool = VisitTool()  # no summarizer endpoint -> returns raw content
    assert tool.json_schema()["function"]["name"] == "visit"
    out = tool.execute({"url": "http://x", "goal": "g"})
    assert "PAGE BODY" in out


def test_visit_uses_polaris_jina_post_contract(monkeypatch):
    import unirl.rollout.loop.tools.visit as mod
    from unirl.rollout.loop.tools.polaris import POLARIS_JINA_URL

    monkeypatch.delenv("SUMMARY_URL", raising=False)
    monkeypatch.setenv("POLARIS_APP_ID", "visit-app")
    monkeypatch.setenv("POLARIS_APP_KEY", "visit-key")
    monkeypatch.setenv("JINA_PROVIDER", "jina_ai")
    monkeypatch.setenv("JINA_READER_URL", "https://stale.example/credential-sink")
    monkeypatch.setattr(
        mod.requests,
        "get",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("direct GET must not be used")),
    )

    def fake_post(url, json=None, headers=None, timeout=None):
        assert url == POLARIS_JINA_URL
        assert json == {"url": "https://example.com/page"}
        assert headers["Authorization"] == (
            "Bearer visit-app:visit-key?provider=jina_ai&timeout=60"
        )
        assert timeout == 65.0
        return _Resp(
            payload={"code": 200, "data": {"content": "POLARIS PAGE"}},
            text='{"code":200,"data":{"content":"POLARIS PAGE"}}',
        )

    monkeypatch.setattr(mod.requests, "post", fake_post)
    out = VisitTool().execute({"url": "https://example.com/page", "goal": "g"})
    assert "POLARIS PAGE" in out
    assert '"code"' not in out


def test_visit_rejects_polaris_error_envelope_without_retry(monkeypatch):
    import unirl.rollout.loop.tools.visit as mod

    monkeypatch.setenv("POLARIS_APP_ID", "visit-app")
    monkeypatch.setenv("POLARIS_APP_KEY", "visit-key")
    monkeypatch.setenv("JINA_PROVIDER", "jina_ai")
    calls = 0

    def application_error(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _Resp(payload={"code": 422, "data": None}, text='{"code":422,"data":null}')

    monkeypatch.setattr(mod.requests, "post", application_error)
    out = VisitTool(max_read_retries=3).execute({"url": "https://bad", "goal": "g"})
    assert calls == 1
    assert "could not be accessed" in out
    assert "422" not in out and '"code"' not in out


def test_visit_summarizes(monkeypatch):
    import unirl.rollout.loop.tools.visit as mod

    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: _Resp(text="LONG PAGE"))
    monkeypatch.setattr(mod.requests, "post", lambda *a, **k: _chat("SUMMARY"))
    out = VisitTool(endpoint="http://judge/v1/chat/completions", model="m").execute(
        {"url": "http://x", "goal": "g"}
    )
    assert "SUMMARY" in out


def test_visit_failure_is_clean(monkeypatch):
    import unirl.rollout.loop.tools.visit as mod

    # Every Jina read fails -> the tool must surface a clean "couldn't access" message,
    # NOT leak the raw [visit] sentinel / HTTP error disguised as useful info (LIN-564).
    monkeypatch.setattr(mod.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(
        mod.requests, "get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("422 Client Error"))
    )
    out = VisitTool(max_read_retries=1).execute({"url": "http://dead", "goal": "g"})
    assert "could not be accessed" in out
    assert "[visit]" not in out  # no disguised-failure leak
    assert "422" not in out  # raw HTTP error not surfaced to the model


# --------------------------------------------------------------------------- #
# LLMJudgeRewardScorer
# --------------------------------------------------------------------------- #


def _judge():
    return LLMJudgeRewardScorer(
        config=LLMJudgeSpec(endpoint="http://j/v1/chat/completions", model="m"), base_device="cpu"
    )


def _req(prediction, answer):
    return RewardRequest(
        primitives={"text": Texts(texts=["q"])},
        generated={"text": Texts(texts=[prediction])},
        metadata=[{"answer": answer}],
    )


def test_judge_verdict_parsing(monkeypatch):
    import unirl.reward.local.llm_judge as mod

    scorer = _judge()
    monkeypatch.setattr(mod.requests, "post", lambda *a, **k: _chat("Correct."))
    assert scorer._compute_model_rewards(_req("42", "42")) == [1.0]
    monkeypatch.setattr(mod.requests, "post", lambda *a, **k: _chat("Incorrect"))
    assert scorer._compute_model_rewards(_req("41", "42")) == [0.0]


def test_judge_missing_answer_scores_zero():
    scorer = _judge()
    req = RewardRequest(
        primitives={"text": Texts(texts=["q"])},
        generated={"text": Texts(texts=["x"])},
        metadata=[None],
    )
    assert scorer._compute_model_rewards(req) == [0.0]


def test_extract_answer_require_tag(monkeypatch):
    from unirl.trainer.agentic import _extract_answer

    # <answer> tags always extract the span
    assert _extract_answer("<think>reasoning</think>\n<answer>42</answer>") == "42"
    # no tags, default (tolerant — the math/calc \boxed path): fall back to whole text
    monkeypatch.delenv("REQUIRE_ANSWER_TAG", raising=False)
    assert _extract_answer("the answer is 42") == "the answer is 42"
    # no tags with REQUIRE_ANSWER_TAG (deep-research): scored as no answer -> "" -> reward 0
    monkeypatch.setenv("REQUIRE_ANSWER_TAG", "1")
    assert _extract_answer("the answer is 42") == ""
    assert _extract_answer("<answer>42</answer>") == "42"  # tags still honored
