"""VisitTool — read webpage(s) and summarize toward a goal (LIN-519, hardened).

A concrete :class:`~unirl.rollout.loop.tools.tool.Tool` for the deep-research
agent: fetch a URL's content with direct Jina or the Polaris Jina provider and
summarize the parts relevant to a stated goal with an OpenAI-compatible LLM
(hosted out-of-band; ``$SUMMARY_URL`` / ``$SUMMARY_MODEL`` or the constructor
args — the same endpoint the judge uses). ``execute`` is synchronous and
thread-safe so it runs on concurrent trajectory threads (:meth:`ToolEnvironment.step`) across
concurrent trajectories. If no summarizer is configured it returns the truncated
raw page content, so the tool is usable without a summarizer for smoke tests.

Hardened toward AReaL's tongyi_deepresearch ``tool_visit.py``: Jina reads and the
summarizer call retry on transient failures, and the summarizer returns a
structured ``evidence`` / ``summary`` extraction (JSON-tolerant parse) rather than
free text — higher-signal observations for the policy.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

import requests

from unirl.rollout.loop.tools.polaris import (
    POLARIS_JINA_URL,
    PolarisAuthError,
    ProviderRequestError,
    RequestDiagnostics,
    SuccessTTLCache,
    application_error,
    classify_request_exception,
    full_jitter_delay,
    get_gateway_semaphore,
    optional_application_status_error,
    polaris_headers,
    response_error,
    retry_after_from_response,
)
from unirl.rollout.loop.tools.tool import Tool, ToolExecutionResult

_JINA_READ = "https://r.jina.ai/"
_MAX_CACHED_CONTENT_CHARS = 14000
# Structured extractor (mirrors AReaL's EXTRACTOR_PROMPT): evidence + summary.
_EXTRACT_PROMPT = (
    "Process the following webpage content and extract the information relevant "
    "to the goal.\n\n"
    "## Webpage content\n{content}\n\n"
    "## Goal\n{goal}\n\n"
    "## Task\n"
    "1. evidence: extract the most relevant facts, figures, dates, and quotes "
    "from the content — keep the full original context where possible.\n"
    "2. summary: organize it into a concise paragraph, judging its contribution "
    "to the goal.\n\n"
    'Output ONLY a JSON object with string keys "evidence" and "summary".'
)


@dataclass
class _ReadOutcome:
    content: str
    ok: bool
    failure: str | None
    diagnostics: RequestDiagnostics


@dataclass
class _VisitOutcome:
    text: str
    diagnostics: RequestDiagnostics


def _extract_json(raw: str) -> Optional[dict]:
    """Best-effort JSON parse: strip code fences, else grab the outer ``{...}``."""
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception:
        left, right = raw.find("{"), raw.rfind("}")
        if left != -1 and right != -1 and left <= right:
            try:
                return json.loads(raw[left : right + 1])
            except Exception:
                return None
    return None


class VisitTool(Tool):
    """Visit URL(s) via the Jina reader and summarize the content toward a goal.
    Requires ``$JINA_API_KEYS``; the summarizer endpoint comes from ``$SUMMARY_URL``
    / ``$SUMMARY_MODEL`` or the constructor args."""

    name = "visit"

    def __init__(
        self,
        *,
        endpoint: str = "",
        model: str = "",
        timeout: float = 65.0,
        reader_provider: str = "direct",
        reader_url: str = "",
        # Page content sent to the summarizer per URL. Must fit the summarizer's own
        # context: our judge/summarizer serves at ctx 8192, so ~14000 chars (~4000
        # tokens) + the extractor prompt leaves room for the evidence/summary output.
        # Larger (the old 90000) overran 8192 -> the summarize call failed -> the tool
        # dumped the RAW page (up to 90000 chars, ×N for multi-URL), overflowing the
        # policy's 32768 context on the next turn (LIN-564).
        max_content_chars: int = 14000,
        max_read_retries: int = 4,
        max_summary_retries: int = 2,
        gateway_max_in_flight: int = 2,
        retry_base_s: float = 0.5,
        retry_cap_s: float = 8.0,
        retry_after_cap_s: float = 30.0,
        cache_ttl_s: float = 3600.0,
        cache_max_entries: int = 1024,
    ) -> None:
        self._endpoint = endpoint
        self._model = model
        self._timeout = float(timeout)
        self._reader_provider = os.environ.get("JINA_PROVIDER", reader_provider).lower()
        self._reader_url = reader_url
        self._max_content_chars = int(max_content_chars)
        self._max_read_retries = max(1, int(max_read_retries))
        self._max_summary_retries = max(1, int(max_summary_retries))
        self._retry_base_s = max(0.0, float(retry_base_s))
        self._retry_cap_s = max(0.0, float(retry_cap_s))
        self._retry_after_cap_s = max(0.0, float(retry_after_cap_s))
        self._gateway_limiter = get_gateway_semaphore(gateway_max_in_flight)
        self._cache = SuccessTTLCache(cache_ttl_s, cache_max_entries)

    def json_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Visit webpage(s) and return a summary of the content relevant to a goal.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": ["string", "array"],
                            "items": {"type": "string"},
                            "description": "The URL, or an array of URLs, to visit.",
                        },
                        "goal": {
                            "type": "string",
                            "description": "The specific information to extract from the page(s).",
                        },
                    },
                    "required": ["url", "goal"],
                },
            },
        }

    def execute(self, arguments: Dict[str, Any]) -> str:
        return self.execute_with_info(arguments).text

    def execute_with_info(self, arguments: Dict[str, Any]) -> ToolExecutionResult:
        url = arguments.get("url")
        goal = str(arguments.get("goal", ""))
        urls: List[str] = [url] if isinstance(url, str) else list(url or [])
        if not urls:
            raise ValueError("visit requires a 'url' string or array")
        diagnostics = RequestDiagnostics(tool=self.name, provider=self._diagnostic_provider())
        texts: List[str] = []
        for item in urls:
            result = self._visit_one(str(item), goal)
            texts.append(result.text)
            item_diagnostics = result.diagnostics
            diagnostics.merge(item_diagnostics)
        return ToolExecutionResult(
            text="\n=======\n".join(texts),
            diagnostics=diagnostics.as_dict(),
        )

    def _diagnostic_provider(self) -> str:
        return "polaris" if self._reader_provider == "jina_ai" else "jina"

    def _cache_key(self, url: str) -> tuple:
        endpoint = POLARIS_JINA_URL if self._reader_provider == "jina_ai" else _JINA_READ
        try:
            parsed = urlsplit(url)
            normalized_url = urlunsplit(
                (parsed.scheme, parsed.netloc, parsed.path, parsed.query, "")
            )
        except ValueError:
            normalized_url = url.split("#", 1)[0]
        return (self._reader_provider, endpoint, normalized_url)

    def _visit_one(self, url: str, goal: str) -> _VisitOutcome:
        read = self._read_with_info(url)
        if not read.ok:
            # LIN-564 (AReaL parity): don't leak the raw sentinel (e.g. a Jina 422)
            # into the model's context disguised as "useful information" — that fed a
            # re-visit loop. Temporary exhaustion is deliberately distinct and tells
            # the policy not to hammer the exact same request again.
            if read.failure == "transient":
                message = (
                    "The webpage reader is temporarily unavailable. Do not repeat the "
                    "identical request; use a different URL or source."
                )
            elif read.failure == "auth":
                message = "The webpage reader could not authenticate. Please use a different source."
            else:
                message = (
                    "The provided webpage content could not be accessed. "
                    "Please check the URL or try a different source."
                )
            return _VisitOutcome(
                text=(
                    f"The useful information in {url} for goal {goal}:\n"
                    f"{message}"
                ),
                diagnostics=read.diagnostics,
            )
        summary = self._summarize(read.content[: self._max_content_chars], goal)
        return _VisitOutcome(
            text=f"The useful information in {url} for goal {goal}:\n{summary}",
            diagnostics=read.diagnostics,
        )

    def _request_content(self, url: str, diagnostics: RequestDiagnostics) -> str:
        """Perform exactly one Jina attempt; return at most 14k successful chars."""

        if self._reader_provider == "jina_ai":
            headers = polaris_headers("jina_ai")
            if headers is None:
                raise PolarisAuthError("Jina Polaris authentication is not configured")
            with self._gateway_limiter:
                diagnostics.request_count += 1
                resp = requests.post(
                    # Never send Polaris app credentials to an environment-
                    # overridden endpoint.
                    POLARIS_JINA_URL,
                    json={"url": url},
                    headers=headers,
                    timeout=self._timeout,
                )
        elif self._reader_provider == "direct":
            key = os.environ.get("JINA_API_KEYS", "")
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            diagnostics.request_count += 1
            resp = requests.get(_JINA_READ + url, headers=headers, timeout=self._timeout)
        else:
            raise RuntimeError(f"unsupported Jina reader provider: {self._reader_provider!r}")

        http_error = response_error(resp)
        if http_error is not None:
            raise http_error
        diagnostics.record_status(f"http_{int(resp.status_code)}")

        if self._reader_provider == "jina_ai":
            try:
                payload = resp.json()
            except (TypeError, ValueError) as exc:
                raise ProviderRequestError("transient", "malformed_jina_envelope") from exc
            if not isinstance(payload, dict) or "code" not in payload:
                raise ProviderRequestError("transient", "malformed_jina_envelope")
            retry_after_s = retry_after_from_response(resp)
            app_failure = application_error(
                payload.get("code"), retry_after_s=retry_after_s
            )
            if app_failure is not None:
                raise app_failure
            diagnostics.record_status("app_200")
            if "status" in payload:
                status_failure = optional_application_status_error(
                    payload.get("status"),
                    prefix="app_status",
                    retry_after_s=retry_after_s,
                )
                if status_failure is not None:
                    raise status_failure
            data = payload.get("data")
            content = data.get("content") if isinstance(data, dict) else None
            if not isinstance(content, str) or not content.strip():
                raise ProviderRequestError("transient", "empty_jina_content")
        else:
            content = resp.text
            if not isinstance(content, str) or not content.strip():
                raise ProviderRequestError("transient", "empty_jina_content")
        return content[:_MAX_CACHED_CONTENT_CHARS]

    def _read_with_info(self, url: str) -> _ReadOutcome:
        provider = self._diagnostic_provider()

        def load() -> tuple[_ReadOutcome, bool]:
            diagnostics = RequestDiagnostics(tool=self.name, provider=provider)
            saw_transient = False
            for attempt in range(self._max_read_retries):
                try:
                    content = self._request_content(url, diagnostics)
                except Exception as exc:  # noqa: BLE001 — never expose raw request exceptions
                    failure = classify_request_exception(exc)
                    diagnostics.record_status(failure.status_key)
                    if failure.category == "transient" and attempt + 1 < self._max_read_retries:
                        saw_transient = True
                        diagnostics.retry_count += 1
                        time.sleep(
                            full_jitter_delay(
                                attempt,
                                retry_after_s=failure.retry_after_s,
                                base_s=self._retry_base_s,
                                cap_s=self._retry_cap_s,
                                retry_after_cap_s=self._retry_after_cap_s,
                            )
                        )
                        continue
                    if failure.category == "transient":
                        diagnostics.transient_exhausted_count += 1
                    elif failure.category == "auth":
                        diagnostics.auth_error_count += 1
                    else:
                        diagnostics.permanent_error_count += 1
                    return (
                        _ReadOutcome(
                            content="",
                            ok=False,
                            failure=failure.category,
                            diagnostics=diagnostics,
                        ),
                        False,
                    )

                diagnostics.success_count += 1
                if saw_transient:
                    diagnostics.recovered_transient_count += 1
                return (
                    _ReadOutcome(
                        content=content,
                        ok=True,
                        failure=None,
                        diagnostics=diagnostics,
                    ),
                    True,
                )
            raise AssertionError("unreachable retry loop")

        outcome, source = self._cache.get_or_compute(self._cache_key(url), load)
        if source == "load":
            return outcome

        diagnostics = RequestDiagnostics(tool=self.name, provider=provider)
        if outcome.ok:
            diagnostics.success_count = 1
            diagnostics.cache_hit_count = 1
        elif outcome.failure == "transient":
            diagnostics.transient_exhausted_count = 1
        elif outcome.failure == "auth":
            diagnostics.auth_error_count = 1
        else:
            diagnostics.permanent_error_count = 1
        return _ReadOutcome(
            content=outcome.content,
            ok=outcome.ok,
            failure=outcome.failure,
            diagnostics=diagnostics,
        )

    def _read(self, url: str) -> str:
        """Backward-compatible private adapter used by older smoke tests."""

        outcome = self._read_with_info(url)
        if outcome.ok:
            return outcome.content
        return f"[visit] failed to read {url}: request failed"

    def _summarize(self, content: str, goal: str) -> str:
        """Summarize toward the goal via the out-of-band LLM into a structured
        evidence/summary block. No endpoint -> raw (truncated) content. On repeated
        failure -> raw content, so a dead summarizer degrades rather than crashes."""
        endpoint = os.environ.get("SUMMARY_URL", self._endpoint)
        model = os.environ.get("SUMMARY_MODEL", self._model)
        if not endpoint:
            return content  # no summarizer configured — return raw (truncated) content
        headers = {"Content-Type": "application/json"}
        key = os.environ.get("SUMMARY_API_KEY", "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": _EXTRACT_PROMPT.format(goal=goal, content=content)}],
            "temperature": 0.2,
        }
        for _ in range(self._max_summary_retries):
            try:
                resp = requests.post(endpoint, json=payload, headers=headers, timeout=self._timeout)
                resp.raise_for_status()
                raw = str(resp.json()["choices"][0]["message"]["content"]).strip()
                obj = _extract_json(raw)
                if obj is not None:
                    evidence = str(obj.get("evidence", "")).strip()
                    summary = str(obj.get("summary", "")).strip()
                    if evidence or summary:
                        return f"Evidence:\n{evidence}\n\nSummary:\n{summary}"
                if raw:
                    return raw  # not JSON, but the model's text is still a usable summary
            except Exception:  # noqa: BLE001 — retry, then fall back to raw content
                pass
            time.sleep(0.5)
        return content  # summarizer failed after retries — raw truncated content
