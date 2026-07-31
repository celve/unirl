"""SearchTool — batched web search via Serper or SerpApi (LIN-519, hardened).

A concrete :class:`~unirl.rollout.loop.tools.tool.Tool` for the deep-research
agent: given an array of query strings it returns the top web results per query
as text. Two providers, selected by ``$SEARCH_PROVIDER`` (or the constructor):

- ``serper``  (default): Serper — ``POST serper.dev`` with an ``X-API-KEY`` header.
- ``serpapi``          : SerpApi — ``GET serpapi.com`` with an ``api_key`` param.

The serper endpoint is overridable for gateways/proxies that speak the same
response shape. ``$SERPER_AUTH=polaris`` selects the fixed internal gateway and
uses ``$POLARIS_APP_ID`` + ``$POLARIS_APP_KEY`` to synthesize the provider-scoped
bearer token at request time. Legacy bearer and public Serper remain supported.

Both read the API key from ``$SERPER_KEY_ID``. ``execute`` is synchronous and
thread-safe (it holds no state) so it runs cleanly under
concurrent trajectory threads (:meth:`ToolEnvironment.step`).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import requests

from unirl.rollout.loop.tools.polaris import (
    POLARIS_SERPER_URL,
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

_SERPER_URL = "https://google.serper.dev/search"
_SERPAPI_URL = "https://serpapi.com/search"


@dataclass
class _SearchOutcome:
    text: str
    ok: bool
    failure: str | None
    diagnostics: RequestDiagnostics


class SearchTool(Tool):
    """Google web search via Serper or SerpApi. Accepts one or more queries;
    returns the top results per query as text. Requires ``$SERPER_KEY_ID``;
    ``$SEARCH_PROVIDER=serpapi`` switches from Serper to SerpApi."""

    name = "search"

    def __init__(
        self,
        *,
        top_k: int = 10,
        timeout: float = 65.0,
        provider: str = "serper",
        max_retries: int = 4,
        gateway_max_in_flight: int = 2,
        retry_base_s: float = 0.5,
        retry_cap_s: float = 8.0,
        retry_after_cap_s: float = 30.0,
        cache_ttl_s: float = 900.0,
        cache_max_entries: int = 2048,
    ) -> None:
        self._top_k = int(top_k)
        self._timeout = float(timeout)
        self._provider = os.environ.get("SEARCH_PROVIDER", provider).lower()
        self._max_retries = max(1, int(max_retries))
        self._retry_base_s = max(0.0, float(retry_base_s))
        self._retry_cap_s = max(0.0, float(retry_cap_s))
        self._retry_after_cap_s = max(0.0, float(retry_after_cap_s))
        self._gateway_limiter = get_gateway_semaphore(gateway_max_in_flight)
        self._cache = SuccessTTLCache(cache_ttl_s, cache_max_entries)

    def json_schema(self) -> Dict[str, Any]:
        # Description/schema aligned verbatim to AReaL tongyi_deepresearch (LIN-564): the
        # neutral "Accepts multiple queries" wording, NOT an instruction to batch. Our prior
        # "Use multiple complementary queries in one call" made the base model cram ~3 queries
        # into ONE call (num_search≈1) instead of iterating (search→read→reformulate), which
        # both lowered the num_search start point and starved GRPO of the deep-search behavior
        # to amplify. AReaL's base Qwen3-1.7B issues ~3 separate calls at step 0 with this text.
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Perform Google web searches then returns a string of the top search "
                    "results. Accepts multiple queries."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "array",
                            "items": {"type": "string", "description": "The search query."},
                            "minItems": 1,
                            "description": "The list of search queries.",
                        }
                    },
                    "required": ["query"],
                },
            },
        }

    def execute(self, arguments: Dict[str, Any]) -> str:
        return self.execute_with_info(arguments).text

    def execute_with_info(self, arguments: Dict[str, Any]) -> ToolExecutionResult:
        queries = arguments.get("query")
        if isinstance(queries, str):
            queries = [queries]
        if not isinstance(queries, list) or not queries:
            raise ValueError("search requires a non-empty 'query' string or array")
        diagnostics = RequestDiagnostics(tool=self.name, provider=self._diagnostic_provider())
        texts: List[str] = []
        for query in queries:
            outcome = self._search_one(str(query))
            texts.append(outcome.text)
            diagnostics.merge(outcome.diagnostics)
        return ToolExecutionResult(
            text="\n=======\n".join(texts),
            diagnostics=diagnostics.as_dict(),
        )

    def _diagnostic_provider(self) -> str:
        if self._provider == "serper" and os.environ.get("SERPER_AUTH", "").lower() == "polaris":
            return "polaris"
        return self._provider

    def _cache_key(self, query: str) -> tuple:
        auth_mode = os.environ.get("SERPER_AUTH", "").lower()
        if self._provider == "serpapi":
            endpoint = _SERPAPI_URL
        elif auth_mode == "polaris":
            endpoint = POLARIS_SERPER_URL
        else:
            endpoint = os.environ.get("SERPER_URL", _SERPER_URL)
        # Search treats inconsequential surrounding/repeated whitespace as the
        # same request while preserving case and the original model-visible text.
        normalized_query = " ".join(query.split())
        return (self._provider, auth_mode, endpoint, self._top_k, normalized_query)

    def _request_organic(self, query: str, diagnostics: RequestDiagnostics) -> list:
        """Perform exactly one provider attempt and return organic result dicts."""

        key = os.environ.get("SERPER_KEY_ID", "")
        if self._provider == "serpapi":
            diagnostics.request_count += 1
            resp = requests.get(
                _SERPAPI_URL,
                params={"q": query, "engine": "google", "api_key": key, "num": self._top_k},
                timeout=self._timeout,
            )
        else:
            auth_mode = os.environ.get("SERPER_AUTH", "").lower()
            headers = {"Content-Type": "application/json"}
            if auth_mode == "polaris":
                # Never send Polaris app credentials to an environment-overridden URL.
                url = POLARIS_SERPER_URL
                headers = polaris_headers("serper")
                if headers is None:
                    raise PolarisAuthError("Serper Polaris authentication is not configured")
                with self._gateway_limiter:
                    diagnostics.request_count += 1
                    resp = requests.post(
                        url,
                        json={"q": query, "num": self._top_k},
                        headers=headers,
                        timeout=self._timeout,
                    )
            else:
                url = os.environ.get("SERPER_URL", _SERPER_URL)
                if auth_mode == "bearer":
                    if not key:
                        raise PolarisAuthError("Serper bearer authentication is not configured")
                    headers["Authorization"] = f"Bearer {key}"
                else:
                    headers["X-API-KEY"] = key
                diagnostics.request_count += 1
                resp = requests.post(
                    url,
                    json={"q": query, "num": self._top_k},
                    headers=headers,
                    timeout=self._timeout,
                )

        http_error = response_error(resp)
        if http_error is not None:
            raise http_error
        diagnostics.record_status(f"http_{int(resp.status_code)}")
        try:
            payload = resp.json()
        except (TypeError, ValueError) as exc:
            raise ProviderRequestError("transient", "malformed_response") from exc
        if not isinstance(payload, dict):
            raise ProviderRequestError("transient", "malformed_response")

        # Polaris may return either the Serper body directly or a provider error
        # envelope.  Successful wrapped data is tolerated for forward compatibility.
        if "code" in payload:
            retry_after_s = retry_after_from_response(resp)
            app_failure = application_error(payload.get("code"), retry_after_s=retry_after_s)
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
            if isinstance(payload.get("data"), dict):
                payload = payload["data"]

        organic_key = "organic_results" if self._provider == "serpapi" else "organic"
        organic = payload.get(organic_key) or []
        if not isinstance(organic, list) or any(not isinstance(page, dict) for page in organic):
            raise ProviderRequestError("transient", "malformed_response")
        return organic

    @staticmethod
    def _failure_text(failure: str) -> str:
        if failure == "transient":
            return (
                "[search] temporarily unavailable. Do not repeat the identical request; "
                "use a different query or source."
            )
        if failure == "auth":
            return "[search] error: search service authentication failed. Use a different source."
        return "[search] error: request failed. Try a different query or source."

    def _search_one(self, query: str) -> _SearchOutcome:
        provider = self._diagnostic_provider()

        def load() -> tuple[_SearchOutcome, bool]:
            diagnostics = RequestDiagnostics(tool=self.name, provider=provider)
            saw_transient = False
            for attempt in range(self._max_retries):
                try:
                    organic = self._request_organic(query, diagnostics)
                except Exception as exc:  # noqa: BLE001 — classify without exposing the exception
                    failure = classify_request_exception(exc)
                    diagnostics.record_status(failure.status_key)
                    if failure.category == "transient" and attempt + 1 < self._max_retries:
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
                        _SearchOutcome(
                            text=self._failure_text(failure.category),
                            ok=False,
                            failure=failure.category,
                            diagnostics=diagnostics,
                        ),
                        False,
                    )

                diagnostics.success_count += 1
                if saw_transient:
                    diagnostics.recovered_transient_count += 1
                if not organic:
                    text = f"No results for {query!r}. Try a less specific query."
                else:
                    lines: List[str] = []
                    for i, page in enumerate(organic[: self._top_k], start=1):
                        title = page.get("title", "")
                        link = page.get("link", "")
                        snippet = page.get("snippet", "")
                        lines.append(f"{i}. [{title}]({link})\n{snippet}")
                    text = f"Results for {query!r}:\n" + "\n\n".join(lines)
                return (
                    _SearchOutcome(
                        text=text,
                        ok=True,
                        failure=None,
                        diagnostics=diagnostics,
                    ),
                    True,
                )

            raise AssertionError("unreachable retry loop")

        outcome, source = self._cache.get_or_compute(self._cache_key(query), load)
        if source == "load":
            return outcome

        # A cache/single-flight waiter performed no transport request itself.
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
        return _SearchOutcome(
            text=outcome.text,
            ok=outcome.ok,
            failure=outcome.failure,
            diagnostics=diagnostics,
        )
