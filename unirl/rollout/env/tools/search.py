"""SearchTool — batched web search through the Polaris gateway (LIN-892).

Given an array of query strings, returns the top web results per query as text.
Credentials and endpoint come from :mod:`~unirl.rollout.env.tools.polaris`; there is no
per-provider key to configure.

The schema wording, locale choice, retry budget and result formatting are inherited
verbatim from AReaL's Tongyi DeepResearch ``tool_search.py``. **Do not reword the output
while LIN-714's AReaL comparison is running** — those strings are the observation the
policy reads, and the comparison is against AReaL's own reward curve. The failure text
deliberately omits the exception, so a transport error cannot echo a credential.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import aiohttp

from unirl.rollout.env.tools.base import Tool
from unirl.rollout.env.tools.polaris import POLARIS_URL, SEARCH_PATH, polaris_headers, run_sync

_MAX_ATTEMPTS = 5
_RETRY_BACKOFF_SECONDS = 0.5
_CONNECT_TIMEOUT_SECONDS = 10.0
_READ_TIMEOUT_SECONDS = 30.0
_TOTAL_TIMEOUT_SECONDS = 45.0


class SearchTool(Tool):
    """Batched Google web search via the gateway's ``serper`` provider."""

    name = "search"

    def __init__(self) -> None:
        self._timeout = aiohttp.ClientTimeout(
            total=_TOTAL_TIMEOUT_SECONDS,
            connect=_CONNECT_TIMEOUT_SECONDS,
            sock_connect=_CONNECT_TIMEOUT_SECONDS,
            sock_read=_READ_TIMEOUT_SECONDS,
        )

    def json_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Performs batched web searches: supply an array 'query'; the tool "
                    "retrieves the top 10 results for each query in one call."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Array of query strings. Include multiple complementary "
                                "search queries in a single call."
                            ),
                        }
                    },
                    "required": ["query"],
                },
            },
        }

    def execute(self, arguments: Dict[str, Any]) -> str:
        try:
            query = arguments["query"]
        except Exception:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"
        if isinstance(query, str):
            return run_sync(lambda: self._search_one(query))
        assert isinstance(query, list)
        return run_sync(lambda: self._search_many(query))

    async def _search_many(self, queries: List[Any]) -> str:
        responses = await asyncio.gather(*(self._search_one(query) for query in queries))
        return "\n=======\n".join(responses)

    async def _search_one(self, query: str) -> str:
        payload = (
            {"q": query, "location": "China", "gl": "cn", "hl": "zh-cn"}
            if any("\u4e00" <= char <= "\u9fff" for char in query)
            else {"q": query, "location": "United States", "gl": "us", "hl": "en"}
        )
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            for attempt in range(_MAX_ATTEMPTS):
                try:
                    async with session.post(
                        f"{POLARIS_URL}{SEARCH_PATH}",
                        json=payload,
                        headers=polaris_headers("serper"),
                    ) as response:
                        text = await response.text()
                        if response.status < 200 or response.status >= 300:
                            raise RuntimeError("search service returned a non-success status")
                        try:
                            results = json.loads(text)
                        except Exception:
                            return f"[Search] Failed to parse response for '{query}'."
                        if "organic" not in results:
                            return f"No results found for query: '{query}'. Use a less specific query."
                        snippets = []
                        for index, page in enumerate(results.get("organic", []), start=1):
                            date = f"\nDate published: {page['date']}" if page.get("date") else ""
                            source = f"\nSource: {page['source']}" if page.get("source") else ""
                            snippet = f"\n{page['snippet']}" if page.get("snippet") else ""
                            item = (
                                f"{index}. [{page.get('title', '')}]({page.get('link', '')}){date}{source}\n{snippet}"
                            ).replace("Your browser can't play this video.", "")
                            snippets.append(item)
                        return (
                            f"A Google search for '{query}' found {len(snippets)} results:"
                            "\n\n## Web Results\n" + "\n\n".join(snippets)
                        )
                except Exception:
                    if attempt + 1 < _MAX_ATTEMPTS:
                        await asyncio.sleep(_RETRY_BACKOFF_SECONDS)
        return "Google search Timeout or error; return None, Please try again later."


__all__ = ["SearchTool"]
