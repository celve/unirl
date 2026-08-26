"""FetchTool — retrieve URL content through the Polaris gateway, unsummarised (LIN-892).

The gateway's ``jina_ai`` provider returns readable text for a page and a one-line
caption for an image URL, so this is how an agent inspects a candidate image without any
pixels reaching the sample: search returns handles, fetch describes them.

Unlike :class:`~unirl.rollout.env.tools.visit.VisitTool` there is no summariser and no
``goal`` — content comes back as-is, truncated at ``max_chars``. That makes it right for
image URLs and small pages and wrong for long articles, where ``visit`` should be used
instead.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import aiohttp

from unirl.rollout.env.tools.base import Tool
from unirl.rollout.env.tools.polaris import READER_PATH, client_timeout, post_json, run_sync

_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 0.5
_TOTAL_TIMEOUT_SECONDS = 75.0


class FetchTool(Tool):
    """Fetch URL content via the gateway's ``jina_ai`` provider and return it as text."""

    name = "fetch"

    def __init__(self, *, max_chars: int = 8000) -> None:
        self._max_chars = max(1, int(max_chars))
        self._timeout = client_timeout(_TOTAL_TIMEOUT_SECONDS)

    def json_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Fetch the content of one or more URLs and return it as text. For an "
                    "image URL this returns a description of the image, so it can be used "
                    "to inspect a candidate found by image_search."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": ["string", "array"],
                            "items": {"type": "string"},
                            "minItems": 1,
                            "description": "The URL, or an array of URLs, to fetch.",
                        }
                    },
                    "required": ["url"],
                },
            },
        }

    def execute(self, arguments: Dict[str, Any]) -> str:
        try:
            url = arguments["url"]
        except Exception:
            return "[Fetch] Invalid request format: Input must be a JSON object containing 'url' field"
        urls = [url] if isinstance(url, str) else list(url or [])
        if not urls:
            return "[Fetch] Invalid request format: 'url' must be a non-empty string or array"
        return run_sync(lambda: self._fetch_many([str(u) for u in urls]))

    async def _fetch_many(self, urls: List[str]) -> str:
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            responses = await asyncio.gather(*(self._fetch_one(session, u) for u in urls))
        return "\n=======\n".join(responses)

    async def _fetch_one(self, session: aiohttp.ClientSession, url: str) -> str:
        for attempt in range(_MAX_ATTEMPTS):
            try:
                body = await post_json(session, READER_PATH, {"url": url}, provider="jina_ai")
            except Exception:
                if attempt + 1 < _MAX_ATTEMPTS:
                    await asyncio.sleep(_RETRY_BACKOFF_SECONDS * (2**attempt))
                continue
            data = body.get("data") if isinstance(body, dict) else None
            content = data.get("content") if isinstance(data, dict) else None
            if not isinstance(content, str) or not content.strip():
                return f"[Fetch] No readable content at {url}."
            title = (data.get("title") or "").strip()
            heading = f"Content of {url}" + (f" ({title})" if title else "")
            return f"{heading}:\n{content[: self._max_chars]}"
        return f"[Fetch] Timeout or error for {url}; please try again later."


__all__ = ["FetchTool"]
