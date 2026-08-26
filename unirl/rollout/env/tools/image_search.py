"""ImageSearchTool — batched image search through the Polaris gateway (LIN-892).

Serper serves images separately from web results, so this is a sibling of
:class:`~unirl.rollout.env.tools.search.SearchTool` rather than a mode of it: the gateway
proxies ``/images`` to the provider and the response carries an ``images`` array instead
of ``organic``.

Results are returned as text — a page link, the image URL, its pixel dimensions and the
source — which keeps the tool inside the ``execute -> str`` contract and needs no
framework change. The image URL is the handle a later stage resolves to pixels (LIN-893);
the dimensions are here because a candidate that becomes a diffusion reference is chosen
partly on resolution and aspect.

``num`` and Serper's ``tbs`` filter are constructor arguments rather than schema fields:
wanting reference-quality images is a property of the workload, not a per-call decision,
and every extra field is surface the policy has to learn.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import aiohttp

from unirl.rollout.env.tools.base import Tool
from unirl.rollout.env.tools.polaris import IMAGES_PATH, client_timeout, post_json, run_sync

_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 0.5
_TOTAL_TIMEOUT_SECONDS = 75.0


class ImageSearchTool(Tool):
    """Google image search via the gateway's ``serper`` provider; returns URLs and metadata."""

    name = "image_search"

    def __init__(
        self,
        *,
        top_k: int = 10,
        image_size: str = "isz:l",
        location: str = "United States",
        hl: str = "en",
    ) -> None:
        self._top_k = max(1, int(top_k))
        self._image_size = str(image_size)
        self._location = str(location)
        self._hl = str(hl)
        self._timeout = client_timeout(_TOTAL_TIMEOUT_SECONDS)

    def json_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Performs batched image searches: supply an array 'query'; the tool "
                    "returns candidate images for each query with their URL, pixel "
                    "dimensions and source page. Use multiple complementary queries in "
                    "one call."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": ("Array of query strings describing the images to find."),
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
            return "[ImageSearch] Invalid request format: Input must be a JSON object containing 'query' field"
        queries = [query] if isinstance(query, str) else list(query or [])
        if not queries:
            return "[ImageSearch] Invalid request format: 'query' must be a non-empty string or array"
        return run_sync(lambda: self._search_many([str(q) for q in queries]))

    async def _search_many(self, queries: List[str]) -> str:
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            responses = await asyncio.gather(*(self._search_one(session, q) for q in queries))
        return "\n=======\n".join(responses)

    async def _search_one(self, session: aiohttp.ClientSession, query: str) -> str:
        payload: Dict[str, Any] = {
            "q": query,
            "location": self._location,
            "hl": self._hl,
            "num": self._top_k,
        }
        if self._image_size:
            payload["tbs"] = self._image_size

        for attempt in range(_MAX_ATTEMPTS):
            try:
                body = await post_json(session, IMAGES_PATH, payload, provider="serper")
            except Exception:
                if attempt + 1 < _MAX_ATTEMPTS:
                    await asyncio.sleep(_RETRY_BACKOFF_SECONDS * (2**attempt))
                continue
            images = body.get("images") if isinstance(body, dict) else None
            if not images:
                return f"No images found for query: '{query}'. Use a less specific query."
            return self._render(query, images[: self._top_k])
        return f"[ImageSearch] Timeout or error for '{query}'; please try again later."

    @staticmethod
    def _render(query: str, images: List[Dict[str, Any]]) -> str:
        entries = []
        for index, image in enumerate(images, start=1):
            title = image.get("title", "")
            link = image.get("link", "")
            image_url = image.get("imageUrl", "")
            width = image.get("imageWidth")
            height = image.get("imageHeight")
            source = image.get("source") or image.get("domain") or ""
            size = f"{width}x{height}" if width and height else "unknown size"
            attribution = f"{size} · {source}" if source else size
            entries.append(f"{index}. [{title}]({link})\n   image: {image_url}\n   {attribution}")
        return f"Image results for '{query}' ({len(entries)}):\n\n" + "\n\n".join(entries)


__all__ = ["ImageSearchTool"]
