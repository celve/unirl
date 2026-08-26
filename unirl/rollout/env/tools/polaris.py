"""Polaris — the internal gateway the web tools in this package talk to (LIN-892).

One host fronts several providers, so a tool needs no per-provider key or endpoint.
Credentials are an app-id / app-key pair read from ``$POLARIS_APP_ID`` and
``$POLARIS_APP_KEY``; the provider and timeout ride as query parameters **inside** the
Bearer value rather than on the URL, which is the gateway's own grammar:

``Authorization: Bearer <app-id>:<app-key>?provider=serper&timeout=60``

Providers in use: ``serper`` for web search (``/search``) and image search
(``/images``), and ``jina_ai`` for page and image retrieval (``/``).

Transport only. Retry policy and result formatting stay with each tool, because they
differ on purpose — :class:`~unirl.rollout.env.tools.search.SearchTool` and
:class:`~unirl.rollout.env.tools.visit.VisitTool` reproduce AReaL's wording for the
deep-research comparison, while the image tools format for a VLM prompt-enhancer.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from typing import Any, Dict, TypeVar

import aiohttp

_T = TypeVar("_T")

POLARIS_URL = "http://trpc-gpt-eval.production.polaris:8080"
GATEWAY_TIMEOUT_SECONDS = 60

SEARCH_PATH = "/search"
IMAGES_PATH = "/images"
READER_PATH = "/"


def run_sync(factory: Callable[[], Awaitable[_T]]) -> _T:
    """Bridge an async implementation onto the synchronous :meth:`Tool.execute` contract.

    ``ToolEnvironment.step`` runs ``execute`` in an executor thread with no loop of its
    own, so ``asyncio.run`` is safe there and a running loop means a misrouted call.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())
    raise RuntimeError("Polaris tools require synchronous Tool.execute dispatch")


def polaris_headers(provider: str, *, timeout: int = GATEWAY_TIMEOUT_SECONDS) -> Dict[str, str]:
    """Gateway auth headers for ``provider``. Raises when credentials are absent."""
    app_id = os.environ.get("POLARIS_APP_ID", "")
    app_key = os.environ.get("POLARIS_APP_KEY", "")
    if not app_id or not app_key:
        raise RuntimeError("tool proxy credentials are not configured")
    authorization = f"Bearer {app_id}:{app_key}?provider={provider}&timeout={timeout}"
    return {"Authorization": authorization, "Content-Type": "application/json"}


def client_timeout(total: float, *, connect: float = 10.0) -> aiohttp.ClientTimeout:
    """A client timeout budget. Keep ``total`` above :data:`GATEWAY_TIMEOUT_SECONDS` so the
    gateway's own timeout fires first and the failure carries a server-side reason."""
    return aiohttp.ClientTimeout(total=total, connect=connect, sock_connect=connect, sock_read=total)


async def post_json(
    session: aiohttp.ClientSession,
    path: str,
    payload: Dict[str, Any],
    *,
    provider: str,
) -> Any:
    """POST ``payload`` to ``path`` through ``provider``; return the decoded body.

    The caller owns the session so a batched call can share one across ``asyncio.gather``.
    """
    async with session.post(f"{POLARIS_URL}{path}", json=payload, headers=polaris_headers(provider)) as response:
        if response.status < 200 or response.status >= 300:
            await response.read()
            raise RuntimeError(f"polaris {provider} returned status {response.status}")
        return await response.json(content_type=None)


def preflight() -> None:
    """One real search, raising on failure — call at launcher startup.

    Every tool here turns a transport failure into fallback text so the policy can
    recover, which means a missing credential or an unroutable gateway produces a
    healthy-looking rollout with no live retrieval behind it. That invalidated a LIN-714
    run before it was caught by hand.
    """

    async def probe() -> Any:
        async with aiohttp.ClientSession(timeout=client_timeout(30.0)) as session:
            return await post_json(session, SEARCH_PATH, {"q": "polaris preflight", "num": 1}, provider="serper")

    body = run_sync(probe)
    if not isinstance(body, dict) or "organic" not in body:
        raise RuntimeError("polaris preflight: search returned no organic results")


__all__ = [
    "GATEWAY_TIMEOUT_SECONDS",
    "IMAGES_PATH",
    "POLARIS_URL",
    "READER_PATH",
    "SEARCH_PATH",
    "client_timeout",
    "polaris_headers",
    "post_json",
    "preflight",
    "run_sync",
]
