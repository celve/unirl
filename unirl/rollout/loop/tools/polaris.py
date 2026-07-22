"""Shared authentication helpers for the Polaris web-tool gateway.

Credentials are resolved at request time so pod-local rotation takes effect
without rebuilding tool objects. Secret values are never included in errors.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import requests

_PROVIDER_PREFIX = {"serper": "SERPER", "jina_ai": "JINA"}
POLARIS_GATEWAY_URL = "http://trpc-gpt-eval.production.polaris:8080"
POLARIS_SERPER_URL = f"{POLARIS_GATEWAY_URL}/search"
POLARIS_JINA_URL = f"{POLARIS_GATEWAY_URL}/"


def polaris_headers(provider: str) -> Optional[Dict[str, str]]:
    """Return Polaris JSON/auth headers, or ``None`` when no app pair is set.

    The common ``POLARIS_APP_*`` pair is canonical. Provider-specific
    ``SERPER_APP_*`` / ``JINA_APP_*`` values are only a legacy fallback when the
    common pair is entirely absent. A half-configured pair is rejected without
    echoing either credential.
    """
    if provider not in _PROVIDER_PREFIX:
        raise ValueError(f"unsupported Polaris provider: {provider!r}")
    prefix = _PROVIDER_PREFIX[provider]
    common_id = os.environ.get("POLARIS_APP_ID", "")
    common_key = os.environ.get("POLARIS_APP_KEY", "")
    if common_id or common_key:
        app_id, app_key = common_id, common_key
    else:
        app_id = os.environ.get(f"{prefix}_APP_ID", "")
        app_key = os.environ.get(f"{prefix}_APP_KEY", "")
    if bool(app_id) != bool(app_key):
        raise RuntimeError(f"{prefix} Polaris authentication requires both app id and app key")
    if not app_id:
        return None
    try:
        timeout = int(os.environ.get("POLARIS_PROVIDER_TIMEOUT", "60"))
    except ValueError as exc:
        raise RuntimeError("POLARIS_PROVIDER_TIMEOUT must be a positive integer") from exc
    if timeout <= 0:
        raise RuntimeError("POLARIS_PROVIDER_TIMEOUT must be a positive integer")
    token = f"{app_id}:{app_key}?provider={provider}&timeout={timeout}"
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def is_transient_request_error(exc: BaseException) -> bool:
    """Whether an HTTP-layer failure is safe and useful to retry.

    Configuration/parser errors and permanent 4xx responses fail immediately;
    timeouts, connection failures, 429s, and server failures are retried.
    """
    if isinstance(exc, requests.HTTPError):
        response = exc.response
        if response is None:
            return False
        status = response.status_code
        return status == 429 or status >= 500
    return isinstance(exc, (requests.Timeout, requests.ConnectionError))


__all__ = [
    "POLARIS_GATEWAY_URL",
    "POLARIS_JINA_URL",
    "POLARIS_SERPER_URL",
    "is_transient_request_error",
    "polaris_headers",
]
