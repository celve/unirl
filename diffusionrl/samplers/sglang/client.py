"""HTTP client for external SGLang diffusion service."""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SGLangClientError(RuntimeError):
    """Base error for SGLang client failures."""


class SGLangTimeoutError(SGLangClientError):
    """Raised when request/queue waits exceed configured timeout."""


class SGLangProtocolError(SGLangClientError):
    """Raised when server response is invalid for current client contract."""


class SGLangClient:
    """Thin HTTP client with capability handshake and bounded concurrency."""

    def __init__(
        self,
        *,
        server_url: str,
        handshake_timeout_s: float = 5.0,
        request_timeout_s: float = 60.0,
        max_retries: int = 1,
        retry_backoff_s: float = 0.5,
        max_outstanding_requests: int = 1,
        queue_timeout_s: Optional[float] = None,
        handshake_paths: Optional[List[str]] = None,
    ) -> None:
        if not server_url:
            raise ValueError("server_url must be non-empty")
        if max_outstanding_requests < 1:
            raise ValueError(f"max_outstanding_requests must be >= 1, got {max_outstanding_requests}")
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")

        self.server_url = server_url.rstrip("/")
        self.handshake_timeout_s = float(handshake_timeout_s)
        self.request_timeout_s = float(request_timeout_s)
        self.max_retries = int(max_retries)
        self.retry_backoff_s = float(retry_backoff_s)
        self.queue_timeout_s = (
            float(queue_timeout_s) if queue_timeout_s is not None else self.request_timeout_s
        )
        self.handshake_paths = handshake_paths or [
            "/capabilities",
            "/handshake",
            "/v1/models",
            "/v1/model/info",
            "/health",
        ]

        self._semaphore = threading.BoundedSemaphore(int(max_outstanding_requests))
        self._last_capabilities: Dict[str, Any] = {}

    def handshake(self) -> Dict[str, Any]:
        """Negotiate basic capability schema with server."""
        errors: List[str] = []
        for path in self.handshake_paths:
            try:
                payload = self.request_json("GET", path, timeout_s=self.handshake_timeout_s)
            except Exception as exc:  # pragma: no cover - exercised via fallback path
                errors.append(f"{path}: {exc}")
                continue

            capabilities = self._normalize_capabilities(payload)
            self._last_capabilities = capabilities
            return capabilities

        raise SGLangClientError(
            "SGLang capability handshake failed. "
            f"server={self.server_url}, tried={self.handshake_paths}, errors={errors}"
        )

    def request_json(
        self,
        method: str,
        path: str,
        *,
        payload: Optional[Dict[str, Any]] = None,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        timeout = self.request_timeout_s if timeout_s is None else float(timeout_s)
        request_path = path if path.startswith("/") else f"/{path}"
        url = urllib.parse.urljoin(f"{self.server_url}/", request_path.lstrip("/"))

        acquired = self._semaphore.acquire(timeout=self.queue_timeout_s)
        if not acquired:
            raise SGLangTimeoutError(
                f"SGLang request queue wait timeout after {self.queue_timeout_s:.2f}s"
            )

        try:
            return self._request_with_retries(method=method, url=url, payload=payload, timeout=timeout)
        finally:
            self._semaphore.release()

    def _request_with_retries(
        self,
        *,
        method: str,
        url: str,
        payload: Optional[Dict[str, Any]],
        timeout: float,
    ) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._request_once(method=method, url=url, payload=payload, timeout=timeout)
            except SGLangTimeoutError as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
            except SGLangClientError as exc:
                last_error = exc
                if attempt == self.max_retries or "HTTP 5" not in str(exc):
                    break

            sleep_s = self.retry_backoff_s * float(attempt + 1)
            if sleep_s > 0:
                time.sleep(sleep_s)

        if last_error is None:  # pragma: no cover
            raise SGLangClientError("Unknown SGLang request failure")
        raise last_error

    def _request_once(
        self,
        *,
        method: str,
        url: str,
        payload: Optional[Dict[str, Any]],
        timeout: float,
    ) -> Dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url=url, data=body, headers=headers, method=method.upper())
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            msg = f"HTTP {exc.code} for {url}"
            if 500 <= int(exc.code) < 600:
                raise SGLangClientError(msg) from exc
            raise SGLangProtocolError(msg) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, socket.timeout):
                raise SGLangTimeoutError(f"Request timeout after {timeout:.2f}s for {url}") from exc
            raise SGLangClientError(f"Network error for {url}: {exc}") from exc
        except socket.timeout as exc:
            raise SGLangTimeoutError(f"Request timeout after {timeout:.2f}s for {url}") from exc
        except TimeoutError as exc:
            raise SGLangTimeoutError(f"Request timeout after {timeout:.2f}s for {url}") from exc

        if not raw:
            return {}

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SGLangProtocolError(f"Invalid JSON from {url}: {raw[:200]}") from exc

        if not isinstance(parsed, dict):
            raise SGLangProtocolError(f"Expected JSON object from {url}, got {type(parsed).__name__}")
        return parsed

    def _normalize_capabilities(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if "capabilities" in payload and isinstance(payload["capabilities"], dict):
            capabilities = dict(payload["capabilities"])
        elif "data" in payload and isinstance(payload["data"], list):
            # OpenAI-compatible /v1/models response fallback.
            capabilities = {
                "supports_trajectory": False,
                "supports_logprob": False,
                "supports_prompt_embeddings": False,
                "supports_guidance_scale": False,
                "models": [str(item.get("id")) for item in payload["data"] if isinstance(item, dict) and item.get("id")],
            }
            # Some servers may expose capability hints per model.
            if payload["data"]:
                first = payload["data"][0]
                if isinstance(first, dict) and isinstance(first.get("capabilities"), dict):
                    capabilities.update(first["capabilities"])
        else:
            capabilities = dict(payload)
        return capabilities
