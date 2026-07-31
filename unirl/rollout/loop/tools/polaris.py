"""Shared reliability primitives for the Polaris web-tool gateway.

Credentials are resolved at request time so pod-local rotation takes effect
without rebuilding tool objects.  This module also owns the process-wide
Polaris concurrency limiter, transport failure classification/backoff, safe
diagnostic aggregation, and the small success-only caches used by the web
tools.  No helper stores or reports credentials, request inputs, endpoints, or
response bodies.
"""

from __future__ import annotations

import email.utils
import os
import random
import re
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Hashable, Mapping, Optional, Tuple

import requests

_PROVIDER_PREFIX = {"serper": "SERPER", "jina_ai": "JINA"}
POLARIS_GATEWAY_URL = "http://trpc-gpt-eval.production.polaris:8080"
POLARIS_SERPER_URL = f"{POLARIS_GATEWAY_URL}/search"
POLARIS_JINA_URL = f"{POLARIS_GATEWAY_URL}/"

_TRANSIENT_STATUSES = {408, 425, 429}
_AUTH_STATUSES = {401, 403}
_JITTER_RANDOM = random.SystemRandom()
_SAFE_STATUS_RE = re.compile(
    r"^(?:(?:http|app|app_status)_[1-5][0-9]{2}|"
    r"app(?:_status)?_malformed|timeout|connection|auth_config|client_error|"
    r"http_error|http_unknown|malformed_response|malformed_jina_envelope|"
    r"empty_jina_content|other)$"
)


class PolarisAuthError(RuntimeError):
    """Credential/configuration failure whose message contains no secret."""


class ProviderRequestError(RuntimeError):
    """Classified provider failure used by the retry loops.

    ``category`` is one of ``transient``, ``permanent``, or ``auth``.
    ``status_key`` is already reduced to a low-cardinality, safe diagnostic
    label (for example ``http_429`` or ``timeout``).
    """

    def __init__(
        self,
        category: str,
        status_key: str,
        *,
        retry_after_s: Optional[float] = None,
    ) -> None:
        super().__init__("provider request failed")
        self.category = category
        self.status_key = _safe_status_key(status_key)
        self.retry_after_s = retry_after_s


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
        raise PolarisAuthError(f"{prefix} Polaris authentication requires both app id and app key")
    if not app_id:
        return None
    try:
        timeout = int(os.environ.get("POLARIS_PROVIDER_TIMEOUT", "60"))
    except ValueError as exc:
        raise PolarisAuthError("POLARIS_PROVIDER_TIMEOUT must be a positive integer") from exc
    if timeout <= 0:
        raise PolarisAuthError("POLARIS_PROVIDER_TIMEOUT must be a positive integer")
    token = f"{app_id}:{app_key}?provider={provider}&timeout={timeout}"
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


# One limiter is shared by SearchTool and VisitTool in a Python process.  The
# first configured capacity wins; production constructs both with the same
# default.  Keeping the registry keyed by gateway leaves room for a second
# trusted gateway without accidentally coupling unrelated traffic.
_LIMITER_LOCK = threading.Lock()
_LIMITERS: Dict[str, threading.BoundedSemaphore] = {}
_LIMITER_CAPACITIES: Dict[str, int] = {}


def get_gateway_semaphore(capacity: int = 2, gateway: str = POLARIS_GATEWAY_URL) -> threading.BoundedSemaphore:
    """Return the shared per-process semaphore for ``gateway``."""

    capacity = int(capacity)
    if capacity <= 0:
        raise ValueError("gateway_max_in_flight must be positive")
    with _LIMITER_LOCK:
        limiter = _LIMITERS.get(gateway)
        if limiter is None:
            limiter = threading.BoundedSemaphore(capacity)
            _LIMITERS[gateway] = limiter
            _LIMITER_CAPACITIES[gateway] = capacity
        elif _LIMITER_CAPACITIES[gateway] != capacity:
            raise ValueError("all Polaris tools in one process must use the same gateway_max_in_flight")
        return limiter


def _safe_status_key(value: str) -> str:
    value = str(value)
    return value if _SAFE_STATUS_RE.fullmatch(value) else "other"


def _status_family(value: Any) -> Optional[int]:
    """Normalize HTTP-like provider codes such as ``42206`` to ``422``."""

    try:
        status = int(value)
    except (TypeError, ValueError):
        return None
    status = abs(status)
    if status > 999:
        status = int(str(status)[:3])
    return status


def status_category(status: int) -> str:
    """Classify an HTTP/application status into retry, auth, or permanent."""

    if status in _TRANSIENT_STATUSES or status >= 500:
        return "transient"
    if status in _AUTH_STATUSES:
        return "auth"
    # 422 is explicitly permanent; all other non-success 4xx statuses are also
    # permanent rather than useful retry candidates.
    return "permanent"


def application_error(
    value: Any,
    *,
    prefix: str = "app",
    retry_after_s: Optional[float] = None,
) -> Optional[ProviderRequestError]:
    """Return a classified error for a non-200 provider application code."""

    status = _status_family(value)
    if status == 200:
        return None
    if status is None:
        return ProviderRequestError("transient", f"{prefix}_malformed")
    return ProviderRequestError(
        status_category(status),
        f"{prefix}_{status}",
        retry_after_s=retry_after_s,
    )


def optional_application_status_error(
    value: Any,
    *,
    prefix: str = "app_status",
    retry_after_s: Optional[float] = None,
) -> Optional[ProviderRequestError]:
    """Classify an optional status field only when it is error-shaped.

    Polaris success envelopes observed in production use ``status=20000``;
    other compatible gateways commonly use ``status=0`` or textual values such
    as ``"success"``.  The required ``code`` field remains authoritative.  An
    optional status therefore augments failures only when it clearly encodes a
    4xx/5xx family, rather than turning benign provider metadata into an outage.
    """

    status = _status_family(value)
    if status is None or status == 0 or status < 400:
        return None
    return ProviderRequestError(
        status_category(status),
        f"{prefix}_{status}",
        retry_after_s=retry_after_s,
    )


def _parse_retry_after(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass
    try:
        parsed = email.utils.parsedate_to_datetime(str(value))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def retry_after_from_response(response: Any) -> Optional[float]:
    headers = getattr(response, "headers", None)
    value = headers.get("Retry-After") if isinstance(headers, Mapping) else None
    return _parse_retry_after(value)


def response_error(response: Any) -> Optional[ProviderRequestError]:
    """Classify a non-2xx HTTP response without retaining its body or headers."""

    status = int(getattr(response, "status_code", 0) or 0)
    if 200 <= status < 300:
        return None
    return ProviderRequestError(
        status_category(status),
        f"http_{status}" if status else "http_unknown",
        retry_after_s=retry_after_from_response(response),
    )


def classify_request_exception(exc: BaseException) -> ProviderRequestError:
    """Convert arbitrary request/config failures to a credential-safe class."""

    if isinstance(exc, ProviderRequestError):
        return exc
    if isinstance(exc, PolarisAuthError):
        return ProviderRequestError("auth", "auth_config")
    if isinstance(exc, requests.HTTPError):
        response = exc.response
        if response is not None:
            classified = response_error(response)
            if classified is not None:
                return classified
        return ProviderRequestError("permanent", "http_error")
    if isinstance(exc, requests.Timeout):
        return ProviderRequestError("transient", "timeout")
    if isinstance(exc, requests.ConnectionError):
        return ProviderRequestError("transient", "connection")
    # Preserve fail-fast behavior for unknown programming/client exceptions.
    # Every provider/config/protocol failure handled by the web tools is wrapped
    # explicitly above, so retrying an arbitrary RuntimeError would only repeat a
    # local code bug and can hide it behind four identical attempts.
    return ProviderRequestError("permanent", "client_error")


def is_transient_request_error(exc: BaseException) -> bool:
    """Backward-compatible predicate used by older callers/tests."""

    return classify_request_exception(exc).category == "transient"


def full_jitter_delay(
    retry_index: int,
    *,
    retry_after_s: Optional[float] = None,
    base_s: float = 0.5,
    cap_s: float = 8.0,
    retry_after_cap_s: float = 30.0,
) -> float:
    """Exponential full jitter, respecting a bounded ``Retry-After`` floor."""

    base_s = max(0.0, float(base_s))
    cap_s = max(0.0, float(cap_s))
    retry_after_cap_s = max(0.0, float(retry_after_cap_s))
    window = min(cap_s, base_s * (2 ** max(0, int(retry_index))))
    # Use an isolated system RNG so concurrent retry timing cannot perturb the
    # Python PRNG state used by dataset/model sampling elsewhere in the worker.
    delay = _JITTER_RANDOM.uniform(0.0, window) if window else 0.0
    if retry_after_s is not None:
        delay = max(delay, min(max(0.0, float(retry_after_s)), retry_after_cap_s))
    return delay


@dataclass
class RequestDiagnostics:
    """Mutable accumulator that serializes only canonical safe fields."""

    tool: str
    provider: str
    request_count: int = 0
    success_count: int = 0
    cache_hit_count: int = 0
    retry_count: int = 0
    recovered_transient_count: int = 0
    transient_exhausted_count: int = 0
    permanent_error_count: int = 0
    auth_error_count: int = 0
    status_counts: Counter = field(default_factory=Counter)

    def record_status(self, status_key: str) -> None:
        self.status_counts[_safe_status_key(status_key)] += 1

    def merge(self, other: "RequestDiagnostics") -> None:
        for field_name in (
            "request_count",
            "success_count",
            "cache_hit_count",
            "retry_count",
            "recovered_transient_count",
            "transient_exhausted_count",
            "permanent_error_count",
            "auth_error_count",
        ):
            setattr(self, field_name, getattr(self, field_name) + getattr(other, field_name))
        self.status_counts.update(other.status_counts)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "provider": self.provider,
            "request_count": int(self.request_count),
            "success_count": int(self.success_count),
            "cache_hit_count": int(self.cache_hit_count),
            "retry_count": int(self.retry_count),
            "recovered_transient_count": int(self.recovered_transient_count),
            "transient_exhausted_count": int(self.transient_exhausted_count),
            "permanent_error_count": int(self.permanent_error_count),
            "auth_error_count": int(self.auth_error_count),
            "status_counts": {key: int(value) for key, value in sorted(self.status_counts.items())},
        }


@dataclass
class _CacheEntry:
    expires_at: float
    value: Any


@dataclass
class _Flight:
    event: threading.Event = field(default_factory=threading.Event)
    value: Any = None
    error: Optional[BaseException] = None
    cacheable: bool = False


class SuccessTTLCache:
    """Thread-safe bounded TTL cache with per-key single-flight loading.

    ``loader`` returns ``(value, cacheable)``.  Failed outcomes can therefore be
    shared with callers already waiting on the same flight, but are never
    inserted into the TTL cache; the next independent call retries the provider.
    The source string is ``load``, ``cache``, or ``singleflight``.
    """

    def __init__(self, ttl_s: float, max_entries: int) -> None:
        self._ttl_s = max(0.0, float(ttl_s))
        self._max_entries = int(max_entries)
        if self._max_entries <= 0:
            raise ValueError("cache_max_entries must be positive")
        self._lock = threading.Lock()
        self._entries: "OrderedDict[Hashable, _CacheEntry]" = OrderedDict()
        self._inflight: Dict[Hashable, _Flight] = {}

    def get_or_compute(
        self,
        key: Hashable,
        loader: Callable[[], Tuple[Any, bool]],
    ) -> Tuple[Any, str]:
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                if entry.expires_at > now:
                    self._entries.move_to_end(key)
                    return entry.value, "cache"
                del self._entries[key]
            flight = self._inflight.get(key)
            if flight is None:
                flight = _Flight()
                self._inflight[key] = flight
                leader = True
            else:
                leader = False

        if not leader:
            flight.event.wait()
            if flight.error is not None:
                raise flight.error
            return flight.value, "singleflight"

        try:
            value, cacheable = loader()
            flight.value = value
            flight.cacheable = bool(cacheable)
            with self._lock:
                if flight.cacheable and self._ttl_s > 0:
                    now = time.monotonic()
                    expired = [k for k, v in self._entries.items() if v.expires_at <= now]
                    for expired_key in expired:
                        del self._entries[expired_key]
                    self._entries[key] = _CacheEntry(now + self._ttl_s, value)
                    self._entries.move_to_end(key)
                    while len(self._entries) > self._max_entries:
                        self._entries.popitem(last=False)
        except BaseException as exc:
            flight.error = exc
            raise
        finally:
            with self._lock:
                self._inflight.pop(key, None)
                flight.event.set()
        return flight.value, "load"


__all__ = [
    "POLARIS_GATEWAY_URL",
    "POLARIS_JINA_URL",
    "POLARIS_SERPER_URL",
    "PolarisAuthError",
    "ProviderRequestError",
    "RequestDiagnostics",
    "SuccessTTLCache",
    "application_error",
    "classify_request_exception",
    "full_jitter_delay",
    "get_gateway_semaphore",
    "is_transient_request_error",
    "optional_application_status_error",
    "polaris_headers",
    "retry_after_from_response",
    "response_error",
]
