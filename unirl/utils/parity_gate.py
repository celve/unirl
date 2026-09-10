"""Tolerance gate on rollout/replay log-prob parity — see README.md for the env vars."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

ABSDIFF_MAX_KEY = "rollout_replay_logp_absdiff_max"
ABSDIFF_MEAN_KEY = "rollout_replay_logp_absdiff_mean"

BEFORE_FIRST_UPDATE = "before_first_update"
AFTER_PUBLICATION = "after_publication"
STEADY_STATE = "steady_state"

# Numeric codes, because the metric sinks coerce every value to a float.
_CONTEXT_CODES = {BEFORE_FIRST_UPDATE: 1.0, AFTER_PUBLICATION: 2.0, STEADY_STATE: 0.0}


def parity_measurement_enabled() -> bool:
    """Whether to compute the gauge at all; separate from the tolerance that gates raising."""
    explicit = os.environ.get("UNIRL_PARITY_MEASURE")
    if explicit is not None:
        return explicit not in ("0", "false", "False", "")
    return os.environ.get("UNIRL_PARITY_TOLERANCE") not in (None, "")


class ParityViolation(RuntimeError):
    """Rollout and replay log-probs disagree beyond the stated tolerance."""


def micro_absdiff_max(micros: Iterable[Any]) -> Optional[float]:
    """True max over per-micro maxima; the step-level metric is their *mean* and hides an outlier."""
    values = [
        float(m.metrics[ABSDIFF_MAX_KEY])
        for m in micros
        if getattr(m, "metrics", None) and ABSDIFF_MAX_KEY in m.metrics
    ]
    return max(values) if values else None


def micro_absdiff_mean(micros: Iterable[Any]) -> Optional[float]:
    """Mean over per-micro means, matching how the gauge is reported."""
    values = [
        float(m.metrics[ABSDIFF_MEAN_KEY])
        for m in micros
        if getattr(m, "metrics", None) and ABSDIFF_MEAN_KEY in m.metrics
    ]
    return sum(values) / len(values) if values else None


@dataclass
class ParityGate:
    """Checks each step's rollout/replay agreement and names the moment it checked."""

    tolerance: Optional[float] = None
    strict: bool = True
    _pending_publication: Optional[str] = None
    _checked_first_update: bool = False

    @classmethod
    def from_env(cls) -> "ParityGate":
        raw = os.environ.get("UNIRL_PARITY_TOLERANCE")
        tolerance = float(raw) if raw not in (None, "") else None
        strict = os.environ.get("UNIRL_PARITY_STRICT", "1") not in ("0", "false", "False")
        return cls(tolerance=tolerance, strict=strict)

    @property
    def armed(self) -> bool:
        return self.tolerance is not None

    def note_publication(self, path: str) -> None:
        """Record which publication path just ran, so the next check is test 2 for that path."""
        self._pending_publication = path

    def check(self, result: Any, *, rollout_id: int) -> dict:
        """Metrics for this step's parity check; raises when an armed tolerance is breached."""
        micros = getattr(result, "micros", None) or []
        observed_max = micro_absdiff_max(micros)
        observed_mean = micro_absdiff_mean(micros)

        if self._pending_publication is not None:
            context = AFTER_PUBLICATION
            path = self._pending_publication
            self._pending_publication = None
        elif not self._checked_first_update:
            context = BEFORE_FIRST_UPDATE
            path = None
        else:
            context = STEADY_STATE
            path = None
        self._checked_first_update = True

        metrics = {
            "parity/context": _CONTEXT_CODES[context],
            "parity/tolerance": float(self.tolerance) if self.armed else math.nan,
            "parity/measured": 1.0 if observed_max is not None else 0.0,
        }
        if observed_max is None:
            # A diffusion algorithm that never wired the gauge, or an empty step.
            if context in (BEFORE_FIRST_UPDATE, AFTER_PUBLICATION) and self.armed:
                raise ParityViolation(
                    f"parity gate armed (tolerance={self.tolerance}) but rollout {rollout_id} at "
                    f"{context} reported no {ABSDIFF_MAX_KEY}: the algorithm does not emit the "
                    "rollout/replay gauge, so this run carries no correctness-test-1 evidence"
                )
            return metrics

        metrics["parity/absdiff_max"] = observed_max
        if observed_mean is not None:
            metrics["parity/absdiff_mean"] = observed_mean

        breached = self.armed and observed_max > float(self.tolerance)
        metrics["parity/violation"] = 1.0 if breached else 0.0
        if not breached:
            return metrics

        where = f"{context}" + (f" via {path}" if path else "")
        message = (
            f"rollout/replay log-prob parity failed at rollout {rollout_id} ({where}): "
            f"max|Δlogp|={observed_max:.3e} exceeds tolerance {self.tolerance:.3e}. "
            "Rollout and train-side replay disagree, so the estimator's importance ratio is "
            "not measuring what it assumes; fix correctness before reading any speed number."
        )
        if self.strict:
            raise ParityViolation(message)
        logger.warning("%s (UNIRL_PARITY_STRICT=0, continuing)", message)
        return metrics


def publication_path_name(weight_sync: Any) -> str:
    """The handler class actually used, so test 2 is attributable per publication path."""
    return type(weight_sync).__name__ if weight_sync is not None else "none"


__all__ = [
    "ABSDIFF_MAX_KEY",
    "ABSDIFF_MEAN_KEY",
    "AFTER_PUBLICATION",
    "BEFORE_FIRST_UPDATE",
    "STEADY_STATE",
    "ParityGate",
    "ParityViolation",
    "micro_absdiff_max",
    "micro_absdiff_mean",
    "parity_measurement_enabled",
    "publication_path_name",
]
