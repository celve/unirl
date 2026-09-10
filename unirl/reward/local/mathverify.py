r"""math-verify reward scorer — the paper's grader (HuggingFace Math-Verify)."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from unirl.reward.base import BaseRewardComponentSpec
from unirl.types.reward import RewardRequest

from .base import LocalRewardBackend

logger = logging.getLogger(__name__)

# math-verify's own timeouts are signal.alarm-based, and signal only works on the
# main thread — under the reward's Ray worker thread they raise, which would score
# every sample 0. A thread cannot be killed and a runaway grade holds the GIL, so the
# grade runs in a child PROCESS that can actually be terminated on expiry.
_VERIFY_TIMEOUT_S = float(os.environ.get("UNIRL_MATHVERIFY_TIMEOUT_S", "10"))

# One long-lived single-worker pool per process: a pool-per-call would fork 512 times
# a rollout. Replaced wholesale when a timeout kills its worker.
_POOL: Optional[Any] = None
_POOL_LOCK = threading.Lock()

# A 0.0 has three causes — wrong answer, expiry, grader error — and a reward curve
# cannot distinguish them. Counted so a flat curve is attributable after the fact.
VERIFY_COUNTERS: Dict[str, int] = {"graded": 0, "expired": 0, "failed": 0}


def _grade(gold: str, prediction: str) -> bool:
    """Run in the child: the raw math-verify call, with its own timeouts left off."""
    from math_verify import parse, verify

    return bool(
        verify(
            parse("\\boxed{" + gold + "}", parsing_timeout=None),
            parse(prediction, parsing_timeout=None),
            timeout_seconds=None,
        )
    )


def _reset_pool() -> None:
    """Drop the pool, terminating a worker still spinning on a runaway expression."""
    global _POOL
    pool, _POOL = _POOL, None
    if pool is None:
        return
    pool.terminate()
    threading.Thread(target=pool.join, daemon=True).start()


def _verify_with_deadline(gold: str, prediction: str, *, seconds: float) -> bool:
    """Grade one answer, returning False if math-verify does not finish in ``seconds``."""
    import multiprocessing

    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = multiprocessing.get_context("fork").Pool(processes=1)
        try:
            out = bool(_POOL.apply_async(_grade, (gold, prediction)).get(timeout=seconds))
            VERIFY_COUNTERS["graded"] += 1
            return out
        except multiprocessing.TimeoutError:
            VERIFY_COUNTERS["expired"] += 1
            _reset_pool()
            return False
        except Exception:
            VERIFY_COUNTERS["failed"] += 1
            _reset_pool()
            return False

class MathVerifyRewardScorer(LocalRewardBackend):
    r"""Numeric/symbolic reward via HuggingFace ``math-verify`` (1.0 match / 0.0)."""

    canonical_model_name = "math_verify"
    input_kind = "text"

    def __init__(self, *, config: "MathVerifySpec", base_device: str) -> None:
        del base_device
        super().__init__()

    def _load_model(self) -> None:
        self.model = "math_verify"

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        generated = request.texts
        if generated is None:
            raise ValueError("MathVerifyRewardScorer requires request.texts (generated answers).")
        metadata_list = request.metadata or [None] * len(generated)
        rewards: List[float] = []
        for text, meta in zip(generated, metadata_list):
            if meta is None or "answer" not in meta:
                rewards.append(0.0)
                continue
            gt = str(meta["answer"]).strip()
            try:
                ok = _verify_with_deadline(gt, text or "", seconds=_VERIFY_TIMEOUT_S)
            except Exception:
                ok = False
            rewards.append(1.0 if ok else 0.0)
        if VERIFY_COUNTERS["expired"] or VERIFY_COUNTERS["failed"]:
            logger.warning(
                "math-verify counters: graded=%d expired=%d failed=%d — a 0.0 reward from an "
                "expiry or a grader error is indistinguishable from a wrong answer in the curve",
                VERIFY_COUNTERS["graded"],
                VERIFY_COUNTERS["expired"],
                VERIFY_COUNTERS["failed"],
            )
        return rewards


@dataclass
class MathVerifySpec(BaseRewardComponentSpec):
    r"""Config for the math-verify scorer."""
