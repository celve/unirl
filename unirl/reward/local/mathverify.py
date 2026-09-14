r"""math-verify reward scorer — the paper's grader (HuggingFace Math-Verify)."""

from __future__ import annotations

import atexit
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

# One long-lived grading child per process: a child-per-call would fork 512 times a
# rollout. Replaced whenever it dies or overruns.
_PROC: Optional[Any] = None
_CONN: Optional[Any] = None
_LOCK = threading.Lock()

# Grace above the grade deadline before a caller gives up on the lock itself, so one
# stuck thread cannot silently wedge every later grade in the process.
_LOCK_GRACE_S = float(os.environ.get("UNIRL_MATHVERIFY_LOCK_GRACE_S", "30"))

# A 0.0 has three causes — wrong answer, expiry, grader error — and a reward curve
# cannot distinguish them. Counted so a flat curve is attributable after the fact.
VERIFY_COUNTERS: Dict[str, int] = {"graded": 0, "expired": 0, "failed": 0}
_LAST_REPORTED: Dict[str, int] = {"graded": 0, "expired": 0, "failed": 0}
_LAST_REPORTED_BAD = 0

def _grade_loop(conn: Any) -> None:
    """Run in the child: grade jobs off the pipe until it closes."""
    while True:
        try:
            job = conn.recv()
        except (EOFError, OSError):
            return
        if job is None:
            return
        gold, prediction = job
        try:
            from math_verify import parse, verify

            ok = bool(
                verify(
                    parse("\\boxed{" + gold + "}", parsing_timeout=None),
                    parse(prediction, parsing_timeout=None),
                    timeout_seconds=None,
                )
            )
        except Exception:
            ok = False
        try:
            conn.send(ok)
        except (BrokenPipeError, OSError):
            return


def _grader_context() -> Any:
    """A start method that does not fork this process.

    ``fork`` runs ``logging``'s registered at-fork handler, which acquires the logging
    module lock with no timeout — so forking a Ray worker whose other threads log can
    block the forking thread forever, while this module's lock is held. ``forkserver``
    execs a small server once and forks children from that instead.
    """
    import multiprocessing

    try:
        ctx = multiprocessing.get_context("forkserver")
        ctx.set_forkserver_preload(["math_verify"])
        return ctx
    except (ValueError, ImportError, OSError):
        logger.warning("math-verify: forkserver unavailable, falling back to fork")
        return multiprocessing.get_context("fork")


def _spawn_grader() -> Any:
    """Start a grading child and keep only the parent's end of its pipe."""
    ctx = _grader_context()
    parent_conn, child_conn = ctx.Pipe(duplex=True)
    proc = ctx.Process(target=_grade_loop, args=(child_conn,), daemon=True)
    proc.start()
    child_conn.close()
    return proc, parent_conn


def _discard_grader() -> None:
    """Drop the child with no unbounded wait — the Pool this replaced parked forever in
    terminate(), on a queue semaphore a child that died while idle still held."""
    global _PROC, _CONN
    proc, conn = _PROC, _CONN
    _PROC, _CONN = None, None
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
    if proc is None:
        return
    try:
        if proc.is_alive():
            proc.kill()
        proc.join(timeout=1.0)
    except Exception:
        pass


def _verify_with_deadline(gold: str, prediction: str, *, seconds: float) -> bool:
    """Grade one answer, returning False if math-verify does not finish in ``seconds``."""
    global _PROC, _CONN
    if not _LOCK.acquire(timeout=seconds + _LOCK_GRACE_S):
        VERIFY_COUNTERS["failed"] += 1
        return False
    try:
        if _PROC is None or not _PROC.is_alive():
            _discard_grader()
            _PROC, _CONN = _spawn_grader()
        _CONN.send((gold, prediction))
        if not _CONN.poll(seconds):
            VERIFY_COUNTERS["expired"] += 1
            _discard_grader()
            return False
        out = bool(_CONN.recv())
        VERIFY_COUNTERS["graded"] += 1
        return out
    except Exception:
        VERIFY_COUNTERS["failed"] += 1
        _discard_grader()
        return False
    finally:
        _LOCK.release()


def _atexit_discard() -> None:
    """Never block interpreter shutdown on a grade still in flight."""
    if _LOCK.acquire(blocking=False):
        try:
            _discard_grader()
        finally:
            _LOCK.release()


atexit.register(_atexit_discard)

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
        # Only on change: the counters are cumulative, so an unconditional check would
        # repeat one line every batch for the rest of the run and bury the event.
        global _LAST_REPORTED_BAD
        bad = VERIFY_COUNTERS["expired"] + VERIFY_COUNTERS["failed"]
        if bad > _LAST_REPORTED_BAD:
            d_exp = VERIFY_COUNTERS["expired"] - _LAST_REPORTED["expired"]
            d_fail = VERIFY_COUNTERS["failed"] - _LAST_REPORTED["failed"]
            _LAST_REPORTED.update(VERIFY_COUNTERS)
            _LAST_REPORTED_BAD = bad
            logger.warning(
                "math-verify: +%d expired +%d failed this batch (cumulative graded=%d "
                "expired=%d failed=%d) - a 0.0 from an expiry or grader error is "
                "indistinguishable from a wrong answer in the reward curve",
                d_exp,
                d_fail,
                VERIFY_COUNTERS["graded"],
                VERIFY_COUNTERS["expired"],
                VERIFY_COUNTERS["failed"],
            )
        return rewards


@dataclass
class MathVerifySpec(BaseRewardComponentSpec):
    r"""Config for the math-verify scorer."""
