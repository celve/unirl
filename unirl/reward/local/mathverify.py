r"""math-verify reward scorer — the paper's grader (HuggingFace Math-Verify)."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, List, Optional

from unirl.reward.base import BaseRewardComponentSpec
from unirl.types.reward import RewardRequest

from .base import LocalRewardBackend

logger = logging.getLogger(__name__)

# math-verify's own guards are signal.alarm-based, and signal only works on the main
# thread — the reward path is not it, so enabling them raises and scores every sample
# 0. A thread cannot be killed and a runaway grade holds the GIL, so the grade runs in
# a child process that can be.
_VERIFY_TIMEOUT_S = float(os.environ.get("UNIRL_MATHVERIFY_TIMEOUT_S", "10"))

# One child, reused. Measured on an H20 worker: a child per grade costs ~1.2 s, since
# it must import this module to unpickle the target, against ~0.1 ms for a pipe
# round-trip to a child already running.
_PROC: Optional[Any] = None
_CONN: Optional[Any] = None
_LOCK = threading.Lock()


def _grade_loop(conn: Any) -> None:
    """Run in the child: grade jobs off the pipe until it closes."""
    while True:
        try:
            gold, prediction = conn.recv()
        except (EOFError, OSError):
            return
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


def _start_grader() -> Any:
    """Start a grading child on ``forkserver`` rather than ``fork``: ``fork`` runs
    ``logging``'s at-fork handler, which acquires the logging lock with no timeout, so
    forking a worker whose other threads log can block the forking thread forever."""
    import multiprocessing

    try:
        ctx = multiprocessing.get_context("forkserver")
        ctx.set_forkserver_preload(["math_verify"])
    except (ValueError, ImportError, OSError):
        ctx = multiprocessing.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=True)
    proc = ctx.Process(target=_grade_loop, args=(child_conn,), daemon=True)
    proc.start()
    child_conn.close()
    return proc, parent_conn


def _drop_grader() -> None:
    """Kill and reap the child with a bound. Nothing here may wait unbounded: the Pool
    this replaced blocked in terminate() on a queue semaphore that a child which died
    while idle still held, with the module lock held, wedging every later grade."""
    global _PROC, _CONN
    proc, conn = _PROC, _CONN
    _PROC, _CONN = None, None
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
    if proc is not None:
        try:
            proc.kill()
            proc.join(timeout=1.0)
        except Exception:
            pass


def _verify_with_deadline(gold: str, prediction: str, *, seconds: float) -> bool:
    """Grade one answer, returning False if the child does not answer in ``seconds``."""
    global _PROC, _CONN
    if not _LOCK.acquire(timeout=seconds + 30.0):
        logger.warning("math-verify: grader lock held past the deadline, scored 0.0")
        return False
    try:
        if _PROC is None or not _PROC.is_alive():
            _drop_grader()
            _PROC, _CONN = _start_grader()
        _CONN.send((gold, prediction))
        if _CONN.poll(seconds):
            return bool(_CONN.recv())
        logger.warning("math-verify: grade exceeded %.1fs, killed and scored 0.0", seconds)
        _drop_grader()
        return False
    except Exception as exc:
        logger.warning("math-verify: grader failed (%r), scored 0.0", exc)
        _drop_grader()
        return False
    finally:
        _LOCK.release()


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
            ok = _verify_with_deadline(gt, text or "", seconds=_VERIFY_TIMEOUT_S)
            rewards.append(1.0 if ok else 0.0)
        return rewards


@dataclass
class MathVerifySpec(BaseRewardComponentSpec):
    r"""Config for the math-verify scorer."""
