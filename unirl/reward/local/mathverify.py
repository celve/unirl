r"""math-verify reward scorer — the paper's grader (HuggingFace Math-Verify)."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import List

from unirl.reward.base import BaseRewardComponentSpec
from unirl.types.reward import RewardRequest

from .base import LocalRewardBackend

# math-verify's own timeouts are signal.alarm-based, and signal only works on the
# main thread — under the reward's Ray worker thread they raise, which would score
# every sample 0. A worker thread with a join deadline is the portable equivalent.
_VERIFY_TIMEOUT_S = float(os.environ.get("UNIRL_MATHVERIFY_TIMEOUT_S", "10"))


def _verify_with_deadline(gold: str, prediction: str, *, seconds: float) -> bool:
    """Grade one answer, returning False if math-verify does not finish in ``seconds``."""
    from math_verify import parse, verify

    result: List[bool] = []

    def _run() -> None:
        result.append(
            bool(
                verify(
                    parse("\\boxed{" + gold + "}", parsing_timeout=None),
                    parse(prediction, parsing_timeout=None),
                    timeout_seconds=None,
                )
            )
        )

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(seconds)
    return result[0] if result else False


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
        return rewards


@dataclass
class MathVerifySpec(BaseRewardComponentSpec):
    r"""Config for the math-verify scorer."""
