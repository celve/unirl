"""Autoregressive model interfaces.

Pipeline-level: ``ARStage[C]`` — ``C → TextSegment``, iterates an ``ARStep``
token-by-token. Parameterized on the conditions container type ``C`` for
parity with ``DiffusionStage[C]`` so each bundle declares its own typed
container.

Step-level kernel: ``ARStep`` — per-token sampling kernel (tensor I/O).

The legacy ``ARTrajectory`` type is deleted — ``TextSegment`` (in
``diffusionrl/types/segments/text.py``) replaces it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Tuple, TypeVar, runtime_checkable

import torch

from diffusionrl.types.segments import TextSegment

C = TypeVar("C")


@dataclass
class ARSamplingParams:
    """Sampling configuration for an AR rollout call."""

    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    stop_token_id: int | None = None


@runtime_checkable
class ARStage(Protocol[C]):
    """Rollout-level AR stage: ``C → TextSegment``.

    Schedule-equivalent (``sampling_params``) is passed at call time, not
    held on the instance. Returned ``TextSegment`` is varlen-packed.

    The conditions type ``C`` is per-bundle: each AR bundle declares its
    own typed conditions container.

    ``replay`` recomputes per-token log-probs for a stored rollout's
    response tokens via a single teacher-forced forward over
    ``prompt + response``. Returns a packed-varlen ``[total_tokens]``
    tensor aligned with ``segment.log_probs``. Used by GRPO/PPO-style
    policy-gradient training.
    """

    def autoregress(
        self,
        conditions: C,
        *,
        sampling_params: ARSamplingParams,
        **kwargs: Any,
    ) -> TextSegment: ...

    def replay(
        self,
        conditions: C,
        *,
        segment: TextSegment,
    ) -> torch.Tensor: ...


@runtime_checkable
class ARStep(Protocol):
    """Per-step AR token sampling kernel.

    Given the model's logits over the vocabulary at the current position,
    sample the next token and return its log-probability.
    """

    def step(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: ...


__all__ = ["ARSamplingParams", "ARStage", "ARStep"]
