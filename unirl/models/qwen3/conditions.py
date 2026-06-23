"""Qwen3ARConditions — typed conditions container for the Qwen3 AR stage.

Concrete instantiation of the ``ARStage[C]`` type parameter. Mirrors
:class:`unirl.models.qwen_image.QwenImageConditions` in shape:
a single typed slot (``prompt``) carrying a :class:`TextTokenCondition`
with the chat-template-built ``input_ids`` + ``attention_mask``.

The ``TextTokenCondition`` (declared in
:mod:`unirl.types.conditions.text`) is the canonical
pre-encoder-text condition for unified-vocab models — Qwen3's transformer
owns its own embedding table and consumes ``input_ids`` directly, so this
is the right wire format. The chat-template stage produces it; the AR
stage's ``autoregress`` / ``replay`` read it.

Pairs ``from_dict`` / ``to_dict`` for round-tripping between the typed
form (used inside the pipeline at stage call sites) and the generic
``Conditions = Dict[str, Condition]`` shape on ``RolloutResp``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from unirl.distributed.tensor.batch import Batch, FieldKind, field
from unirl.types.conditions import Condition, TextTokenCondition


@dataclass
class Qwen3ARConditions(Batch):
    """Typed conditions container for the Qwen3 AR stage."""

    prompt: Optional[TextTokenCondition] = field(kind=FieldKind.CONCAT, default=None)

    @classmethod
    def from_dict(cls, d: Dict[str, Condition]) -> "Qwen3ARConditions":
        """Build from the generic ``Conditions`` dict shape.

        Validates that the ``"prompt"`` slot is present and is a
        ``TextTokenCondition``.
        """
        prompt = d.get("prompt")
        if not isinstance(prompt, TextTokenCondition):
            raise TypeError(
                f"Qwen3ARConditions.from_dict: expected d['prompt'] to be a "
                f"TextTokenCondition, got "
                f"{type(prompt).__name__ if prompt is not None else 'None'}"
            )
        return cls(prompt=prompt)

    def to_dict(self) -> Dict[str, Condition]:
        """Convert back to the generic ``Conditions`` dict shape for
        packing into ``RolloutResp.tracks["ar"].conditions``.
        """
        if self.prompt is None:
            raise ValueError("Qwen3ARConditions.to_dict: prompt field is None")
        return {"prompt": self.prompt}

    @classmethod
    def from_input_segment(cls, segment, *, pad_id: int = 0) -> "Qwen3ARConditions":
        """Derive the AR prompt condition from an INPUT Part's packed token segment.

        The (C)+(I) half of the Sample/Part refactor (LIN-446 §3, §4.5): instead of
        *storing* the prompt condition at rollout time, rebuild the right-padded
        :class:`TextTokenCondition` from the input Part's per-sample prompt tokens
        (a packed ``TextSegment``). ``packed_replay`` reads only the *real* tokens
        per sample (``attention_mask.sum()`` then ``input_ids[:real]``), so the pad
        width/value are immaterial — this reproduces the exact prompt context the
        rollout conditioned on. This is the model-owned assembly rule that the
        generic ``Sample.conditions_for`` walker delegates to for the AR path.
        """
        tokens = segment.tokens
        cu = segment.cu_seqlens
        if tokens is None or cu is None:
            raise ValueError("Qwen3ARConditions.from_input_segment: segment lacks packed tokens / cu_seqlens")
        bounds = [int(c) for c in cu.tolist()]
        rows = [tokens[bounds[i] : bounds[i + 1]] for i in range(len(bounds) - 1)]
        batch = len(rows)
        max_len = max((int(r.numel()) for r in rows), default=0)
        input_ids = tokens.new_full((batch, max_len), int(pad_id))
        attention_mask = tokens.new_zeros((batch, max_len))
        for i, row in enumerate(rows):
            n = int(row.numel())
            input_ids[i, :n] = row
            attention_mask[i, :n] = 1
        return cls(prompt=TextTokenCondition(input_ids=input_ids, attention_mask=attention_mask))


__all__ = ["Qwen3ARConditions"]
