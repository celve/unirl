"""Leo2 conditions -- carries hymm's whole per-sample ``model_kwargs`` blob; see README.md."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from unirl.distributed.tensor.batch import Batch, concat_field
from unirl.types.conditions import TextEmbedCondition


@dataclass
class Leo2Conditions(Batch):
    """Conditions passed to the Leo2 diffusion stage."""

    # Human-readable / logging-friendly view of the text conditioning.
    text: Optional[TextEmbedCondition] = concat_field(default=None)
    # Per-sample opaque hymm blobs: {"input_ids": Tensor, "model_kwargs": dict}.
    hymm: Optional[List[Dict[str, Any]]] = concat_field(default=None)

    @classmethod
    def from_dict(cls, d: dict) -> "Leo2Conditions":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return {name: value for name in self.__dataclass_fields__ if (value := getattr(self, name)) is not None}


__all__ = ["Leo2Conditions"]
