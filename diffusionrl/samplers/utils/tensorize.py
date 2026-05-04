"""Best-effort coercion of arbitrary values into ``torch.Tensor``.

Handles tensors / numpy arrays / PIL images and homogeneous lists/tuples of
the same. Returns ``None`` when no sensible conversion exists, instead of
raising — callers decide whether absence is an error.
"""

from __future__ import annotations

from typing import Any, Optional

import torch


def tensorize(value: Any) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value
    try:
        import numpy as np
        from PIL import Image

        if isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        if isinstance(value, Image.Image):
            return torch.from_numpy(np.array(value))
        if isinstance(value, (list, tuple)) and value:
            if all(torch.is_tensor(v) for v in value):
                return torch.stack([v.detach() for v in value], dim=0)
            if all(isinstance(v, np.ndarray) for v in value):
                return torch.from_numpy(np.stack(value, axis=0))
            if all(isinstance(v, Image.Image) for v in value):
                return torch.from_numpy(np.stack([np.array(v) for v in value], axis=0))
    except Exception:
        pass
    return None
