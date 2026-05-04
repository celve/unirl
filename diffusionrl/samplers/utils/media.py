"""Decoded image/video sample normalization to channels-first canonical layout."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from diffusionrl.config.require import require
from diffusionrl.samplers.utils.tensorize import tensorize

if TYPE_CHECKING:
    import numpy as np
    from PIL.Image import Image as PILImage


def normalize_media(sample: torch.Tensor) -> torch.Tensor:
    """Permute a decoded sample to channels-first canonical layout.

    Recognized inputs: ``[C,H,W]``, ``[H,W,C]``, ``[C,T,H,W]``,
    ``[T,C,H,W]``, ``[T,H,W,C]``. Returns ``[C,H,W]`` for 3D image,
    ``[C,T,H,W]`` for 4D video. Raises ``ValueError`` for unrecognized
    layouts.
    """
    if sample.dim() == 3:
        if sample.shape[0] in (1, 3, 4):
            return sample
        require(sample.shape[-1] in (1, 3, 4), f"Unrecognized 3D media layout: {tuple(sample.shape)}")
        return sample.permute(2, 0, 1)

    require(sample.dim() == 4, f"Unrecognized media tensor dim {sample.dim()}: shape={tuple(sample.shape)}")

    if sample.shape[0] in (1, 3, 4):
        return sample
    if sample.shape[1] in (1, 3, 4):
        return sample.permute(1, 0, 2, 3)
    require(sample.shape[-1] in (1, 3, 4), f"Unrecognized 4D media layout: {tuple(sample.shape)}")
    return sample.permute(3, 0, 1, 2)


def decode_sample(
    sample: torch.Tensor | np.ndarray | PILImage | tuple | list | None,
) -> Optional[torch.Tensor]:
    """Read a SGLang ``result.samples`` payload into a canonical media tensor.

    Handles the ``(video, audio)`` 2-tuple wrap from SGLang's
    ``attach_audio_to_video_sample`` (audio is dropped — ``result.audio`` is
    the canonical channel for that). Returns ``None`` when no recognizable
    sample is present.
    """
    if isinstance(sample, (tuple, list)) and len(sample) == 2:
        sample = sample[0]
    sample_tensor = tensorize(sample)
    if sample_tensor is None:
        return None
    canonical = normalize_media(sample_tensor.detach().cpu())
    # decoded_images contract: [C,H,W] floats in [0, 1] (matches FSDP path).
    # VAEs routinely overshoot [0, 1] by a few percent — clamp here so
    # consumers don't have to guess the range.
    if canonical.is_floating_point():
        canonical = canonical.clamp(0.0, 1.0)
    return canonical
