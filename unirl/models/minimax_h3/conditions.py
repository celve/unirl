"""MiniMax-H3 conditions -- typed container for the diffusion stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from unirl.distributed.tensor.batch import Batch, concat_field
from unirl.types.conditions import TextEmbedCondition


@dataclass
class MiniMaxH3Conditions(Batch):
    """Conditions passed to the MiniMax-H3 diffusion stage.

    Deliberately one slot. MiniMax-H3's checkpoint is guidance-distilled: there
    is no CFG, no unconditional pass and no negative prompt, so -- unlike every
    other video model in this repo -- there is no ``negative_text`` twin here.
    A recipe that sets ``guidance_scale`` is not doing anything; the stage
    ignores it rather than silently batching a second forward.

    Slots:
        text: Qwen3-VL hidden states from layer 50, unnormalized.
        text_token_tags: Per-text-row modality tag. Uniformly TEXT for t2va; for
            fl2va the rows of a keyframe's vision block are tagged VIDEO, so the
            tags cannot be reconstructed from the embedding length alone.
        keyframe_latent: fl2va only -- the noise-augmented keyframe conditioning
            rows, already packed. These are ANCHORS: the denoising loop never
            writes them and they stay pinned at ``t = 0.999``. They are kept out
            of ``LatentSegment`` deliberately, so the tracked trajectory and
            ``sde_logp`` cover generated rows only (the token-concat-then-slice
            idiom qwen_image_edit_plus and flux2_klein use).
        keyframe_anchor_codes: Which anchors the keyframes occupy, encoded as
            ints (0 first, 1 last) in packed order. A lone keyframe is ambiguous
            without this -- ``fl2va`` anchors at the first latent frame,
            ``fl2va_last_frame`` at the last -- and the row count cannot say
            which. Carried here so ``replay`` can rebuild the identical layout
            from conditions alone, which is all FlowGRPO hands it.
    """

    text: Optional[TextEmbedCondition] = concat_field(default=None)
    text_token_tags: Optional[torch.Tensor] = concat_field(default=None)
    keyframe_latent: Optional[torch.Tensor] = concat_field(default=None)
    keyframe_anchor_codes: Optional[torch.Tensor] = concat_field(default=None)
    reference_video_latent: Optional[torch.Tensor] = concat_field(default=None)
    reference_audio_latent: Optional[torch.Tensor] = concat_field(default=None)
    reference_geometry: Optional[torch.Tensor] = concat_field(default=None)

    @classmethod
    def from_dict(cls, d: dict) -> "MiniMaxH3Conditions":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return {name: value for name in self.__dataclass_fields__ if (value := getattr(self, name)) is not None}


__all__ = ["MiniMaxH3Conditions"]
