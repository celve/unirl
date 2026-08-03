"""MiniMax-H3 keyframe conditioning (fl2va) -- images -> noised anchor rows.

``fl2va`` conditions generation on a first and/or last keyframe. Each keyframe
reaches the transformer TWICE, by two different routes:

* through the **video VAE**, becoming packed conditioning rows that sit between
  the text block and the target audio/video rows (this module), and
* through the **Qwen3-VL conditioner**, as a vision block interleaved into the
  text token stream whose rows are tagged VIDEO rather than TEXT (see
  ``text_embed.py``).

The rows produced here are anchors, not state: the denoising loop never writes
them, and they stay pinned at ``t = 0.999`` for every step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Sequence, Tuple

import numpy as np
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from unirl.config.require import require

from .config import MINIMAX_H3_LATENT_CHANNELS, MINIMAX_H3_PATCH_SIZE
from .packing import MiniMaxH3Geometry
from .vendor import (
    MINIMAX_H3_KEYFRAME_ENCODE_SEED,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    MiniMaxH3Scheduler,
    keyframe_condition_noise,
    patchify_video_latents,
    prepare_keyframe_image,
)

if TYPE_CHECKING:
    from .bundle import MiniMaxH3Bundle


# Anchor names, encoded as a tiny int tensor so they can ride on Conditions
# (which transports tensors, not strings). Order is the packed order.
_ANCHOR_NAMES = ("first", "last")


def resolve_keyframe_anchors(*, has_first: bool, has_last: bool) -> Tuple[str, ...]:
    """Anchor names in packed order for the keyframes actually supplied.

    A lone keyframe is ambiguous on its own -- ``fl2va`` anchors it at the first
    latent frame, ``fl2va_last_frame`` at the last -- so which one was given
    has to be carried explicitly, not inferred from the row count.
    """
    anchors: List[str] = []
    if has_first:
        anchors.append("first")
    if has_last:
        anchors.append("last")
    return tuple(anchors)


def encode_keyframe_anchors(anchors: Sequence[str]) -> torch.Tensor:
    return torch.tensor([_ANCHOR_NAMES.index(a) for a in anchors], dtype=torch.long)


def decode_keyframe_anchors(codes) -> Tuple[str, ...]:
    if codes is None:
        return ()
    return tuple(_ANCHOR_NAMES[int(c)] for c in codes.reshape(-1).tolist())


class MiniMaxH3KeyframeEncodeStage:
    """Encode keyframes into noise-augmented conditioning rows."""

    def __init__(self, bundle: "MiniMaxH3Bundle") -> None:
        self.vae = bundle.vae
        # scale_noise is pure arithmetic (x_t = t*x0 + (1-t)*noise) and holds no
        # schedule state, so a bare instance is the whole dependency. The
        # bundle deliberately does not carry a scheduler -- sigma comes from the
        # policy layer -- and borrowing one here would reintroduce that.
        self._scheduler = MiniMaxH3Scheduler()

    @torch.no_grad()
    def encode(
        self,
        images: Sequence,
        geometry: MiniMaxH3Geometry,
        *,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """``images`` (PIL, in packed order) -> ``[num_condition_rows, C*prod(patch)]``.

        Two details here are exact-reproduction contracts, not style:

        * The VAE posterior is **sampled** under a generator seeded with 42,
          independently of the request seed.
        * The sampled latent is **rounded to float16** before normalization.
          That discards ~11 bits of every conditioning latent, and the released
          model's conditioning cannot be reproduced without it.

        Keyframes are single frames, so they go through the VAE's spatial
        encoder (``_encode_clip``) alone -- none of its 17-frame temporal
        chunking applies.
        """
        require(len(images) > 0, "MiniMaxH3KeyframeEncodeStage: no keyframes to encode")
        device = self.vae.device
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1)
        pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)

        rows = []
        for image in images:
            pixels = torch.from_numpy(np.array(image)).to(device).permute(2, 0, 1)[None, :, None]
            pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
            moments = self.vae._encode_clip(pixels)
            posterior = DiagonalGaussianDistribution(moments)
            latents = posterior.sample(generator=torch.Generator().manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED))
            latents = latents.to(torch.float16).float().cpu()
            rows.append(patchify_video_latents((latents - latents_mean) / latents_std, MINIMAX_H3_PATCH_SIZE))
        condition_latents = torch.cat(rows).to(device)

        del geometry  # shape comes from the images themselves; kept for symmetry
        return self._scheduler.scale_noise(
            condition_latents, MINIMAX_H3_KEYFRAME_NOISE_AUG, noise.to(condition_latents)
        )

    @staticmethod
    def condition_noise(
        geometry: MiniMaxH3Geometry,
        num_keyframes: int,
        *,
        generator=None,
        device=None,
    ) -> torch.Tensor:
        """The noise the conditioning rows are mixed with, in packed order."""
        return keyframe_condition_noise(
            ((1, geometry.latent_height, geometry.latent_width),) * int(num_keyframes),
            MINIMAX_H3_PATCH_SIZE,
            MINIMAX_H3_LATENT_CHANNELS,
            generator=generator,
            device=device,
        )


def prepare_keyframes(images: Sequence, geometry: MiniMaxH3Geometry) -> List:
    """Fit keyframes to the canvas, in packed order.

    The FIRST keyframe of a request is the geometry anchor and is *stretched*
    onto the canvas; any following keyframe is cover-cropped to match it.
    """
    return [
        prepare_keyframe_image(image, geometry.height, geometry.width, stretch=(index == 0))
        for index, image in enumerate(images)
    ]


__all__ = [
    "MiniMaxH3KeyframeEncodeStage",
    "decode_keyframe_anchors",
    "encode_keyframe_anchors",
    "prepare_keyframes",
    "resolve_keyframe_anchors",
]
