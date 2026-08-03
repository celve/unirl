"""MiniMax-H3 omni-reference conditioning (ref2va).

``ref2va`` conditions generation on an ORDERED list of up to 12 references --
images, videos (optionally with their soundtrack) and audio clips -- packed one
block per reference ahead of the generated rows::

    [ text | ref block 1 | ref block 2 | ... | target audio | target video ]

Order is semantic twice over: it fixes the ``"<Picture i>"`` / ``"<Audio j>"`` /
``"<Video k>"`` labels of the prompt presentation, and it advances the shared
rotary clock. UniRL carries that order by giving each reference its own ancestor
Part, since ``Sample.conditioning()`` returns the ancestor chain in chronological
order; a single Part would collapse several references into canonical modality
order and lose the interleaving.

Unlike an ``fl2va`` keyframe, a reference never binds the target geometry -- each
is prepared at its own resolution and carries its own latent shape. That is why
the per-reference geometry has to be **serialized onto Conditions**: FlowGRPO
calls ``replay(conditions, segment=, params=, step_indices=)`` and the layout has
to be rebuilt from that alone, without re-encoding the media.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import numpy as np
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from unirl.config.require import require

from .config import MINIMAX_H3_LATENT_CHANNELS, MINIMAX_H3_PATCH_SIZE
from .vendor import (
    MINIMAX_H3_KEYFRAME_ENCODE_SEED,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    MiniMaxH3Scheduler,
    keyframe_condition_noise,
    patchify_video_latents,
)
from .vendor.packing_ref2va import (
    MiniMaxH3PreparedReference,
    trim_reference_num_frames,
)

if TYPE_CHECKING:
    from .bundle import MiniMaxH3Bundle

# MiniMax-H3's own request limits.
MINIMAX_H3_MAX_REFERENCES = 12
MINIMAX_H3_MAX_AUDIO_REFERENCES = 3

# Reference kinds, encoded as ints so the geometry table is a plain tensor.
_KINDS = ("image", "video", "audio")

# Columns of the serialized geometry table.
_GEOMETRY_COLUMNS = ("kind", "num_latent_frames", "latent_height", "latent_width", "num_audio_latents")


def encode_reference_geometry(references: Sequence[MiniMaxH3PreparedReference]) -> torch.Tensor:
    """Serialize per-reference geometry to ``[num_refs, 5]`` for transport.

    ``has_audio`` is not stored: it is exactly ``num_audio_latents > 0``.
    """
    return torch.tensor(
        [
            [
                _KINDS.index(r.kind),
                int(r.num_latent_frames),
                int(r.latent_height),
                int(r.latent_width),
                int(r.num_audio_latents),
            ]
            for r in references
        ],
        dtype=torch.long,
    )


def decode_reference_geometry(table: Optional[torch.Tensor]) -> List[MiniMaxH3PreparedReference]:
    """Rebuild geometry-only ``MiniMaxH3PreparedReference`` rows for layout use.

    The media fields stay empty -- only the latent geometry participates in
    ``build_ref2va_packed_sequence``, which is precisely what makes replay able
    to reconstruct the layout without touching a VAE.
    """
    if table is None:
        return []
    rows = table.reshape(-1, len(_GEOMETRY_COLUMNS)).tolist()
    references = []
    for kind_code, num_latent_frames, latent_height, latent_width, num_audio_latents in rows:
        references.append(
            MiniMaxH3PreparedReference(
                kind=_KINDS[int(kind_code)],
                has_audio=int(num_audio_latents) > 0,
                num_latent_frames=int(num_latent_frames),
                latent_height=int(latent_height),
                latent_width=int(latent_width),
                num_audio_latents=int(num_audio_latents),
            )
        )
    return references


def validate_references(references: Sequence[MiniMaxH3PreparedReference]) -> None:
    """Enforce MiniMax-H3's request limits, loudly."""
    require(
        0 < len(references) <= MINIMAX_H3_MAX_REFERENCES,
        f"MiniMax-H3 ref2va takes 1..{MINIMAX_H3_MAX_REFERENCES} references, got {len(references)}.",
    )
    audio_only = [r for r in references if r.kind == "audio"]
    require(
        len(audio_only) <= MINIMAX_H3_MAX_AUDIO_REFERENCES,
        f"MiniMax-H3 ref2va takes at most {MINIMAX_H3_MAX_AUDIO_REFERENCES} audio references, got {len(audio_only)}.",
    )
    require(
        len(audio_only) < len(references),
        "MiniMax-H3 ref2va: audio references cannot be the only references -- at least one image or video is required.",
    )


class MiniMaxH3ReferenceEncodeStage:
    """Encode ordered references into packed conditioning rows.

    Visual and audio references are treated differently, and both are checkpoint
    contracts:

    * **Visual** (image / video) go through the video VAE with the same recipe
      fl2va keyframes use -- posterior *sampled* under a generator seeded 42
      independently of the request seed, then rounded to float16 before
      normalization -- and are afterwards noise-augmented to ``t = 0.999``.
      An image is encoded by the spatial encoder alone (``_encode_clip``); a
      video goes through the 17-frames-to-5-latents temporal chunking
      (``_encode``).
    * **Audio** takes the posterior **mean** (``mode()``), is never sampled, and
      rides along **clean** -- no noise augmentation at all.
    """

    def __init__(self, bundle: "MiniMaxH3Bundle") -> None:
        self.vae = bundle.vae
        self.audio_vae = bundle.audio_vae
        self._scheduler = MiniMaxH3Scheduler()

    @torch.no_grad()
    def encode(
        self,
        references: List[MiniMaxH3PreparedReference],
        *,
        condition_noise: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """``references`` (in packed order) -> ``(video_rows, audio_rows)``.

        Fills each reference's latent geometry in place -- encoding is what
        resolves it, so callers must serialize the geometry *after* this runs.
        """
        device = self.vae.device
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1)
        pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
        audio_mean = torch.tensor(self.audio_vae.config.latents_mean).view(1, 1, -1)
        audio_std = torch.tensor(self.audio_vae.config.latents_std).view(1, 1, -1)
        audio_latent_channels = int(self.audio_vae.config.latent_channels)

        video_rows: List[torch.Tensor] = []
        audio_rows: List[torch.Tensor] = []
        for reference in references:
            if reference.kind != "audio":
                if reference.kind == "image":
                    pixels = torch.from_numpy(np.array(reference.image)).to(device).permute(2, 0, 1)[None, :, None]
                else:
                    frames = reference.frames[: trim_reference_num_frames(reference.frames.shape[0])]
                    pixels = torch.from_numpy(frames.copy()).to(device).permute(3, 0, 1, 2)[None]
                pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
                moments = self.vae._encode_clip(pixels) if reference.kind == "image" else self.vae._encode(pixels)
                latents = DiagonalGaussianDistribution(moments).sample(
                    generator=torch.Generator().manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED)
                )
                latents = latents.to(torch.float16).float().cpu()
                reference.num_latent_frames = latents.shape[2]
                reference.latent_height, reference.latent_width = latents.shape[3], latents.shape[4]
                video_rows.append(patchify_video_latents((latents - latents_mean) / latents_std, MINIMAX_H3_PATCH_SIZE))

            if reference.has_audio:
                posterior = self.audio_vae.encode(reference.waveform.to(device)[:, None], return_dict=False)[0]
                # Posterior MEAN, not a sample -- reference soundtracks are never
                # stochastic. Channel-major: the two stereo channels are two
                # batch items of the mono audio VAE.
                latents = posterior.mode().float().cpu().transpose(1, 2)
                reference.num_audio_latents = latents.shape[1]
                audio_rows.append(((latents - audio_mean) / audio_std).reshape(-1, audio_latent_channels))

        video = torch.cat(video_rows).to(device) if video_rows else None
        audio = torch.cat(audio_rows).to(device) if audio_rows else None

        if video is not None:
            require(
                condition_noise is not None,
                "MiniMaxH3ReferenceEncodeStage: visual references need condition_noise for the t=0.999 augmentation.",
            )
            video = self._scheduler.scale_noise(video, MINIMAX_H3_KEYFRAME_NOISE_AUG, condition_noise.to(video))
        # Audio conditioning rows are NOT noised: they ride clean at t = 1.0.
        return video, audio

    @staticmethod
    def condition_noise_shapes(
        references: Sequence[MiniMaxH3PreparedReference],
    ) -> Tuple[Tuple[int, int, int], ...]:
        """``(num_latent_frames, latent_h, latent_w)`` per VISUAL reference."""
        return tuple((r.num_latent_frames, r.latent_height, r.latent_width) for r in references if r.kind != "audio")

    @staticmethod
    def condition_noise(
        shapes: Sequence[Tuple[int, int, int]],
        *,
        generator=None,
        device=None,
    ) -> Optional[torch.Tensor]:
        if not shapes:
            return None
        return keyframe_condition_noise(
            tuple(shapes),
            MINIMAX_H3_PATCH_SIZE,
            MINIMAX_H3_LATENT_CHANNELS,
            generator=generator,
            device=device,
        )


__all__ = [
    "MINIMAX_H3_MAX_AUDIO_REFERENCES",
    "MINIMAX_H3_MAX_REFERENCES",
    "MiniMaxH3ReferenceEncodeStage",
    "decode_reference_geometry",
    "encode_reference_geometry",
    "validate_references",
]
