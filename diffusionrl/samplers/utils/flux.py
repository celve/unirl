"""FLUX positional encoding helpers.

SGLang's ``OutputBatch`` does not carry FLUX's ``text_ids`` / ``image_ids``
positional encodings — the consumer must synthesize them client-side from
``prompt_embeds.shape`` and the target ``(height, width)``. These helpers
mirror what FLUX's diffusers reference pipeline does at the same boundary.
"""

from __future__ import annotations

import torch

from diffusionrl.config.require import require


def build_flux_text_ids(prompt_embeds: torch.Tensor) -> torch.Tensor:
    require(prompt_embeds.dim() >= 3, f"FLUX prompt_embeds must be [B,seq,hidden], got {tuple(prompt_embeds.shape)}")
    batch_size = int(prompt_embeds.shape[0])
    seq_len = int(prompt_embeds.shape[1])
    return torch.zeros(
        batch_size,
        seq_len,
        3,
        device=prompt_embeds.device,
        dtype=prompt_embeds.dtype,
    )


def build_flux_image_ids(
    *,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    latent_h = max(1, int(height) // 8)
    latent_w = max(1, int(width) // 8)
    packed_h = max(1, latent_h // 2)
    packed_w = max(1, latent_w // 2)

    image_ids = torch.zeros(packed_h, packed_w, 3, device=device, dtype=dtype)
    image_ids[..., 1] = torch.arange(packed_h, device=device, dtype=dtype)[:, None]
    image_ids[..., 2] = torch.arange(packed_w, device=device, dtype=dtype)[None, :]
    return image_ids.reshape(packed_h * packed_w, 3)
