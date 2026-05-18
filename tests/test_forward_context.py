from __future__ import annotations

import torch

from diffusionrl.types.forward_context import (
    DefaultForwardContext,
    HunyuanVeido1p5ForwardContext,
)


def test_default_forward_context_image_ids_are_shared() -> None:
    image_ids = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    ctx = DefaultForwardContext(
        prompt_embeds=torch.zeros(3, 1, 1),
        image_ids=image_ids,
    )

    selected = ctx.select(torch.tensor([2, 0]))
    assert torch.equal(selected.prompt_embeds, torch.zeros(2, 1, 1))
    assert selected.image_ids is image_ids

    sliced = ctx.slice(1, 3)
    assert torch.equal(sliced.prompt_embeds, torch.zeros(2, 1, 1))
    assert sliced.image_ids is image_ids

    other_image_ids = torch.full((3, 2), 99.0)
    concatenated = DefaultForwardContext.concat(
        [
            ctx,
            DefaultForwardContext(
                prompt_embeds=torch.ones(2, 1, 1),
                image_ids=other_image_ids,
            ),
        ]
    )
    assert torch.equal(concatenated.prompt_embeds, torch.cat([torch.zeros(3, 1, 1), torch.ones(2, 1, 1)]))
    assert concatenated.image_ids is image_ids


def test_hunyuan_veido1p5_forward_context_concat_and_select() -> None:
    """Per-sample tensors concat along dim 0; ``attention_kwargs`` is shared."""
    ctx_a = HunyuanVeido1p5ForwardContext(
        guidance_scale=6.0,
        prompt_embeds=torch.zeros(3, 5, 8),
        prompt_embeds_mask=torch.ones(3, 5, dtype=torch.long),
        prompt_embeds_2=torch.zeros(3, 6, 4),
        prompt_embeds_mask_2=torch.zeros(3, 6, dtype=torch.long),
        cond_latents=torch.zeros(3, 32, 2, 4, 8),
        cond_mask=torch.zeros(3, 1, 2, 4, 8),
        attention_kwargs={"scale": 1.0},
    )
    ctx_b = HunyuanVeido1p5ForwardContext(
        guidance_scale=6.0,
        prompt_embeds=torch.ones(2, 5, 8),
        prompt_embeds_mask=torch.ones(2, 5, dtype=torch.long),
        prompt_embeds_2=torch.ones(2, 6, 4),
        prompt_embeds_mask_2=torch.ones(2, 6, dtype=torch.long),
        cond_latents=torch.ones(2, 32, 2, 4, 8),
        cond_mask=torch.ones(2, 1, 2, 4, 8),
        attention_kwargs={"scale": 999.0},  # shared field — first wins
    )
    merged = HunyuanVeido1p5ForwardContext.concat([ctx_a, ctx_b])

    assert merged.prompt_embeds.shape == (5, 5, 8)
    assert merged.cond_latents.shape == (5, 32, 2, 4, 8)
    assert merged.cond_mask.shape == (5, 1, 2, 4, 8)
    assert merged.attention_kwargs == {"scale": 1.0}

    selected = merged.select(torch.tensor([4, 0]))
    assert torch.equal(selected.prompt_embeds[0], torch.ones(5, 8))
    assert torch.equal(selected.prompt_embeds[1], torch.zeros(5, 8))


def test_hunyuan_veido1p5_forward_context_cast_dtype_only_floats() -> None:
    ctx = HunyuanVeido1p5ForwardContext(
        prompt_embeds=torch.zeros(2, 5, 8, dtype=torch.float32),
        prompt_embeds_mask=torch.ones(2, 5, dtype=torch.long),
        prompt_embeds_2=torch.zeros(2, 6, 4, dtype=torch.float32),
        prompt_embeds_mask_2=torch.zeros(2, 6, dtype=torch.long),
    )
    casted = ctx.cast_dtype(torch.float16)
    assert casted.prompt_embeds.dtype == torch.float16
    assert casted.prompt_embeds_2.dtype == torch.float16
    # Integer masks must NOT be cast to a float dtype.
    assert casted.prompt_embeds_mask.dtype == torch.long
    assert casted.prompt_embeds_mask_2.dtype == torch.long
