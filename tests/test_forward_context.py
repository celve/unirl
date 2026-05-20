from __future__ import annotations

import torch

from diffusionrl.types.forward_context import DefaultForwardContext


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
