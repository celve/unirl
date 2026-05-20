"""Tests for Condition types and Conditions composition."""

from __future__ import annotations

import torch

from diffusionrl.types.conditions import (
    Condition,
    Conditions,
    ImageEmbedCondition,
    ImageLatentCondition,
    Modality,
    TextEmbedCondition,
)

# ---------------------------------------------------------------------------
# Modality identity
# ---------------------------------------------------------------------------


def test_text_condition_modality():
    t = TextEmbedCondition(embeds=torch.zeros(2, 8, 16))
    assert t.modality is Modality.TEXT
    assert t.batch_size == 2


def test_image_latent_condition_modality():
    i = ImageLatentCondition(latents=torch.zeros(3, 4, 8, 8))
    assert i.modality is Modality.IMAGE
    assert i.batch_size == 3


# ---------------------------------------------------------------------------
# Concat / select
# ---------------------------------------------------------------------------


def test_text_condition_concat_across_shards():
    a = TextEmbedCondition(embeds=torch.zeros(2, 4, 8))
    b = TextEmbedCondition(embeds=torch.ones(3, 4, 8))
    merged = TextEmbedCondition.concat([a, b])
    assert merged.embeds.shape == (5, 4, 8)
    assert torch.equal(merged.embeds[:2], torch.zeros(2, 4, 8))
    assert torch.equal(merged.embeds[2:], torch.ones(3, 4, 8))


def test_text_condition_select_picks_correct_rows():
    embeds = torch.arange(40, dtype=torch.float).reshape(5, 4, 2)
    t = TextEmbedCondition(embeds=embeds)
    sel = t.select(torch.tensor([0, 2, 4]))
    assert sel.embeds.shape == (3, 4, 2)
    assert torch.equal(sel.embeds[0], embeds[0])
    assert torch.equal(sel.embeds[1], embeds[2])
    assert torch.equal(sel.embeds[2], embeds[4])


def test_image_latent_condition_concat_preserves_shape():
    a = ImageLatentCondition(latents=torch.zeros(1, 4, 8, 8))
    b = ImageLatentCondition(latents=torch.ones(2, 4, 8, 8))
    merged = ImageLatentCondition.concat([a, b])
    assert merged.latents.shape == (3, 4, 8, 8)


def test_image_embed_condition_modality():
    e = ImageEmbedCondition(embeds=torch.zeros(2, 16, 768), attn_mask=torch.ones(2, 16))
    assert e.modality is Modality.IMAGE
    assert e.batch_size == 2


def test_image_embed_condition_concat_across_shards():
    a = ImageEmbedCondition(embeds=torch.zeros(2, 16, 768), attn_mask=torch.ones(2, 16))
    b = ImageEmbedCondition(embeds=torch.ones(3, 16, 768), attn_mask=torch.ones(3, 16))
    merged = ImageEmbedCondition.concat([a, b])
    assert merged.embeds.shape == (5, 16, 768)
    assert merged.attn_mask.shape == (5, 16)
    assert torch.equal(merged.embeds[:2], torch.zeros(2, 16, 768))
    assert torch.equal(merged.embeds[2:], torch.ones(3, 16, 768))


def test_image_embed_condition_select_picks_correct_rows():
    embeds = torch.arange(2 * 4 * 3, dtype=torch.float).reshape(2, 4, 3)
    e = ImageEmbedCondition(embeds=embeds, attn_mask=torch.ones(2, 4))
    sel = e.select(torch.tensor([1]))
    assert sel.embeds.shape == (1, 4, 3)
    assert torch.equal(sel.embeds[0], embeds[1])
    assert sel.attn_mask.shape == (1, 4)


# ---------------------------------------------------------------------------
# Conditions = Dict[str, Condition]
# ---------------------------------------------------------------------------


def test_conditions_alias_is_dict():
    conds: Conditions = {
        "text": TextEmbedCondition(embeds=torch.zeros(2, 4, 8)),
        "image_grid": ImageLatentCondition(latents=torch.zeros(2, 4, 8, 8)),
    }
    assert "text" in conds
    assert isinstance(conds["text"], Condition)
    assert isinstance(conds["image_grid"], Condition)
