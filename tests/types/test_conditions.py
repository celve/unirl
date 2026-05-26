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
    TextTokenCondition,
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


def test_text_token_condition_concat_pads_to_common_seq_len():
    """Cross-shard concat right-pads variable seq_len to the global max.

    SGLang's per-shard build_rollout_resp pads to each shard's in-batch
    max — different shards yield different lengths. Without the
    pad-aware concat, ``Batched.concat`` raises in ``torch.cat(dim=0)``
    on the dim-1 mismatch (the bug that blocked pe_joint full training).
    """
    a = TextTokenCondition(
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
    )
    b = TextTokenCondition(
        input_ids=torch.tensor([[4, 5, 6, 7, 8]], dtype=torch.long),
        attention_mask=torch.ones((1, 5), dtype=torch.long),
    )
    merged = TextTokenCondition.concat([a, b])
    assert merged.input_ids.shape == (2, 5)
    # a was padded: first 3 real tokens, then two zero pads.
    assert merged.input_ids[0].tolist() == [1, 2, 3, 0, 0]
    assert merged.input_ids[1].tolist() == [4, 5, 6, 7, 8]
    # attention_mask zeroes the pad positions on a, full ones for b.
    assert merged.attention_mask[0].tolist() == [1, 1, 1, 0, 0]
    assert merged.attention_mask[1].tolist() == [1, 1, 1, 1, 1]


def test_text_token_condition_concat_single_item_passthrough():
    """One-element concat reduces to a passthrough (no spurious padding)."""
    a = TextTokenCondition(
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
    )
    merged = TextTokenCondition.concat([a])
    assert merged.input_ids.shape == (1, 3)
    assert torch.equal(merged.input_ids, a.input_ids)


def test_text_token_condition_concat_same_length_no_pad():
    """When all shards already share a seq_len, the fast path skips padding."""
    a = TextTokenCondition(
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
    )
    b = TextTokenCondition(
        input_ids=torch.tensor([[4, 5, 6]], dtype=torch.long),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
    )
    merged = TextTokenCondition.concat([a, b])
    assert merged.input_ids.shape == (2, 3)
    assert merged.input_ids[0].tolist() == [1, 2, 3]
    assert merged.input_ids[1].tolist() == [4, 5, 6]


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
