"""Tests for RolloutResp container and its concat sample_indices remap."""

from __future__ import annotations

import torch

from diffusionrl.types.conditions import TextEmbedCondition
from diffusionrl.types.primitives import Image, Images
from diffusionrl.types.rollout_resp import RolloutResp
from diffusionrl.types.segments import LatentSegment


def _make_shard(
    sample_ids,
    *,
    text_embeds,
    image_seg_sample_indices,
    image_seg_positions,
    image_latents,
    decoded_pixels,
    rewards,
):
    return RolloutResp(
        conditions={"text": TextEmbedCondition(embeds=text_embeds)},
        rollout_traces={
            "image_latent": LatentSegment(
                sample_indices=image_seg_sample_indices,
                positions=image_seg_positions,
                latents=image_latents,
            )
        },
        decoded={
            "image_latent": Images.from_list([Image(pixels=p) for p in decoded_pixels]),
        },
        sample_ids=sample_ids,
        group_ids=["g"] * len(sample_ids),
        rewards=rewards,
    )


def test_concat_remaps_sample_indices_via_offset_shift():
    a = _make_shard(
        sample_ids=["s0", "s1"],
        text_embeds=torch.zeros(2, 4, 8),
        image_seg_sample_indices=torch.tensor([0, 1]),
        image_seg_positions=torch.tensor([0, 0]),
        image_latents=torch.zeros(2, 2, 4, 4, 4),
        decoded_pixels=[torch.zeros(3, 4, 4), torch.zeros(3, 4, 4)],
        rewards=torch.tensor([0.5, 0.7]),
    )
    b = _make_shard(
        sample_ids=["s2"],
        text_embeds=torch.ones(1, 4, 8),
        image_seg_sample_indices=torch.tensor([0]),
        image_seg_positions=torch.tensor([0]),
        image_latents=torch.ones(1, 2, 4, 4, 4),
        decoded_pixels=[torch.ones(3, 4, 4)],
        rewards=torch.tensor([0.9]),
    )

    merged = RolloutResp.concat([a, b])
    assert merged.batch_size == 3
    assert merged.sample_ids == ["s0", "s1", "s2"]

    # Segment sample_indices are offset-shifted in the second shard.
    seg = merged.rollout_traces["image_latent"]
    assert seg.sample_indices.tolist() == [0, 1, 2]
    assert seg.positions.tolist() == [0, 0, 0]
    assert seg.latents.shape == (3, 2, 4, 4, 4)


def test_concat_merges_dict_keyed_fields_by_key():
    a = _make_shard(
        sample_ids=["s0"],
        text_embeds=torch.zeros(1, 4, 8),
        image_seg_sample_indices=torch.tensor([0]),
        image_seg_positions=torch.tensor([0]),
        image_latents=torch.zeros(1, 2, 4, 4, 4),
        decoded_pixels=[torch.zeros(3, 4, 4)],
        rewards=torch.tensor([0.1]),
    )
    b = _make_shard(
        sample_ids=["s1"],
        text_embeds=torch.ones(1, 4, 8),
        image_seg_sample_indices=torch.tensor([0]),
        image_seg_positions=torch.tensor([0]),
        image_latents=torch.ones(1, 2, 4, 4, 4),
        decoded_pixels=[torch.ones(3, 4, 4)],
        rewards=torch.tensor([0.2]),
    )

    merged = RolloutResp.concat([a, b])

    # Conditions merged by key.
    assert "text" in merged.conditions
    assert merged.conditions["text"].embeds.shape == (2, 4, 8)

    # Decoded merged by key.
    assert merged.decoded["image_latent"].pixels.shape == (2, 3, 4, 4)


def test_concat_does_not_mutate_input_shards():
    a = _make_shard(
        sample_ids=["s0"],
        text_embeds=torch.zeros(1, 4, 8),
        image_seg_sample_indices=torch.tensor([0]),
        image_seg_positions=torch.tensor([0]),
        image_latents=torch.zeros(1, 2, 4, 4, 4),
        decoded_pixels=[torch.zeros(3, 4, 4)],
        rewards=torch.tensor([0.1]),
    )
    b = _make_shard(
        sample_ids=["s1"],
        text_embeds=torch.ones(1, 4, 8),
        image_seg_sample_indices=torch.tensor([0]),
        image_seg_positions=torch.tensor([0]),
        image_latents=torch.ones(1, 2, 4, 4, 4),
        decoded_pixels=[torch.ones(3, 4, 4)],
        rewards=torch.tensor([0.2]),
    )
    RolloutResp.concat([a, b])
    # Second shard's segment sample_indices unchanged.
    assert b.rollout_traces["image_latent"].sample_indices.tolist() == [0]


def test_slice_picks_subset_along_sample_axis():
    a = _make_shard(
        sample_ids=["s0", "s1", "s2"],
        text_embeds=torch.zeros(3, 4, 8),
        image_seg_sample_indices=torch.tensor([0, 1, 2]),
        image_seg_positions=torch.tensor([0, 0, 0]),
        image_latents=torch.zeros(3, 2, 4, 4, 4),
        decoded_pixels=[torch.zeros(3, 4, 4)] * 3,
        rewards=torch.tensor([0.1, 0.2, 0.3]),
    )
    sub = a.slice(1, 3)
    assert sub.batch_size == 2
    assert sub.sample_ids == ["s1", "s2"]


def test_per_sample_raw_indexing():
    """Confirm the no-SegmentView access pattern works."""
    a = _make_shard(
        sample_ids=["s0", "s1"],
        text_embeds=torch.zeros(2, 4, 8),
        image_seg_sample_indices=torch.tensor([0, 0, 1]),  # sample 0 has two segments
        image_seg_positions=torch.tensor([0, 1, 0]),
        image_latents=torch.arange(3 * 2 * 4 * 4 * 4, dtype=torch.float).reshape(3, 2, 4, 4, 4),
        decoded_pixels=[torch.zeros(3, 4, 4), torch.zeros(3, 4, 4)],
        rewards=torch.tensor([0.1, 0.2]),
    )

    seg = a.rollout_traces["image_latent"]
    mask = seg.sample_indices == 0
    sample_0_latents = seg.latents[mask]
    assert sample_0_latents.shape == (2, 2, 4, 4, 4)

    mask = seg.sample_indices == 1
    sample_1_latents = seg.latents[mask]
    assert sample_1_latents.shape == (1, 2, 4, 4, 4)
