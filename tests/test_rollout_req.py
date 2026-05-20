"""Tests for RolloutReq — primitives + request_conditions container."""

from __future__ import annotations

import torch

from diffusionrl.types.conditions.image import ImageLatentCondition
from diffusionrl.types.primitives import Image, Images, Text, Texts
from diffusionrl.types.rollout_req import RolloutReq


def _make_shard(sample_ids, group_ids, *, texts=None, images=None, stage_params=None):
    primitives: dict = {}
    if texts is not None:
        primitives["text"] = Texts.from_list([Text(t) for t in texts])
    if images is not None:
        primitives["image"] = Images.from_list([Image(pixels=p) for p in images])
    return RolloutReq(
        sample_ids=list(sample_ids),
        group_ids=list(group_ids),
        primitives=primitives,
        stage_params=stage_params or {},
    )


def test_construction_and_field_access():
    req = _make_shard(
        sample_ids=["s0", "s1"],
        group_ids=["g0", "g0"],
        texts=["a", "b"],
        stage_params={"diffusion": {"num_steps": 28}, "ar": {"max_tokens": 64}},
    )
    assert req.batch_size == 2
    assert req.sample_ids == ["s0", "s1"]
    assert req.primitives["text"].texts == ["a", "b"]
    assert req.stage_params["diffusion"]["num_steps"] == 28


def test_concat_extends_sample_id_lists():
    a = _make_shard(["s0", "s1"], ["g0", "g0"], texts=["a", "b"])
    b = _make_shard(["s2"], ["g1"], texts=["c"])

    merged = RolloutReq.concat([a, b])

    assert merged.batch_size == 3
    assert merged.sample_ids == ["s0", "s1", "s2"]
    assert merged.group_ids == ["g0", "g0", "g1"]


def test_concat_merges_primitives_by_key():
    a = _make_shard(["s0", "s1"], ["g0", "g0"], texts=["a", "b"])
    b = _make_shard(["s2"], ["g0"], texts=["c"])

    merged = RolloutReq.concat([a, b])
    assert merged.primitives["text"].texts == ["a", "b", "c"]


def test_concat_merges_multimodal_primitives_by_key():
    img_zero = torch.zeros(3, 4, 4)
    img_one = torch.ones(3, 4, 4)
    a = _make_shard(["s0"], ["g0"], texts=["a"], images=[img_zero])
    b = _make_shard(["s1"], ["g0"], texts=["b"], images=[img_one])

    merged = RolloutReq.concat([a, b])
    assert merged.primitives["text"].texts == ["a", "b"]
    assert merged.primitives["image"].pixels.shape == (2, 3, 4, 4)
    assert torch.equal(merged.primitives["image"].pixels[0], img_zero)
    assert torch.equal(merged.primitives["image"].pixels[1], img_one)


def test_stage_params_shared_consistent_values_survive_concat():
    a = _make_shard(["s0"], ["g0"], texts=["a"], stage_params={"diffusion": {"num_steps": 28}})
    b = _make_shard(["s1"], ["g0"], texts=["b"], stage_params={"diffusion": {"num_steps": 28}})

    merged = RolloutReq.concat([a, b])
    assert merged.stage_params == {"diffusion": {"num_steps": 28}}


def test_select_picks_subset_along_sample_axis():
    a = _make_shard(["s0", "s1", "s2"], ["g0", "g0", "g0"], texts=["a", "b", "c"])
    sub = a.select(torch.tensor([0, 2]))
    assert sub.batch_size == 2
    assert sub.sample_ids == ["s0", "s2"]
    assert sub.primitives["text"].texts == ["a", "c"]


def test_slice_picks_range_along_sample_axis():
    a = _make_shard(["s0", "s1", "s2"], ["g0", "g0", "g0"], texts=["a", "b", "c"])
    sub = a.slice(1, 3)
    assert sub.batch_size == 2
    assert sub.sample_ids == ["s1", "s2"]
    assert sub.primitives["text"].texts == ["b", "c"]


def test_empty_primitives_dict_is_valid():
    req = RolloutReq(sample_ids=["s0"], group_ids=["g0"])
    assert req.batch_size == 1
    assert req.primitives == {}
    assert req.request_conditions == {}
    assert req.stage_params == {}


def test_request_conditions_slice_propagates_to_condition_tensors():
    """Slicing a RolloutReq must slice tensors held inside request_conditions."""
    latents = torch.arange(3 * 4 * 5).reshape(3, 4, 5).to(torch.float32)
    req = _make_shard(["s0", "s1", "s2"], ["g0", "g0", "g0"], texts=["a", "b", "c"])
    req.request_conditions = {"initial_latents": ImageLatentCondition(latents=latents)}

    sub = req.slice(1, 3)
    assert sub.batch_size == 2
    assert sub.sample_ids == ["s1", "s2"]
    sub_lat = sub.request_conditions["initial_latents"].latents
    assert sub_lat.shape == (2, 4, 5)
    assert torch.equal(sub_lat, latents[1:3])


def test_request_conditions_concat_merges_condition_tensors():
    """Concatenating RolloutReqs must concat tensors inside request_conditions."""
    lat_a = torch.zeros(2, 4, 5)
    lat_b = torch.ones(1, 4, 5)
    a = _make_shard(["s0", "s1"], ["g0", "g0"], texts=["a", "b"])
    a.request_conditions = {"initial_latents": ImageLatentCondition(latents=lat_a)}
    b = _make_shard(["s2"], ["g1"], texts=["c"])
    b.request_conditions = {"initial_latents": ImageLatentCondition(latents=lat_b)}

    merged = RolloutReq.concat([a, b])
    merged_lat = merged.request_conditions["initial_latents"].latents
    assert merged_lat.shape == (3, 4, 5)
    assert torch.equal(merged_lat[:2], lat_a)
    assert torch.equal(merged_lat[2:], lat_b)
