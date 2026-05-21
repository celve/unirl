"""Tests for the new ``sde_logp`` / ``sde_indices`` fields on ``LatentSegment``."""

from __future__ import annotations

import torch

from diffusionrl.types.segments import LatentSegment


def test_latent_segment_sde_logp_default_none():
    seg = LatentSegment(
        sample_indices=torch.tensor([0, 1]),
        positions=torch.tensor([0, 0]),
        latents=torch.zeros(2, 4, 16, 8, 8),
    )
    assert seg.sde_logp is None
    assert seg.sde_indices is None


def test_latent_segment_sde_logp_storage_roundtrip():
    seg = LatentSegment(
        sample_indices=torch.tensor([0, 1]),
        positions=torch.tensor([0, 0]),
        latents=torch.zeros(2, 4, 16, 8, 8),
        sde_logp=torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        sde_indices=torch.tensor([5, 6, 7], dtype=torch.long),
    )
    # [N=2, S=3]
    assert seg.sde_logp.shape == (2, 3)
    # Index s=1 -> step 6
    assert int(seg.sde_indices[1].item()) == 6


def test_latent_segment_concat_extends_sde_logp_along_batch():
    a = LatentSegment(
        sample_indices=torch.tensor([0]),
        positions=torch.tensor([0]),
        latents=torch.zeros(1, 4, 16, 8, 8),
        sde_logp=torch.tensor([[1.0, 2.0]]),
        sde_indices=torch.tensor([3, 7], dtype=torch.long),
    )
    b = LatentSegment(
        sample_indices=torch.tensor([0]),
        positions=torch.tensor([0]),
        latents=torch.ones(1, 4, 16, 8, 8),
        sde_logp=torch.tensor([[10.0, 20.0]]),
        sde_indices=torch.tensor([3, 7], dtype=torch.long),
    )
    merged = LatentSegment.concat([a, b])
    # sde_logp is concat along dim 0 (batch) -> [2, 2]
    assert merged.sde_logp.shape == (2, 2)
    assert merged.sde_logp.tolist() == [[1.0, 2.0], [10.0, 20.0]]
    # sde_indices is shared across the batch -> first shard wins (it's identical anyway).
    assert merged.sde_indices.tolist() == [3, 7]
