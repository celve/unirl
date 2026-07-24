"""Unit tests for the per-prompt driver noise/seed slice (Option A, LIN-513).

Locks the ``_slice_request_drivers`` contract behind ``patch_request_noise_slice``:
a full ``[num_prompts, ...]`` driver-noise tensor and a length-``num_prompts`` seed
list are sliced to this prompt's GLOBAL row; everything else passes through
unchanged. The helper is duck-typed, so the core cases need no torch.
"""

from __future__ import annotations

import pytest

from unirl.rollout.engine.sglang_diffusion._patches.patch_request_noise_slice import (
    _slice_request_drivers,
)


class _FakeTensor:
    """Duck-typed stand-in with ``.shape`` and row slicing (no torch needed)."""

    def __init__(self, rows):
        self.rows = list(rows)

    @property
    def shape(self):
        return (len(self.rows),)

    def __getitem__(self, key):
        return _FakeTensor(self.rows[key]) if isinstance(key, slice) else self.rows[key]

    def __eq__(self, other):
        return isinstance(other, _FakeTensor) and other.rows == self.rows


def test_slices_each_global_row_and_seed():
    noise = _FakeTensor(["x0", "x1", "x2", "x3"])
    seeds = ["s0", "s1", "s2", "s3"]
    for i in range(4):
        lat, sd = _slice_request_drivers(noise, seeds, prompt_index=i, num_prompts=4)
        assert lat == _FakeTensor([f"x{i}"])
        assert sd == [f"s{i}"]


def test_multigroup_uses_global_index_not_within_group():
    # G=2, K=2 -> B=4. Prompt 2 (group 1, sample 0) must map to GLOBAL row 2 --
    # exactly what the deleted within-group worker slice got wrong for G>1.
    noise = _FakeTensor([10, 11, 12, 13])
    lat, sd = _slice_request_drivers(noise, ["a", "b", "c", "d"], prompt_index=2, num_prompts=4)
    assert lat == _FakeTensor([12])
    assert sd == ["c"]


def test_passthrough_single_prompt():
    noise = _FakeTensor(["only"])
    lat, sd = _slice_request_drivers(noise, ["s"], prompt_index=0, num_prompts=1)
    assert lat is noise
    assert sd == ["s"]


def test_passthrough_none_drivers():
    lat, sd = _slice_request_drivers(None, None, prompt_index=1, num_prompts=4)
    assert lat is None
    assert sd is None


def test_passthrough_on_batch_length_mismatch():
    noise = _FakeTensor([0, 1])  # shape[0]=2 != num_prompts=4
    seeds = ["a", "b"]  # len 2 != 4
    lat, sd = _slice_request_drivers(noise, seeds, prompt_index=1, num_prompts=4)
    assert lat is noise
    assert sd is seeds


def test_real_torch_tensor_row_slice():
    torch = pytest.importorskip("torch")
    noise = torch.randn(4, 3, 8, 8)
    lat, sd = _slice_request_drivers(noise, ["s0", "s1", "s2", "s3"], prompt_index=2, num_prompts=4)
    assert lat.shape[0] == 1
    assert torch.equal(lat[0], noise[2])
    assert sd == ["s2"]
