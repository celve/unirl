"""Tests for the new reduction field kinds in ``diffusionrl.utils.batched``
and for ``RewardResponse`` as a ``Batched``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pytest
import torch

from diffusionrl.types.reward import RewardResponse
from diffusionrl.utils.batched import (
    Batched,
    concat_field,
    max_field,
    mean_field,
    min_field,
    sum_field,
)


@dataclass
class _Toy(Batched):
    rows: List[int] = concat_field(default_factory=list)
    hi: float = max_field(default=0.0)
    lo: float = min_field(default=0.0)
    total: float = sum_field(default=0.0)
    avg: float = mean_field(default=0.0)


def test_reduction_fields_concat_scalars() -> None:
    a = _Toy(rows=[1, 2], hi=1.0, lo=1.0, total=1.0, avg=1.0)
    b = _Toy(rows=[3], hi=5.0, lo=-2.0, total=2.0, avg=3.0)
    c = _Toy(rows=[4, 5], hi=3.0, lo=0.0, total=4.0, avg=5.0)

    out = _Toy.concat([a, b, c])

    assert out.rows == [1, 2, 3, 4, 5]
    assert out.batch_size == 5
    assert out.hi == 5.0
    assert out.lo == -2.0
    assert out.total == pytest.approx(7.0)
    assert out.avg == pytest.approx(3.0)


def test_reduction_fields_concat_tensors() -> None:
    @dataclass
    class _TensorToy(Batched):
        rows: List[int] = concat_field(default_factory=list)
        hi: torch.Tensor = max_field(default_factory=lambda: torch.zeros(2))
        total: torch.Tensor = sum_field(default_factory=lambda: torch.zeros(2))

    a = _TensorToy(rows=[0], hi=torch.tensor([1.0, 4.0]), total=torch.tensor([1.0, 2.0]))
    b = _TensorToy(rows=[0], hi=torch.tensor([3.0, 2.0]), total=torch.tensor([4.0, 5.0]))

    out = _TensorToy.concat([a, b])

    assert torch.equal(out.hi, torch.tensor([3.0, 4.0]))
    assert torch.equal(out.total, torch.tensor([5.0, 7.0]))


def test_reduction_fields_skip_none_entries() -> None:
    @dataclass
    class _MaybeToy(Batched):
        rows: List[int] = concat_field(default_factory=list)
        hi: Optional[float] = max_field(default=None)

    a = _MaybeToy(rows=[0], hi=2.0)
    b = _MaybeToy(rows=[0], hi=None)
    c = _MaybeToy(rows=[0], hi=5.0)

    out = _MaybeToy.concat([a, b, c])
    assert out.hi == 5.0

    all_none = _MaybeToy.concat([
        _MaybeToy(rows=[0], hi=None),
        _MaybeToy(rows=[0], hi=None),
    ])
    assert all_none.hi is None


def test_select_and_slice_pass_reduction_fields_through() -> None:
    inst = _Toy(rows=[10, 20, 30, 40], hi=9.0, lo=-1.0, total=3.0, avg=2.0)

    sel = inst.select(torch.tensor([0, 2]))
    assert sel.rows == [10, 30]
    assert sel.hi == 9.0
    assert sel.lo == -1.0
    assert sel.total == 3.0
    assert sel.avg == 2.0

    sliced = inst.slice(1, 3)
    assert sliced.rows == [20, 30]
    assert sliced.hi == 9.0
    assert sliced.lo == -1.0
    assert sliced.total == 3.0
    assert sliced.avg == 2.0


def test_reward_response_concat() -> None:
    a = RewardResponse(
        rewards=[0.1, 0.2],
        component_rewards={"clip": [0.3, 0.4]},
        successes=[True, True],
        errors=[None, None],
        compute_time=0.5,
    )
    b = RewardResponse(
        rewards=[0.9],
        component_rewards={"clip": [0.8]},
        successes=[False],
        errors=["boom"],
        compute_time=1.2,
    )

    out = RewardResponse.concat([a, b])

    assert out.rewards == [0.1, 0.2, 0.9]
    assert out.component_rewards == {"clip": [0.3, 0.4, 0.8]}
    assert out.successes == [True, True, False]
    assert out.errors == [None, None, "boom"]
    assert out.compute_time == 1.2
    assert out.batch_size == 3


def test_reward_response_concat_empty_components() -> None:
    a = RewardResponse(
        rewards=[0.1],
        component_rewards={},
        successes=[True],
        errors=[None],
        compute_time=0.1,
    )
    b = RewardResponse(
        rewards=[0.5],
        component_rewards={},
        successes=[True],
        errors=[None],
        compute_time=0.3,
    )

    out = RewardResponse.concat([a, b])

    assert out.rewards == [0.1, 0.5]
    assert out.component_rewards == {}
    assert out.compute_time == pytest.approx(0.3)
