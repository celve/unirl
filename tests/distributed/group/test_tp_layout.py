"""Unit tests for the grouped-TP rollout layout + DP-head selection (LIN-535).

Covers the pure-Python framework logic that turns ``rollout.config.tp_size`` into a
``(dp, tp)`` rank layout and a ``DP_HEAD`` execute set. The engine launch + NCCL group
formation are the runtime's job and are exercised by the GPU probes, not here.
"""

from __future__ import annotations

from unirl.distributed.group.dispatch import _is_dp_head
from unirl.distributed.group.handle import _build_rank_infos, _tp_size_from_init_kwargs


def _heads(rank_infos):
    return [i for i, ri in enumerate(rank_infos) if _is_dp_head(ri)]


# ── _tp_size_from_init_kwargs ────────────────────────────────────────────────


def test_tp_read_plain_engine():
    assert _tp_size_from_init_kwargs({"config": {"tp_size": 4}}, 8) == 4


def test_tp_read_agentic_inner():
    # agentic: the inner single-turn engine carries tp_size
    assert _tp_size_from_init_kwargs({"config": {"inner": {"tp_size": 2}}}, 8) == 2


def test_tp_defaults_and_divisibility():
    assert _tp_size_from_init_kwargs({"config": {"tp_size": 1}}, 8) == 1
    assert _tp_size_from_init_kwargs({"config": {"tp_size": 3}}, 8) == 1  # 8 % 3 != 0 → 1
    assert _tp_size_from_init_kwargs(None, 8) == 1
    assert _tp_size_from_init_kwargs({}, 8) == 1
    assert _tp_size_from_init_kwargs({"config": {}}, 8) == 1


# ── _build_rank_infos: (dp, tp) grouping ─────────────────────────────────────


def test_tp_grouping_heads_and_ranks():
    # 8 workers, tp=4 → dp=2 groups; heads at 0 and 4.
    ris = _build_rank_infos(8, sp_size=1, tp_size=4)
    assert _heads(ris) == [0, 4]
    assert [ri.dp_rank for ri in ris] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert [ri.tp_rank for ri in ris] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert ris[0].dp_size == 2 and ris[0].tp_size == 4


def test_tp1_is_flat_dp_every_rank_is_head():
    # tp=1 → DP_HEAD ≡ ALL (unchanged flat-DP path).
    ris = _build_rank_infos(4, sp_size=1, tp_size=1)
    assert _heads(ris) == [0, 1, 2, 3]
    assert [ri.dp_rank for ri in ris] == [0, 1, 2, 3]
    assert all(ri.tp_rank == 0 and ri.tp_size == 1 for ri in ris)


def test_sp_arm_unchanged():
    # SP (training) layout is untouched by the TP addition; tp stays flat.
    ris = _build_rank_infos(4, sp_size=2, tp_size=1)
    assert [ri.sp_rank for ri in ris] == [0, 1, 0, 1]
    assert [ri.dp_rank for ri in ris] == [0, 0, 1, 1]
    assert all(ri.tp_rank == 0 for ri in ris)
    assert _heads(ris) == [0, 2]  # sp-head == tp-head == group head
