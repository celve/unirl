"""Controller-side Ulysses SP rank-layout unit tests (CPU, pytest).

The multi-GPU torchrun scripts in this directory validate the *model-side* SP
(slice/gather + all-to-all) by calling ``init_parallel_state`` directly with an
explicit ``dp_size``/``ulysses_size``. They do NOT exercise the *controller*
layer — how a ``Handle`` derives its ``(dp, sp)`` rank layout and how that layout
feeds ``DP_SCATTER`` data dispatch.

That layer had a bug: the VeOmni backend handle got the SP layout from its
``fsdp_cfg.sp_size``, but dependent roles that dispatch sequence data to the SP
model — the train stack (``fsdp_backend=<SP backend>``) and the trainside
diffusion rollout (samples through the SP model) — stayed flat (sp=1). Their
``DP_SCATTER`` then sharded a batch across ALL world ranks, feeding the two ranks
of an SP pair *different* shards; the model's Ulysses all-to-all saw mismatched
shapes on the two ranks and hung (NCCL ALLTOALL_BASE watchdog timeout). These
tests lock the fix.
"""

import pytest

pytest.importorskip("ray")  # handle.py imports ray at module load

from unirl.distributed.group.handle import (  # noqa: E402
    HandleRef,
    _build_rank_infos,
    _sp_size_from_init_kwargs,
)


class _FsdpCfg:
    """Minimal stand-in for FSDPConfig (only sp_size is read)."""

    def __init__(self, sp_size: int) -> None:
        self.sp_size = sp_size


# ── _build_rank_infos: contiguous (dp, sp) blocks matching VeOmni's mesh ──


def test_rank_infos_flat_when_sp1():
    infos = _build_rank_infos(world_size=8, sp_size=1)
    assert [ri.dp_rank for ri in infos] == list(range(8))
    assert all(ri.dp_size == 8 and ri.sp_size == 1 and ri.sp_rank == 0 for ri in infos)


def test_rank_infos_contiguous_sp_pairs():
    # sp=2 on 8 ranks -> dp=4; SP groups must be CONTIGUOUS pairs (0,1)(2,3)...
    # to match VeOmni's init_device_mesh(mesh_dim_names=("dp_shard","ulysses"))
    # where ulysses is the innermost (fastest-varying) dim:
    #   global_rank = dp_shard_idx * ulysses_size + ulysses_idx.
    infos = _build_rank_infos(world_size=8, sp_size=2)
    assert [ri.dp_rank for ri in infos] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [ri.sp_rank for ri in infos] == [0, 1, 0, 1, 0, 1, 0, 1]
    assert all(ri.dp_size == 4 and ri.sp_size == 2 for ri in infos)


def test_sp_pair_shares_dp_rank_so_dp_scatter_replicates():
    # The core invariant the bug violated: both ranks of an SP group share a
    # dp_rank, so DP_SCATTER (which maps worker -> dp_shards[dp_rank]) hands them
    # the SAME shard. If they differed, the all-to-all desyncs.
    for sp in (2, 4):
        infos = _build_rank_infos(world_size=8, sp_size=sp)
        groups = {}
        for ri in infos:
            groups.setdefault(ri.rank // sp, set()).add(ri.dp_rank)
        # every contiguous block of `sp` ranks resolves to a single dp_rank
        assert all(len(dps) == 1 for dps in groups.values())


# ── _sp_size_from_init_kwargs: who adopts the SP layout ──


def test_backend_reads_sp_from_fsdp_cfg():
    assert _sp_size_from_init_kwargs({"fsdp_cfg": _FsdpCfg(2)}, world_size=8) == 2


def test_stack_inherits_sp_from_fsdp_backend_sibling():
    # The regression case: TrainStack holds fsdp_backend=<SP backend handle>.
    kw = {"micro_batch_size": 1, "fsdp_backend": HandleRef("Backend_0", sp_size=2),
          "algorithm": HandleRef("Alg_0", sp_size=1)}
    assert _sp_size_from_init_kwargs(kw, world_size=8) == 2


def test_trainside_rollout_inherits_sp_from_explicit_hint():
    # The trainside diffusion rollout gets sp_size=backend.sp_size as a hint
    # alongside a flat pipeline sibling; the hint must win.
    kw = {"pipeline": HandleRef("Pipe_0", sp_size=1), "sp_size": 2}
    assert _sp_size_from_init_kwargs(kw, world_size=8) == 2


def test_flat_when_no_sp_signal():
    # sglang rollout / reward: no fsdp_cfg, no SP sibling -> stays flat.
    assert _sp_size_from_init_kwargs({"pipeline": HandleRef("Pipe_0", sp_size=1)}, world_size=8) == 1
    assert _sp_size_from_init_kwargs({}, world_size=8) == 1


def test_sp1_hint_is_noop():
    # SP off: backend.sp_size=1 passed through -> flat, existing recipes unaffected.
    assert _sp_size_from_init_kwargs({"pipeline": HandleRef("P", 1), "sp_size": 1}, world_size=8) == 1


def test_non_divisible_sp_falls_back_to_flat():
    # sp must evenly divide world_size, else flat (guard against bad configs).
    assert _sp_size_from_init_kwargs({"fsdp_cfg": _FsdpCfg(3)}, world_size=8) == 1
    assert _sp_size_from_init_kwargs({"fsdp_backend": HandleRef("B", sp_size=3)}, world_size=8) == 1


def test_max_sp_among_siblings():
    # If multiple SP siblings (unusual), take the largest.
    kw = {"a": HandleRef("A", sp_size=2), "b": HandleRef("B", sp_size=4)}
    assert _sp_size_from_init_kwargs(kw, world_size=8) == 4
