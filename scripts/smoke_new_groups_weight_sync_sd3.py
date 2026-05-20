"""End-to-end weight-sync smoke for the NEW groups against SD3.

Two-mode smoke — placement & transport choices are driven by ``--mode``:

* ``--mode disjoint`` (default): trainer FSDP DP=4 on GPUs 4..7,
  vllm-omni SD3 rollout TP=4 on GPUs 0..3 (8 GPUs total). Verifies
  ``nccl_broadcast`` — the broadcast group needs the two sides on
  *disjoint* physical devices ("Duplicate GPU detected" otherwise).
* ``--mode colocate``: trainer + rollout share GPUs 0..3 via
  ``colocate=true`` (4 GPUs total). Verifies ``tensor_payload`` and
  ``ipc_bucketed`` — both transports pickle CUDA-IPC handles whose
  device UUIDs only resolve when the receiver sees the same physical
  GPU as the sender. The disjoint placement makes them fail with
  ``Invalid device_uuid=…``.

Each transport: randomize trainer weights → sync → randomize again →
sync → assert the rollout workers' loaded parameter checksums differ
between the two probes. ``NewRolloutActor.loaded_param_checksums`` fans
``_diffrl_loaded_param_checksums`` to every worker rank; equality
across (stage, rank, name) implies the transport correctly delivered
the bytes the trainer shipped.

Usage (run both modes for full coverage)::

    cd ~/diffusionrl && source .venv/bin/activate
    PRETRAINED_MODEL=stabilityai/stable-diffusion-3.5-medium \\
        python scripts/smoke_new_groups_weight_sync_sd3.py --mode disjoint
    PRETRAINED_MODEL=stabilityai/stable-diffusion-3.5-medium \\
        python scripts/smoke_new_groups_weight_sync_sd3.py --mode colocate

Exit 0 on PASS, non-zero if any transport fails. Skip a transport via
``--skip ipc,nccl,tensor`` (comma list).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

# Patterns to exclude from probe targets (same logic as
# ``scripts/smoke_weight_sync_e2e.py``). TP-aware weight loaders shard
# the full unsharded tensor internally and assert on the full shape.
# Our trainer ships per-worker shapes (raw_state_dict materializes
# DTensors via redistribute(Replicate())), so feeding TP-sharded names
# trips ``loaded_weight.shape[output_dim] == self.org_vocab_size`` on
# vocab embeddings and similar. Norms / scalar params have no TP
# layout and round-trip cleanly.
# Prefix the trainer prepends to every parameter name before shipping.
# The trainer's ``self.model = policy.model`` is the bare DiT (state-dict
# keys: ``pos_embed.proj.weight``); the rollout-side StableDiffusion3
# pipeline holds the same module under ``transformer.*``. Mirrors
# verl-omni's hardcoded prefix (diffusers_impl.py:832). Used by both
# the sync config builder and the value-equality check.
_SYNC_PREFIX = "transformer."


_TP_SHARDED_NAME_PATTERNS = (
    "embed_tokens",
    "lm_head",
    "qkv_proj",
    "o_proj",
    "gate_up_proj",
    "down_proj",
    "experts.",
    "to_q",
    "to_k",
    "to_v",
    "to_out",
    "ff.",
)


def _is_tp_sharded_name(name: str) -> bool:
    return any(p in name for p in _TP_SHARDED_NAME_PATTERNS)


def _diff_checksums(
    pre: Dict[int, Any],
    post: Dict[int, Any],
) -> Dict[int, List[str]]:
    """Per stage, list of names whose rank-0 checksum changed between pre and post.

    ``pre[sid]`` and ``post[sid]`` are the rollout actor's
    ``loaded_param_checksums(...)`` result for that stage: a list of
    per-rank ``{name: hex}`` dicts. We compare rank-0 since the smoke's
    probe filter keeps only TP-replicated params (every rank holds the
    same bytes).
    """
    changed: Dict[int, List[str]] = {}
    for sid in pre:
        pre_ranks = pre[sid] if isinstance(pre[sid], list) else [pre[sid]]
        post_ranks = post.get(sid, [])
        if isinstance(post_ranks, dict):
            post_ranks = [post_ranks]
        pre_r0 = pre_ranks[0] if pre_ranks and isinstance(pre_ranks[0], dict) else {}
        post_r0 = post_ranks[0] if post_ranks and isinstance(post_ranks[0], dict) else {}
        changed[sid] = sorted(name for name, hex_ in pre_r0.items() if post_r0.get(name) and post_r0[name] != hex_)
    return changed


_EXPERIMENT_BY_MODE = {
    "disjoint": "smoke_vllm_omni_sd3_weight_sync",
    "colocate": "smoke_vllm_omni_sd3_weight_sync_colocate",
}

# Transports that work in each placement mode. Hardware constraints —
# not policy. NCCL's broadcast group rejects duplicate physical devices
# ("Duplicate GPU detected"); tensor_payload and ipc_bucketed both
# pickle CUDA-IPC handles whose device UUIDs only resolve when the
# receiver sees the same physical GPU as the sender.
_TRANSPORTS_BY_MODE = {
    "disjoint": ("nccl_broadcast",),
    "colocate": ("tensor_payload", "ipc_bucketed"),
}


def _build_cfg(mode: str):
    """Compose the Hydra cfg via the mode's experiment yaml."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from diffusionrl.config import register_all_configs
    from diffusionrl.config.polymorphic import expand_polymorphic_lists

    register_all_configs()
    exp_name = _EXPERIMENT_BY_MODE[mode]

    conf_dir = (Path(__file__).resolve().parent.parent / "conf").as_posix()
    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[f"+experiment={exp_name}"],
        )
    OmegaConf.resolve(cfg)
    expand_polymorphic_lists(cfg)
    return cfg


def _make_sync_cfg(name: str):
    """Build a structured sync config for transport ``name``.

    The handlers register their config dataclasses with ``expand=True`` so
    ``build()`` calls ``hydra.utils.instantiate(cfg, **deps)`` — which
    requires the cfg to be a structured-typed ``DictConfig`` whose class
    carries the ``_hydra_expand_`` attribute. Build via ``OmegaConf.structured``
    on the registered config dataclass so the schema rides through.
    """
    from omegaconf import OmegaConf

    from diffusionrl.distributed.weight_sync.ipc import IPCBucketedSyncConfig
    from diffusionrl.distributed.weight_sync.nccl import NcclBroadcastSyncConfig
    from diffusionrl.distributed.weight_sync.tensor import TensorPayloadSyncConfig

    if name == "nccl_broadcast":
        cfg = NcclBroadcastSyncConfig(
            bucket_size=256,
            flush_cache=True,
            target_modules=("transformer",),
            stage_ids=(),
        )
    elif name == "tensor_payload":
        cfg = TensorPayloadSyncConfig(
            bucket_size=256,
            flush_cache=True,
            target_modules=("transformer",),
        )
    elif name == "ipc_bucketed":
        cfg = IPCBucketedSyncConfig(
            bucket_size=2048,
            bucket_size_mb=2048,
            flush_cache=True,
            use_shm=False,
            target_modules=("transformer",),
            stage_ids=(),
        )
    else:
        raise ValueError(f"unknown transport name: {name!r}")
    return OmegaConf.structured(cfg)


def _probe_param_names(rollout_group, limit: int = 16) -> List[str]:
    """Ask the rollout actor for real param names and return TP-flat
    ``transformer.*`` names.

    The trainer ships ``raw_state_dict(bundle.transformer)`` which yields
    unprefixed keys (``pos_embed.proj.weight``). vllm-omni's SD3 pipeline
    loads weights through a ``ComponentSource(prefix="transformer.")``,
    so the worker's parameter dict actually holds those as
    ``transformer.pos_embed.proj.weight``. Probe only those — text encoders /
    VAE are sibling modules and are never in flight for a
    ``target_modules=("transformer",)`` sync.
    """
    import ray

    actors = rollout_group.get_rollout_actors()
    per_actor = ray.get([a.list_param_names.remote(stage_ids=[0]) for a in actors])
    if not per_actor:
        return []
    sets = [set(d.get(0, [])) for d in per_actor]
    common = set.intersection(*sets) if sets else set()
    safe = sorted(n for n in common if n.startswith("transformer.") and not _is_tp_sharded_name(n))
    return safe[: int(limit)]


def _collect_checksums(rollout_group, names: Sequence[str]) -> List[Dict[int, Any]]:
    """Collect per-actor ``loaded_param_checksums`` for ``names`` on stage 0."""
    import ray

    return ray.get(
        [a.loaded_param_checksums.remote(names=list(names), stage_ids=[0]) for a in rollout_group.get_rollout_actors()]
    )


def _collect_trainer_expected(train_group, names: Sequence[str], prefix: str) -> Dict[str, str]:
    """Compute trainer-side expected ``{name: hex}`` for ``names``.

    Every DP rank participates in ``raw_state_dict``'s
    ``_to_full_tensor`` all-gather, so the four returned dicts agree on
    the TP-flat names the smoke probes. Take rank-0's view as the
    canonical expectation.
    """
    import ray

    per_rank = ray.get(
        [
            a.compute_local_param_checksums.remote(names=list(names), prefix=str(prefix))
            for a in train_group.get_actors()
        ]
    )
    if not per_rank:
        return {}
    return dict(per_rank[0])


def _value_mismatches(
    expected: Dict[str, str],
    rollout_post: List[Dict[int, Any]],
    stage_id: int = 0,
) -> Dict[int, List[str]]:
    """Per actor, return names whose rank-0 rollout hash != trainer expected."""
    mismatches: Dict[int, List[str]] = {}
    for idx, post_a in enumerate(rollout_post):
        ranks = post_a.get(stage_id) or []
        if isinstance(ranks, dict):
            ranks = [ranks]
        r0 = ranks[0] if ranks and isinstance(ranks[0], dict) else {}
        mismatches[idx] = sorted(name for name, hex_ in expected.items() if r0.get(name) and r0[name] != hex_)
    return mismatches


def _randomize_trainer(train_group, seed: int) -> None:
    import ray

    ray.get([a.randomize_weights_for_smoke.remote(seed=int(seed)) for a in train_group.get_actors()])


def _run_one_transport(
    *,
    name: str,
    train_group,
    rollout_group,
    placement,
    probe_names: Sequence[str],
    pre_seed: int,
    post_seed: int,
) -> Tuple[bool, str]:
    """Setup → randomize → sync → diff → teardown for one transport."""
    print(f"\n[B.{name}] === {name} ===")
    sync_cfg = _make_sync_cfg(name)
    train_group.setup_weight_sync(
        sync_cfg=sync_cfg,
        placement_cfg=placement.config,
        rollout_runtime=rollout_group,
        param_name_prefix=_SYNC_PREFIX,
    )
    try:
        # Seed-shift between pre and post so the bytes the rollout
        # workers receive must differ from what they're currently holding.
        _randomize_trainer(train_group, seed=pre_seed)
        train_group.sync_weights_to_rollout()
        pre = _collect_checksums(rollout_group, probe_names)

        _randomize_trainer(train_group, seed=post_seed)
        train_group.sync_weights_to_rollout()
        post = _collect_checksums(rollout_group, probe_names)

        # Trainer-side expected hash AFTER the post sync. Same
        # ``fingerprint_tensor`` as the rollout-side probe, so this is a
        # strict byte-equality check on top of the byte-change check.
        # ``raw_state_dict`` runs ``_to_full_tensor`` (DTensor all-gather)
        # so every DP rank participates in the collective; rank-0's
        # view is canonical for the TP-flat probe set.
        prefix = _SYNC_PREFIX
        expected = _collect_trainer_expected(train_group, probe_names, prefix)
        if len(expected) != len(probe_names):
            missing = sorted(set(probe_names) - set(expected.keys()))
            return False, (
                f"transport {name!r}: trainer-side expected hash missing "
                f"{len(missing)} of {len(probe_names)} probe names "
                f"(sample: {missing[:3]!r})"
            )

        # Diff each actor's pre/post checksums; require ≥1 change per actor.
        per_actor_changes: List[int] = []
        per_actor_mismatches = _value_mismatches(expected, post)
        for idx, (pre_a, post_a) in enumerate(zip(pre, post)):
            changed = _diff_checksums(pre_a, post_a)
            n = sum(len(v) for v in changed.values())
            per_actor_changes.append(n)
            mm = per_actor_mismatches.get(idx, [])
            print(
                f"      actor[{idx}] stage0: {n}/{len(probe_names)} names "
                f"changed between pre-sync and post-sync; "
                f"value-check: {len(mm)} mismatch(es)"
            )

        if not all(n > 0 for n in per_actor_changes):
            return False, f"transport {name!r} produced no checksum diff on at least one actor"
        total_mm = sum(len(v) for v in per_actor_mismatches.values())
        if total_mm:
            sample = next(iter(per_actor_mismatches.values()))[:3]
            return False, (
                f"transport {name!r} delivered DIFFERENT bytes than the "
                f"trainer shipped: {total_mm} mismatch(es) across "
                f"{len(per_actor_mismatches)} actor(s) (e.g. {sample!r})"
            )
        return True, (
            f"{sum(per_actor_changes)} total checksum diffs across "
            f"{len(pre)} actor(s); 0 value mismatches (byte-equality verified)"
        )
    finally:
        train_group.teardown_weight_sync()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--mode",
        choices=("disjoint", "colocate"),
        default="disjoint",
        help=(
            "Placement mode. 'disjoint' (default) puts trainer on GPUs 4..7 "
            "and rollout on GPUs 0..3 — verifies nccl_broadcast. 'colocate' "
            "puts both on GPUs 0..3 (time-share) — verifies tensor_payload "
            "+ ipc_bucketed (CUDA IPC needs shared physical GPUs)."
        ),
    )
    parser.add_argument(
        "--skip",
        default="",
        help="Comma-list of transports to skip: nccl,tensor,ipc",
    )
    parser.add_argument(
        "--probe-limit",
        type=int,
        default=16,
        help="Cap on synthetic-target params per transport.",
    )
    args = parser.parse_args()

    skips = {s.strip().lower() for s in args.skip.split(",") if s.strip()}
    skip_map = {"nccl": "nccl_broadcast", "tensor": "tensor_payload", "ipc": "ipc_bucketed"}
    skip_targets = {skip_map[s] for s in skips if s in skip_map}
    targets = [t for t in _TRANSPORTS_BY_MODE[args.mode] if t not in skip_targets]
    if not targets:
        print("[smoke] All transports skipped — nothing to do.")
        return 0
    print(f"[smoke] mode={args.mode!r}; transports={targets}")

    print("[smoke] Phase 0: imports + Hydra compose ...")
    cfg = _build_cfg(args.mode)
    print("[smoke] cfg loaded — rollout/engine modality =", cfg.rollout.engine.get("modality"))

    print("[smoke] Phase 1: Ray init + placement ...")
    import ray

    from diffusionrl.config.instantiate import materialize
    from diffusionrl.ray.placement import Placement

    if not ray.is_initialized():
        # Local cluster: 8 GPUs for disjoint (4 rollout + 4 train);
        # 4 GPUs sufficient for colocate (shared between rollout + train).
        num_gpus = 8 if args.mode == "disjoint" else 4
        ray.init(num_gpus=num_gpus, ignore_reinit_error=True)

    placement_cfg = materialize(cfg.placement)
    placement = Placement.from_config(placement_cfg)
    print(
        "[smoke] placement: %d rollout actors, %d train actors (colocate=%s)"
        % (placement_cfg.num_rollout_actors, placement_cfg.num_train_actors, placement_cfg.colocate)
    )

    train_group = None
    rollout_group = None
    summary: List[Tuple[str, bool, str]] = []
    try:
        print("[smoke] Phase 2: NewRolloutActorGroup + NewTrainActorGroup ...")
        from diffusionrl.ray.group.new_rollout import NewRolloutActorGroup
        from diffusionrl.ray.group.new_train import NewTrainActorGroup

        rollout_group = NewRolloutActorGroup(cfg=cfg, placement=placement)
        print("[smoke] rollout_group ready — %d actor handle(s)" % rollout_group.num_actors)

        train_group = NewTrainActorGroup(cfg=cfg, placement=placement)
        print("[smoke] train_group ready — %d actor handle(s)" % train_group.num_actors)

        print("[smoke] Phase 3: probe worker param names ...")
        probe_names = _probe_param_names(rollout_group, limit=int(args.probe_limit))
        if not probe_names:
            print("[smoke] FAIL: could not probe any TP-flat param names from rollout workers.")
            return 2
        print("[smoke] probed %d TP-flat names; sample: %s" % (len(probe_names), probe_names[:3]))

        print("[smoke] Phase 4-6: per-transport setup → randomize → sync → diff ...")
        # Per-transport seed pair: shift the post seed so each transport
        # forces fresh bytes (otherwise a re-run with the same seed would
        # land identical values and the diff would be a no-op).
        for i, target in enumerate(targets):
            ok, msg = _run_one_transport(
                name=target,
                train_group=train_group,
                rollout_group=rollout_group,
                placement=placement,
                probe_names=probe_names,
                pre_seed=100 + 2 * i,
                post_seed=100 + 2 * i + 1,
            )
            summary.append((target, ok, msg))

    finally:
        print("\n[smoke] Phase 7: teardown ...")
        try:
            if train_group is not None:
                train_group.dispose()
        except Exception as e:
            print(f"[smoke] train_group.dispose raised: {e!r}")
        try:
            if rollout_group is not None:
                rollout_group.dispose()
        except Exception as e:
            print(f"[smoke] rollout_group.dispose raised: {e!r}")
        try:
            placement.destroy()
        except Exception as e:
            print(f"[smoke] placement.destroy raised: {e!r}")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    failed: List[str] = []
    for name, ok, msg in summary:
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {name:18s}  {msg}")
        if not ok:
            failed.append(name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
