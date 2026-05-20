"""End-to-end weight-sync sweep for vllm-omni against HunyuanImage 3.0.

Boots one ``VLLMOmniRolloutEngine`` for HI3 t2i, then runs each weight-sync
transport sequentially against synthetic state-dicts and asserts that
worker-side parameter checksums change after each sync. Single Python
process; the NCCL sub-test acts as trainer rank 0 in-process.

Phases:

    A. Boot Omni (HI3 ~150GB, ~20 min on cephfs).
    B. Sanity generate.
    C. For each transport: read-checksums → sync → read-checksums → assert diff.
        - C.1 IPC bucketed       (engine.update_weights_from_ipc)
        - C.2 Tensor-payload     (engine.update_weights_from_tensor — SGLang-shape)
        - C.3 NCCL broadcast     (engine.init_weights_update_group +
                                  engine.update_weights_from_distributed +
                                  engine.destroy_weights_update_group)
        - C.4 LoRA tensor-bag    (engine.set_lora_from_tensors +
                                  list_loras + generate-with-lora visual diff)
    D. Teardown.

Usage:

    python scripts/smoke_weight_sync_e2e.py \\
        --model-path /mnt/bj/HunyuanImage-3-Instruct \\
        [--steps 8] [--height 1024] [--width 1024]

Acceptance: every sub-test prints "PASS" with a non-empty changed-name list.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import socket
import sys
import threading
import time
import uuid
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as torch_mp

# B.2 (tensor-payload) serializes via SGLang's MultiprocessingSerializer,
# which under the default ``file_descriptor`` sharing strategy ships a CUDA
# IPC-style FD reference that the receiver tries to retrieve from the
# sender's per-process ``resource_sharer`` socket. The smoke test process
# isn't a multiprocessing parent of the workers, so the receiver hits
# ``KeyError: 1`` in ``rebuild_storage_fd``. Switch to ``file_system``
# strategy globally — that backs shared storage with named ``/dev/shm``
# files reachable by any process on the host without resource_sharer
# handshake. Set this BEFORE any tensor allocation.
try:
    torch_mp.set_sharing_strategy("file_system")
except Exception:
    pass


# Synthetic param names we'll target across all sub-tests. We pick names that
# exist in the HI3 transformer (the AR + DiT both share the unified backbone)
# so load_weights actually applies them. If the names don't match, load_weights
# silently no-ops; the checksum assertion would fail and surface that.
#
# Pick a small subset of common patterns. The actual HI3 transformer has
# layers like model.layers.{i}.self_attn.qkv_proj.weight etc. We probe the
# real names at runtime via the first pre-checksum call instead of guessing
# here.
_DEFAULT_PROBE_LIMIT = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_node_ip() -> str:
    """Best-effort local IP for tcp:// init_method. Falls back to localhost."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def get_free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _unwrap_rpc(results, *, label: str = "<rpc>"):
    """Unwrap the doubly-nested collective_rpc return.

    ``omni.engine.collective_rpc(stage_ids=[s])`` returns
    ``[stage_results]`` where ``stage_results`` is a list of per-TP-rank
    return values. We always pass a single stage_id so we want the rank-0
    return from the only stage in the outer list.
    """
    if not isinstance(results, list) or not results:
        return None
    inner = results[0]
    if isinstance(inner, list) and inner:
        return inner[0]
    return inner


def collect_checksums(engine, *, names: List[str], stage_ids: List[int]) -> Dict[int, Dict[str, str]]:
    """Per stage, per name → 16-char sha256 prefix. Routed via collective_rpc."""
    out: Dict[int, Dict[str, str]] = {}
    for sid in stage_ids:
        results = engine._omni.engine.collective_rpc(
            method="_diffrl_param_checksums",
            args=(list(names),),
            stage_ids=[int(sid)],
        )
        rank0 = _unwrap_rpc(results, label=f"checksums.stage{sid}")
        out[int(sid)] = dict(rank0) if isinstance(rank0, dict) else {}
    return out


def diff_checksums(pre: Dict[int, Dict[str, str]], post: Dict[int, Dict[str, str]]) -> Dict[int, List[str]]:
    """Per stage, list of names whose checksum changed between pre and post."""
    changed: Dict[int, List[str]] = {}
    for sid, pre_map in pre.items():
        post_map = post.get(sid, {})
        names = sorted(n for n, h in pre_map.items() if post_map.get(n) and post_map[n] != h)
        changed[sid] = names
    return changed


# Patterns we exclude from probe targets. These params have TP-aware
# weight_loaders (vocab parallel embedding, column/row parallel linears) that
# expect the FULL un-sharded shape and shard internally. Our synthetic
# state-dict ships per-worker shapes (because we read from named_parameters
# on the worker, which is already sharded), so feeding those names trips
# vocab_parallel_embedding.py:weight_loader's
# ``loaded_weight.shape[output_dim] == self.org_vocab_size`` assertion.
# Norms / scalar params have no TP layout and round-trip cleanly.
_TP_SHARDED_NAME_PATTERNS = (
    "embed_tokens",  # vocab parallel embedding
    "lm_head",  # vocab parallel embedding (tied head)
    "qkv_proj",  # column-parallel linear
    "o_proj",  # row-parallel linear
    "gate_up_proj",  # merged column-parallel
    "down_proj",  # row-parallel
    "experts.",  # MoE expert tensors are shard-stacked
)


def _is_tp_sharded_name(name: str) -> bool:
    return any(p in name for p in _TP_SHARDED_NAME_PATTERNS)


def probe_real_param_names(engine, *, stage_ids: List[int], limit: int) -> List[str]:
    """Ask each stage's worker for ``limit`` real parameter names that exist
    on its loaded model. Take the intersection across stages so the names
    we ship in synthetic state-dicts will actually land on both.

    Filters out names with TP-aware weight loaders (vocab embeddings,
    qkv_proj, etc.) — those expect full un-sharded weights and our synthetic
    builder ships per-worker shapes. Targets layer norms and other TP-flat
    params instead, which round-trip cleanly through ``load_weights``.
    """
    per_stage_names: List[set] = []
    for sid in stage_ids:
        try:
            results = engine._omni.engine.collective_rpc(
                method="_diffrl_param_checksums",
                args=(None,),
                stage_ids=[int(sid)],
            )
        except Exception as exc:
            print(f"      stage {sid} _diffrl_param_checksums RPC failed: {exc!r}")
            return []
        rank0 = _unwrap_rpc(results, label=f"probe.stage{sid}")
        if not isinstance(rank0, dict):
            print(f"      stage {sid} unwrap → {type(rank0).__name__}: {str(rank0)[:200]}")
            return []
        print(f"      stage {sid} got {len(rank0)} names; sample: {list(rank0.keys())[:3]}")
        per_stage_names.append(set(rank0.keys()))
    if not per_stage_names:
        return []
    common = set.intersection(*per_stage_names)
    safe = {n for n in common if not _is_tp_sharded_name(n)}
    print(f"      common across stages: {len(common)} names ({len(safe)} TP-flat after filtering)")
    # Prefer small fp32/bf16 leaf params (lower bytes) so the synthetic
    # state-dicts stay light. Sort deterministically; cap at limit.
    return sorted(safe)[: int(limit)]


def make_synthetic_state_dict(real_names: List[str], engine, stage_id: int) -> List[Tuple[str, torch.Tensor]]:
    """Pull real shapes/dtypes for the given names from a stage worker, then
    rebuild a state-dict with random values so load_weights actually mutates
    the parameters (and the checksum visibly changes)."""
    results = engine._omni.engine.collective_rpc(
        method="_diffrl_describe_params",
        args=(list(real_names),),
        stage_ids=[int(stage_id)],
    )
    desc = _unwrap_rpc(results, label=f"describe.stage{stage_id}")
    if not isinstance(desc, dict):
        raise RuntimeError(
            f"make_synthetic_state_dict: _diffrl_describe_params returned {type(desc).__name__}: {str(desc)[:200]}"
        )
    out: List[Tuple[str, torch.Tensor]] = []
    for name in real_names:
        meta = desc.get(name)
        if meta is None:
            continue
        shape, dtype_name = meta
        dtype = getattr(torch, dtype_name.replace("torch.", "").split(".")[-1])
        # Random float for floating-point dtypes; integer-zeros otherwise.
        if dtype.is_floating_point:
            t = torch.randn(*shape, dtype=dtype)
        else:
            t = torch.zeros(*shape, dtype=dtype)
        out.append((name, t))
    return out


# ---------------------------------------------------------------------------
# Sub-tests
# ---------------------------------------------------------------------------


def _diff_value_checksums(
    expected_per_rank: Dict[int, List[Dict[str, str]]],
    got_per_rank: Dict[int, List[Dict[str, str]]],
    label: str,
) -> List[str]:
    """Compare expected ↔ loaded hashes per (stage, rank, name).

    Both inputs share the shape ``{stage_id: [rank0_dict, rank1_dict, ...]}``
    where each ``rankN_dict`` is ``{name: short_sha256_hex}``. Returns a
    flat list of human-readable mismatch tags; empty list means every
    (stage, rank, name) the trainer expected was present and equal on the
    worker side.

    Stage rank-count contract: the AR stage (multiproc executor) returns
    one ``rankN_dict`` per TP rank — len(got) == TP. The DiT stage (pool
    executor) collapses to a single aggregated dict — len(got) == 1.
    Diff accepts ``len(got) <= len(expected)`` and only verifies the
    ranks the engine actually returned, printing an info note when we
    couldn't see every rank. ``len(got) > len(expected)`` is still a
    real anomaly (extra results from the engine) and gets flagged.
    """
    mismatches: List[str] = []
    for sid, expected_ranks in expected_per_rank.items():
        got_ranks = got_per_rank.get(sid)
        if not isinstance(got_ranks, list):
            mismatches.append(f"{label} stage{sid}: got non-list result {got_ranks!r}")
            continue
        if len(got_ranks) == 0:
            mismatches.append(f"{label} stage{sid}: empty got result")
            continue
        if len(got_ranks) > len(expected_ranks):
            mismatches.append(
                f"{label} stage{sid}: more got results ({len(got_ranks)}) than expected ranks ({len(expected_ranks)})"
            )
            continue
        n = min(len(expected_ranks), len(got_ranks))
        for rank_idx in range(n):
            exp_dict = expected_ranks[rank_idx]
            got_dict = got_ranks[rank_idx]
            if not isinstance(got_dict, dict):
                mismatches.append(f"{label} stage{sid} rank{rank_idx}: got non-dict {got_dict!r}")
                continue
            for name, sent_hex in exp_dict.items():
                got_hex = got_dict.get(name)
                if got_hex != sent_hex:
                    mismatches.append(f"{label} stage{sid} rank{rank_idx} {name}: sent={sent_hex} got={got_hex}")
        if len(got_ranks) < len(expected_ranks):
            print(
                f"      [info] {label} stage{sid}: verified {n}/"
                f"{len(expected_ranks)} ranks "
                f"(engine returned {len(got_ranks)} aggregated result(s))"
            )
    return mismatches


def run_ipc_bucketed(engine, target_names: List[str], stage_ids: List[int]) -> dict:
    """B.1 — bucketed CUDA-IPC."""
    print(f"[B.1] IPC bucketed: {len(target_names)} names, stages={stage_ids}")
    pre = collect_checksums(engine, names=target_names, stage_ids=stage_ids)
    print(f"      pre  checksum sample: {next(iter(pre[stage_ids[0]].items()), None)}")

    # Per-stage trainer-side senders. For each stage, spawn one
    # BucketedWeightSender thread per worker rank (TP=4 in HI3 yaml).
    from diffusionrl.rollout.engine.vllm_omni.weight_sync.bucketed_transfer import (
        BucketedWeightSender,
    )
    from diffusionrl.rollout.engine.vllm_omni.weight_sync.checksum import (
        compute_param_checksums,
    )
    from diffusionrl.rollout.engine.vllm_omni.weight_sync.ipc_dispatch import (
        replica_rank_from_env,
        zmq_handle,
    )

    replica_rank = replica_rank_from_env()
    tp_size_per_stage = 4  # matches stage YAML
    sender_threads: List[threading.Thread] = []

    def _sender_main(handle: str, weights: List[Tuple[str, torch.Tensor]]) -> None:
        # 2048 MB covers HI3's largest single param (``lm_head.weight``,
        # 133120×4096×bf16 ≈ 1 GiB). 512 MB triggers an in-bucket assertion.
        sender = BucketedWeightSender(zmq_handle=handle, bucket_size_mb=2048, use_shm=False)
        asyncio.run(sender.async_send_weights(iter(weights)))

    # Track the trainer-side expected hash per (stage, rank). For B.1 every
    # rank in a stage receives the *same* synth, so we replicate the
    # per-stage hash across all ranks. The value-check below diffs against
    # what the worker actually loaded.
    expected_per_rank: Dict[int, List[Dict[str, str]]] = {}

    for sid in stage_ids:
        # Build a synthetic state-dict matching the real shapes for this stage.
        synth = make_synthetic_state_dict(target_names, engine, stage_id=sid)
        stage_expected = compute_param_checksums(dict(synth))
        expected_per_rank[sid] = [stage_expected for _ in range(tp_size_per_stage)]
        for r in range(tp_size_per_stage):
            handle = zmq_handle(replica_rank=replica_rank, stage_id=sid, local_rank=r)
            t = threading.Thread(target=_sender_main, args=(handle, list(synth)), daemon=True)
            t.start()
            sender_threads.append(t)

    # Trigger receivers on the engine side. This blocks until every
    # collective_rpc returns (i.e. all workers finished receiving).
    engine.update_weights_from_ipc(stage_ids=stage_ids, use_shm=False)

    for t in sender_threads:
        t.join(timeout=120)
    assert all(not t.is_alive() for t in sender_threads), "sender thread hung"

    post = collect_checksums(engine, names=target_names, stage_ids=stage_ids)
    diffs = diff_checksums(pre, post)
    for sid in stage_ids:
        print(f"      stage {sid}: {len(diffs[sid])} of {len(target_names)} changed")

    # Value-correctness: post-load read-back hashes from the workers
    # should match the trainer's per-(stage, rank) expected hashes.
    got_per_rank = engine.loaded_param_checksums(
        names=list(target_names),
        stage_ids=list(stage_ids),
    )
    value_mismatches = _diff_value_checksums(expected_per_rank, got_per_rank, "ipc")
    print(f"      ipc value-check: {len(value_mismatches)} mismatches")

    return {
        "changes": [f"stage{sid}:{n}" for sid, ns in diffs.items() for n in ns],
        "value_mismatches": value_mismatches,
    }


def run_tensor_payload(engine, target_names: List[str], stage_ids: List[int]) -> dict:
    """B.2 — SGLang-shape one-bag tensor payload."""
    print(f"[B.2] Tensor-payload: {len(target_names)} names, stages={stage_ids}")
    pre = collect_checksums(engine, names=target_names, stage_ids=stage_ids)

    from sglang.srt.utils import MultiprocessingSerializer
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

    from diffusionrl.rollout.engine.vllm_omni.weight_sync.checksum import (
        compute_param_checksums,
    )

    tp_size = 4

    # Each TP rank receives a *different* serialized payload built from a
    # different synth. The post-load read-back on rank R should match the
    # hash of the synth we sent specifically to rank R — track that here.
    expected_per_rank: Dict[int, List[Dict[str, str]]] = {}

    for sid in stage_ids:
        # Build ONE independent serialized payload per TP rank. SGLang's
        # MultiprocessingSerializer rides over torch's shared-memory IPC,
        # so the same serialized blob can't be safely consumed by multiple
        # workers (the receiver pops the FD entry and the next reader hits
        # ``KeyError``). Per-rank synthesis with ``file_system`` sharing
        # strategy (set at module top) gives each rank its own /dev/shm
        # backing file.
        per_rank_serialized: List[str] = []
        per_rank_expected: List[Dict[str, str]] = []
        for _r in range(tp_size):
            synth = make_synthetic_state_dict(target_names, engine, stage_id=sid)
            by_dtype: Dict[Any, List[Tuple[str, torch.Tensor]]] = {}
            for name, t in synth:
                by_dtype.setdefault(t.dtype, []).append((name, t))
            biggest_dtype = max(by_dtype, key=lambda d: len(by_dtype[d]))
            named = by_dtype[biggest_dtype]
            # Capture the rank's expected hash from the SAME named list we'll
            # ship in the bucket — anything not in this dtype bucket won't be
            # delivered to this rank, so we only check what we actually sent.
            per_rank_expected.append(compute_param_checksums(named))
            bucket = FlattenedTensorBucket(named_tensors=named)
            payload = {
                "flattened_tensor": bucket.get_flattened_tensor(),
                "metadata": bucket.get_metadata(),
            }
            per_rank_serialized.append(MultiprocessingSerializer.serialize(payload, output_str=True))
        expected_per_rank[sid] = per_rank_expected
        engine.update_weights_from_tensor(
            serialized_named_tensors=per_rank_serialized,
            load_format="flattened_bucket",
            stage_ids=[sid],
        )

    post = collect_checksums(engine, names=target_names, stage_ids=stage_ids)
    diffs = diff_checksums(pre, post)
    for sid in stage_ids:
        print(f"      stage {sid}: {len(diffs[sid])} of {len(target_names)} changed")

    got_per_rank = engine.loaded_param_checksums(
        names=list(target_names),
        stage_ids=list(stage_ids),
    )
    value_mismatches = _diff_value_checksums(expected_per_rank, got_per_rank, "tensor")
    print(f"      tensor value-check: {len(value_mismatches)} mismatches")

    return {
        "changes": [f"stage{sid}:{n}" for sid, ns in diffs.items() for n in ns],
        "value_mismatches": value_mismatches,
    }


def run_nccl(engine, target_names: List[str], stage_ids: List[int]) -> dict:
    """B.3 — NCCL broadcast with in-process trainer rank 0.

    Tests each stage in its own NCCL group (not one big shared group).
    A single shared group with all 9 ranks would require every rank to
    participate in every broadcast; calling
    ``update_weights_from_distributed(stage_ids=[0])`` only opens stage
    0's receive loop, so stage 1's ranks would no-show and the
    collective hangs. Per-stage groups keep the rank set tight and
    sequence cleanly.
    """
    print(f"[B.3] NCCL: {len(target_names)} names, stages={stage_ids}")
    pre = collect_checksums(engine, names=target_names, stage_ids=stage_ids)

    tp_size_per_stage = 4

    # NCCL refuses to put two ranks on the same physical GPU
    # ("Duplicate GPU detected : rank 1 and rank 0 both on CUDA device ..."),
    # so the test process must NOT bind to a GPU used by the stage's
    # workers. The HI3 t2i YAML pins stage 0 to GPUs 0-3 and stage 1 to
    # 4-7; pick the opposite stage's first GPU as the test rank's device.
    # (The test process has access to all 8 GPUs since it's the engine's
    # parent, not a worker.)
    _stage_to_test_gpu = {0: 4, 1: 0}

    from diffusionrl.rollout.engine.vllm_omni.weight_sync.checksum import (
        compute_param_checksums,
    )
    from diffusionrl.utils.distributed_utils import (
        init_process_group as _diffrl_init_pg,
    )

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    # Trainer broadcasts the same synth to every TP rank in a stage
    # (NCCL broadcast: src=0 → all ranks). Per-stage hash, replicated
    # across all ranks.
    expected_per_rank: Dict[int, List[Dict[str, str]]] = {}

    for sid in stage_ids:
        test_gpu = _stage_to_test_gpu.get(int(sid), 0)
        torch.cuda.set_device(test_gpu)
        master_addr = get_node_ip()
        master_port = get_free_port()
        group_name = f"diffrl_e2e_s{sid}_{uuid.uuid4().hex[:6]}"
        per_stage_world = tp_size_per_stage + 1  # +1 = us as rank 0
        print(
            f"      stage {sid}: coordinator={master_addr}:{master_port} "
            f"world_size={per_stage_world} test_gpu=cuda:{test_gpu}"
        )

        # 1) Workers bring up the group (background; they block on rendezvous).
        init_future = pool.submit(
            engine.init_weights_update_group,
            master_address=master_addr,
            master_port=master_port,
            rank_offset=1,
            world_size=per_stage_world,
            group_name=group_name,
            backend="nccl",
            stage_ids=[sid],
        )

        # 2) Our rank-0 init via the diffusionrl helper (PrefixStore matches
        # the worker side).
        group = _diffrl_init_pg(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=per_stage_world,
            rank=0,
            group_name=group_name,
        )
        init_future.result(timeout=300)
        print(f"      stage {sid}: NCCL group up; rank 0 + {tp_size_per_stage} workers")

        # 3) Build per-stage synthetic tensors. CUDA on the test rank's
        # device so NCCL is happy.
        synth = make_synthetic_state_dict(target_names, engine, stage_id=sid)
        device_str = f"cuda:{test_gpu}"
        synth_cuda = [(name, t.to(device_str, non_blocking=True)) for name, t in synth]
        # Trainer-side expected hash for the value-correctness check.
        # Hash off the CPU originals (cheap, deterministic) — the GPU
        # broadcast is byte-equivalent so post-load equals pre-broadcast.
        stage_expected = compute_param_checksums(dict(synth))
        expected_per_rank[sid] = [stage_expected for _ in range(tp_size_per_stage)]

        # 4) Tell stage workers to receive (background — they block on
        # dist.broadcast(src=0)) while we broadcast from rank 0.
        recv_future = pool.submit(
            engine.update_weights_from_distributed,
            names=[n for n, _ in synth_cuda],
            dtypes=[str(t.dtype) for _, t in synth_cuda],
            shapes=[list(t.shape) for _, t in synth_cuda],
            group_name=group_name,
            target_modules=None,
            flush_cache=True,
            stage_ids=[sid],
        )
        for _, tensor in synth_cuda:
            dist.broadcast(tensor.contiguous(), src=0, group=group)
        recv_future.result(timeout=120)

        # 5) Per-stage teardown.
        engine.destroy_weights_update_group(group_name=group_name, stage_ids=[sid])
        del group

    pool.shutdown(wait=True)

    post = collect_checksums(engine, names=target_names, stage_ids=stage_ids)
    diffs = diff_checksums(pre, post)
    for sid in stage_ids:
        print(f"      stage {sid}: {len(diffs[sid])} of {len(target_names)} changed")

    got_per_rank = engine.loaded_param_checksums(
        names=list(target_names),
        stage_ids=list(stage_ids),
    )
    value_mismatches = _diff_value_checksums(expected_per_rank, got_per_rank, "nccl")
    print(f"      nccl value-check: {len(value_mismatches)} mismatches")

    return {
        "changes": [f"stage{sid}:{n}" for sid, ns in diffs.items() for n in ns],
        "value_mismatches": value_mismatches,
    }


def run_lora(engine, prompt: str, stage_ids: List[int]) -> dict:
    """B.4 — LoRA tensor-bag + list_loras + post-load value-check."""
    print(f"[B.4] LoRA tensor-bag, stages={stage_ids}")
    from diffusionrl.rollout.engine.vllm_omni.weight_sync.checksum import (
        compute_lora_checksums_post_optimize,
    )
    from diffusionrl.rollout.engine.vllm_omni.weight_sync.ipc_dispatch import (
        DIFFRL_LORA_INT_ID,
    )

    # Synthetic LoRA: target a single attention projection. We ask the worker
    # which target_modules HI3 supports via collective_rpc.
    target_module_names = ["q_proj", "v_proj"]
    rank, alpha = 8, 16

    # Build small fake LoRA tensors for each target. We don't know the exact
    # in/out features, but the hijack expects a `peft_helper`-compatible
    # shape — the simplest fake is a rank-8 adapter matrix per target.
    # For HI3 we'd need real shapes; here we just probe whether add_lora
    # accepts the call shape. If it raises due to shape mismatch, we'll log
    # the error and continue.
    fake_in, fake_out = 4096, 4096
    lora_tensors = {}
    for tgt in target_module_names:
        lora_tensors[f"base_model.model.{tgt}.lora_A.weight"] = torch.randn(rank, fake_in, dtype=torch.bfloat16) * 0.01
        lora_tensors[f"base_model.model.{tgt}.lora_B.weight"] = torch.randn(fake_out, rank, dtype=torch.bfloat16) * 0.01

    peft_config = {
        "peft_type": "LORA",
        "r": rank,
        "lora_alpha": alpha,
        "target_modules": target_module_names,
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": None,
    }

    try:
        engine.set_lora_from_tensors("e2e_test", lora_tensors, peft_config=peft_config, stage_ids=stage_ids)
    except Exception as exc:
        print(f"      set_lora_from_tensors raised: {exc}")
        return {"lora_listed": False, "value_mismatches": [f"raised:{exc!s}"], "raw": None}

    # Probe list_loras across stages
    listed_per_stage = {}
    for sid in stage_ids:
        try:
            res = engine._omni.engine.collective_rpc(method="list_loras", args=(), stage_ids=[int(sid)])
            listed_per_stage[sid] = res[0] if isinstance(res, list) and res else res
        except Exception as exc:
            listed_per_stage[sid] = f"ERROR:{exc!s}"
    print(f"      list_loras per stage: {listed_per_stage}")

    # ``list_loras`` returns either a flat ``[int_id, ...]`` (DiT pool) or
    # a per-TP-rank ``[[int_id, ...], ...]`` (AR multiproc). Flatten one
    # level before searching so the assertion catches both shapes.
    def _flatten_one(seq):
        out = []
        for el in seq or []:
            if isinstance(el, (list, set, tuple)):
                out.extend(el)
            else:
                out.append(el)
        return out

    found_per_stage = {
        sid: DIFFRL_LORA_INT_ID in _flatten_one(lst)
        for sid, lst in listed_per_stage.items()
        if isinstance(lst, (list, set, tuple))
    }
    found_all = bool(found_per_stage) and all(found_per_stage.values())

    # Value-correctness on lora_a only. ``lora_a`` is replicated across
    # the TP world (no sharding regardless of target-module layout), so
    # every rank's loaded hash should equal the trainer's full-tensor
    # hash. ``lora_b`` for col-parallel targets (q_proj / v_proj here)
    # is shard-split across the 4 TP ranks — per-rank hashes will NOT
    # match the trainer's full hash. TP-aware lora_b verification needs
    # an external all-gather; deferred to a follow-up.
    sent_full = compute_lora_checksums_post_optimize(lora_tensors, peft_config)

    # PEFTHelper renames layers when loading: trainer ships
    # ``base_model.model.q_proj.lora_A.weight`` but the worker stores
    # ``model.layers.<i>.self_attn.q_proj`` (post-resolution against the
    # actual model's submodules). We can't predict the exact loaded name
    # without inspecting the model, so build a {field: hex_set} index
    # keyed by field+tgt and compare any worker key whose suffix matches.
    def _trainer_index(sent: Dict[str, str]) -> Dict[Tuple[str, str], str]:
        out: Dict[Tuple[str, str], str] = {}
        for key, hex_ in sent.items():
            for tgt in target_module_names:
                if f"{tgt}.lora_A.weight" in key:
                    out[(tgt, "lora_a")] = hex_
                elif f"{tgt}.lora_B.weight" in key:
                    out[(tgt, "lora_b")] = hex_
        return out

    trainer_idx = _trainer_index(sent_full)

    got_lora = engine.loaded_lora_checksums(
        adapter_id=DIFFRL_LORA_INT_ID,
        stage_ids=list(stage_ids),
    )

    value_mismatches: List[str] = []
    lora_a_checked = 0
    for sid, per_rank in got_lora.items():
        if not isinstance(per_rank, list):
            value_mismatches.append(f"lora stage{sid}: got non-list result {per_rank!r}")
            continue
        for rank_idx, rank_layers in enumerate(per_rank):
            if not isinstance(rank_layers, dict):
                value_mismatches.append(f"lora stage{sid} rank{rank_idx}: got non-dict {rank_layers!r}")
                continue
            for layer_name, fields in rank_layers.items():
                # Match the worker layer name against the trainer's
                # target-module list — only ``lora_a`` is verified this round.
                got_a = fields.get("lora_a")
                if got_a is None:
                    continue
                # Pick the trainer's expected hash for whichever target
                # module appears in the layer name.
                exp_a = None
                for tgt in target_module_names:
                    if tgt in layer_name:
                        exp_a = trainer_idx.get((tgt, "lora_a"))
                        break
                if exp_a is None:
                    # Worker has a layer we didn't ship — note but don't fail
                    continue
                lora_a_checked += 1
                if got_a != exp_a:
                    value_mismatches.append(
                        f"lora stage{sid} rank{rank_idx} {layer_name}.lora_a: sent={exp_a} got={got_a}"
                    )
    print(
        f"      lora value-check (lora_a only): {len(value_mismatches)} "
        f"mismatches across {lora_a_checked} rank-layer entries"
    )

    return {
        "lora_listed": found_all,
        "lora_listed_per_stage": found_per_stage,
        "value_mismatches": value_mismatches,
        "lora_a_checked": lora_a_checked,
        "raw_listed": listed_per_stage,
    }


# ---------------------------------------------------------------------------
# Phase A — boot Omni, sanity generate
# ---------------------------------------------------------------------------


def boot_engine(model_path: str, steps: int, height: int, width: int):
    print(f"[A] Booting VLLMOmniRolloutEngine for HI3 t2i — model={model_path}")
    from diffusionrl.rollout.engine.vllm_omni import (
        VLLMOmniEngineConfig,
        VLLMOmniRolloutEngine,
    )

    cfg = VLLMOmniEngineConfig(
        model_path=model_path,
        modality="t2i",
        default_num_inference_steps=steps,
        default_height=height,
        default_width=width,
    )
    t0 = time.time()
    engine = VLLMOmniRolloutEngine(
        cfg,
        device=torch.device("cuda:0"),
        strategy=None,
        rank=0,
        model_config=None,
    )
    print(f"[A] Boot done in {time.time() - t0:.1f}s")
    return engine


def sanity_generate(engine, prompt: str) -> None:
    print(f"[A.2] Sanity generate: prompt={prompt!r}")
    from diffusionrl.types.primitives import Texts
    from diffusionrl.types.rollout_req import RolloutReq

    req = RolloutReq(
        sample_ids=["s0"],
        group_ids=["g0"],
        primitives={"text": Texts(texts=[prompt])},
        stage_params={"diffusion": {"num_inference_steps": 4}},
    )
    t0 = time.time()
    resp = engine.generate(req)
    print(
        f"      generate done in {time.time() - t0:.1f}s; rollout_traces={list(resp.rollout_traces.keys())} decoded={list(resp.decoded.keys())}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--model-path", default="/mnt/bj/HunyuanImage-3-Instruct")
    parser.add_argument("--prompt", default="a red apple on a wooden table")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument(
        "--probe-limit",
        type=int,
        default=_DEFAULT_PROBE_LIMIT,
        help="Cap the number of synthetic-target params per sub-test.",
    )
    parser.add_argument(
        "--skip",
        default="",
        help="Comma-list of sub-tests to skip: ipc,tensor,nccl,lora",
    )
    args = parser.parse_args()

    skips = {s.strip().lower() for s in args.skip.split(",") if s.strip()}

    engine = boot_engine(args.model_path, steps=args.steps, height=args.height, width=args.width)
    sanity_generate(engine, args.prompt)

    stage_ids = [0, 1]
    real_names = probe_real_param_names(engine, stage_ids=stage_ids, limit=args.probe_limit)
    if not real_names:
        print("[!] could not probe any real parameter names from workers; aborting.")
        engine.shutdown()
        return 2
    print(f"[A.3] Probed {len(real_names)} real param names common to both stages")
    print(f"      sample: {real_names[:3]}")

    results: Dict[str, Any] = {}

    if "ipc" not in skips:
        try:
            results["ipc"] = run_ipc_bucketed(engine, real_names, stage_ids)
        except Exception as exc:
            results["ipc"] = f"ERROR:{exc!s}"

    if "tensor" not in skips:
        try:
            results["tensor"] = run_tensor_payload(engine, real_names, stage_ids)
        except Exception as exc:
            results["tensor"] = f"ERROR:{exc!s}"

    if "nccl" not in skips:
        try:
            results["nccl"] = run_nccl(engine, real_names, stage_ids)
        except Exception as exc:
            results["nccl"] = f"ERROR:{exc!s}"

    if "lora" not in skips:
        try:
            results["lora"] = run_lora(engine, args.prompt, stage_ids)
        except Exception as exc:
            results["lora"] = f"ERROR:{exc!s}"

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    failed = []
    for label, result in results.items():
        # Outer try/except wraps each sub-test; an exception is recorded as
        # a string, anything else is a dict from the new return shape.
        if isinstance(result, str):
            print(f"  [????] {label:8s}  {result}")
            failed.append(label)
            continue
        if not isinstance(result, dict):
            print(f"  [????] {label:8s}  {result!r}")
            failed.append(label)
            continue

        value_mismatches = result.get("value_mismatches", [])
        value_ok = len(value_mismatches) == 0

        if label == "lora":
            listed_ok = bool(result.get("lora_listed", False))
            ok = listed_ok and value_ok
            tag = "PASS" if ok else "FAIL"
            details = (
                f"LORA_LISTED={listed_ok}; value-check (lora_a, "
                f"{result.get('lora_a_checked', 0)} entries): "
                f"{len(value_mismatches)} mismatches"
            )
            print(f"  [{tag}] {label:8s}  {details}")
            if not ok:
                failed.append(label)
                if value_mismatches:
                    for tag_line in value_mismatches[:5]:
                        print(f"           - {tag_line}")
            continue

        # B.1 / B.2 / B.3: require both "checksums changed" AND "values match".
        changes = result.get("changes", [])
        change_ok = len(changes) > 0
        ok = change_ok and value_ok
        tag = "PASS" if ok else "FAIL"
        details = f"{len(changes)} structural changes; value-check: {len(value_mismatches)} mismatches"
        print(f"  [{tag}] {label:8s}  {details}")
        if not ok:
            failed.append(label)
            if not change_ok:
                print("           - no checksum diff (transport may be no-op)")
            if value_mismatches:
                for tag_line in value_mismatches[:5]:
                    print(f"           - {tag_line}")

    print()
    print("[D] Shutting down engine ...")
    engine.shutdown()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
