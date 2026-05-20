"""Per-sub-module GPU memory breakdown WITHIN each HI3 decoder layer.

Sibling of :mod:`scripts.profile_hi3_replay_memory`. That probe measured
per-layer activation cost. This probe drills down: it attaches forward
hooks on every sub-module of interest inside each ``HunyuanImage3DecoderLayer``
so we can answer "of the ~5 GB / layer saved under grad, how much is
attention vs MoE gate vs MoE experts vs layernorms?".

Hook targets per layer (paths under ``model.layers.<i>``):

  input_layernorm
  self_attn                         # whole block
    self_attn.qkv_proj
    self_attn.o_proj                # may have LoRA adapters
    self_attn.query_layernorm
    self_attn.key_layernorm
  post_attention_layernorm
  mlp                               # whole block (MoE; includes 64 experts)
    mlp.gate                        # router (produces combine_weights / dispatch_mask)
    mlp.shared_mlp                  # always-on dense MLP branch

Implicit metrics (computed in summary):
  attention_internals = self_attn_delta - qkv_proj_delta - o_proj_delta
                        - query_layernorm_delta - key_layernorm_delta
                       # = QKV split + scaled_dot_product_attention output
  moe_experts_contrib = mlp_delta - mlp.gate_delta - mlp.shared_mlp_delta
                       # = 64-expert FFN + combine work

Phases (same as the per-layer probe):
  A. rollout_no_grad  — pipe.generate(req) (diffuse under no_grad).
  B. A_replay_no_grad — explicit replay() under torch.no_grad().
  C. B_replay_grad    — replay() WITHOUT no_grad (the training path).
                        Likely OOMs after ~9 layers; sub-module data
                        for the layers that ran is the meat of the report.

Usage::

    cd ~/diffusionrl && source .venv/bin/activate && \\
    PYTORCH_ALLOC_CONF=expandable_segments:True \\
    torchrun --nproc-per-node=8 --master-port=29500 \\
        scripts/profile_hi3_layer_internals.py --ckpt /dockerdata/HunyuanImage-3
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from diffusionrl.models_new.hunyuan_image3.conditions import (
    HunyuanImage3DiffusionConditions,
)
from diffusionrl.models_new.hunyuan_image3.config import HunyuanImage3PipelineConfig
from diffusionrl.models_new.hunyuan_image3.diffusion import HunyuanImage3DiffusionParams
from diffusionrl.models_new.hunyuan_image3.pipeline import HunyuanImage3Pipeline
from diffusionrl.sde.kernels import FlowSDEStrategy
from diffusionrl.training_new.fsdp_policy import FSDPPolicyConfig
from diffusionrl.training_new.lora_policy import LoRAPolicyConfig
from diffusionrl.training_new.policy import Policy, compose_policy, walk_source_chain
from diffusionrl.types.primitives import Texts
from diffusionrl.types.rollout_req import RolloutReq

_GB = 1024.0**3

# Sub-modules to instrument INSIDE each decoder layer. Paths are relative
# to the layer; e.g. for layer 5 the qkv proj is "layers.5.self_attn.qkv_proj".
SUBMODULE_TARGETS: Tuple[str, ...] = (
    "input_layernorm",
    "self_attn",
    "self_attn.qkv_proj",
    "self_attn.o_proj",
    "self_attn.query_layernorm",
    "self_attn.key_layernorm",
    "post_attention_layernorm",
    "mlp",
    "mlp.gate",
    "mlp.shared_mlp",
)


def _r(rank: int, s: str) -> None:
    if rank == 0:
        print(s, flush=True)


@dataclass
class HookRecord:
    phase: str
    layer: int
    submodule: str  # e.g. "self_attn.qkv_proj"
    stage: str  # "pre" or "post"
    mem_bytes: int
    peak_bytes: int


class SubmoduleProbe:
    """Forward pre/post hooks on a fixed set of sub-modules per layer."""

    def __init__(self) -> None:
        self.phase: Optional[str] = None
        self.records: List[HookRecord] = []
        self._handles: List[Any] = []

    def attach(self, layers) -> Tuple[int, int]:
        """Attach hooks. Returns (num_layers_hooked, num_submodules_per_layer)."""
        hooked_count = 0
        for layer_idx, layer in enumerate(layers):
            for sub_path in SUBMODULE_TARGETS:
                target = _resolve_submodule(layer, sub_path)
                if target is None:
                    continue
                self._handles.append(target.register_forward_pre_hook(self._make_pre(layer_idx, sub_path)))
                self._handles.append(target.register_forward_hook(self._make_post(layer_idx, sub_path)))
                hooked_count += 1
        per_layer = hooked_count // max(1, len(layers))
        return len(layers), per_layer

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def set_phase(self, name: str) -> None:
        self.phase = name

    def _record(self, layer: int, sub: str, stage: str) -> None:
        if self.phase is None:
            return
        torch.cuda.synchronize()
        self.records.append(
            HookRecord(
                phase=self.phase,
                layer=layer,
                submodule=sub,
                stage=stage,
                mem_bytes=int(torch.cuda.memory_allocated()),
                peak_bytes=int(torch.cuda.max_memory_allocated()),
            )
        )

    def _make_pre(self, layer: int, sub: str):
        def hook(module, args):
            self._record(layer, sub, "pre")

        return hook

    def _make_post(self, layer: int, sub: str):
        def hook(module, args, output):
            self._record(layer, sub, "post")

        return hook


def _resolve_submodule(root, dotted: str) -> Optional[Any]:
    """Walk ``root.attr1.attr2....`` if all attrs exist; else None."""
    cur = root
    for part in dotted.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _phase_pairs(records: List[HookRecord], phase: str) -> Dict[Tuple[int, str], Tuple[int, int]]:
    """Return {(layer, submodule): (mem_pre, mem_post)} for a phase."""
    pre_map: Dict[Tuple[int, str], int] = {}
    post_map: Dict[Tuple[int, str], int] = {}
    for r in records:
        if r.phase != phase:
            continue
        key = (r.layer, r.submodule)
        if r.stage == "pre":
            pre_map[key] = r.mem_bytes
        else:
            post_map[key] = r.mem_bytes
    out: Dict[Tuple[int, str], Tuple[int, int]] = {}
    for key, pre in pre_map.items():
        if key in post_map:
            out[key] = (pre, post_map[key])
    return out


def _print_per_layer_table(rank: int, phase: str, pairs: Dict[Tuple[int, str], Tuple[int, int]]) -> None:
    """Per-layer × per-submodule delta in GB."""
    if rank != 0 or not pairs:
        return

    layers = sorted({k[0] for k in pairs.keys()})
    print(f"\n--- {phase} per-layer per-submodule (delta GB) ---", flush=True)
    header = "layer," + ",".join(SUBMODULE_TARGETS)
    print(header, flush=True)
    for L in layers:
        row = [str(L)]
        for sub in SUBMODULE_TARGETS:
            v = pairs.get((L, sub))
            if v is None:
                row.append("")
            else:
                pre, post = v
                row.append(f"{(post - pre) / _GB:+.3f}")
        print(",".join(row), flush=True)


def _print_aggregate(rank: int, phase: str, pairs: Dict[Tuple[int, str], Tuple[int, int]]) -> None:
    """Per-submodule mean/min/max/sum across layers."""
    if rank != 0 or not pairs:
        return

    print(f"\n--- {phase} aggregate per-submodule (delta GB across layers) ---", flush=True)
    print("submodule,n_layers,mean,min,max,sum", flush=True)

    by_sub: Dict[str, List[float]] = {}
    for (_layer, sub), (pre, post) in pairs.items():
        by_sub.setdefault(sub, []).append((post - pre) / _GB)

    # Print in target order for stable diff-ability.
    rows: List[Tuple[str, List[float]]] = []
    for sub in SUBMODULE_TARGETS:
        if sub in by_sub:
            rows.append((sub, by_sub[sub]))

    for sub, vals in rows:
        if not vals:
            continue
        mean_v = statistics.fmean(vals)
        min_v = min(vals)
        max_v = max(vals)
        sum_v = sum(vals)
        print(
            f"{sub},{len(vals)},{mean_v:+.3f},{min_v:+.3f},{max_v:+.3f},{sum_v:+.3f}",
            flush=True,
        )

    # Implicit derived metrics — answers the question "where does the
    # MoE block's saved activation actually live?".
    print("", flush=True)
    print("--- derived (per-layer mean GB) ---", flush=True)
    means: Dict[str, float] = {}
    for sub, vals in by_sub.items():
        means[sub] = statistics.fmean(vals) if vals else 0.0

    sa = means.get("self_attn", 0.0)
    qkv = means.get("self_attn.qkv_proj", 0.0)
    op = means.get("self_attn.o_proj", 0.0)
    qln = means.get("self_attn.query_layernorm", 0.0)
    kln = means.get("self_attn.key_layernorm", 0.0)
    attn_internals = sa - qkv - op - qln - kln
    print(
        f"attention_internals (self_attn - qkv - o - q_ln - k_ln) = "
        f"{attn_internals:+.3f}  # SDPA + QKV split + reshapes",
        flush=True,
    )

    mlp = means.get("mlp", 0.0)
    gate = means.get("mlp.gate", 0.0)
    shared = means.get("mlp.shared_mlp", 0.0)
    experts = mlp - gate - shared
    print(
        f"moe_experts_contrib (mlp - gate - shared) = {experts:+.3f}  # 64-expert FFN + combine work",
        flush=True,
    )

    in_ln = means.get("input_layernorm", 0.0)
    post_ln = means.get("post_attention_layernorm", 0.0)
    total = in_ln + sa + post_ln + mlp
    print(
        f"layer_total (in_ln + self_attn + post_ln + mlp) = "
        f"{total:+.3f}  # cross-check vs the per-layer probe (~5 GB / layer expected)",
        flush=True,
    )


def _print_state(rank: int, label: str) -> None:
    if rank != 0:
        return
    alloc = torch.cuda.memory_allocated() / _GB
    peak = torch.cuda.max_memory_allocated() / _GB
    reserved = torch.cuda.memory_reserved() / _GB
    print(
        f"# [{label}] allocated={alloc:.3f}GB peak={peak:.3f}GB reserved={reserved:.3f}GB",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/dockerdata/HunyuanImage-3")
    parser.add_argument("--prompt", default="A cute cat sitting on a wooden chair")
    parser.add_argument("--bot-task", default="image")
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora", action="store_true")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    _r(rank, f"# rank={rank}/{world_size} device={device}")

    # --- Build pipeline + Policy ---------------------------------------
    config = HunyuanImage3PipelineConfig(
        pretrained_model_ckpt_path=args.ckpt,
        model_precision="bf16",
        autocast_precision="bf16",
        trajectory_precision="bf16",
        logprob_precision="fp32",
        device=device,
    )
    t0 = time.time()
    pipe = HunyuanImage3Pipeline.from_meta_config(config, strategy=FlowSDEStrategy())
    _r(rank, f"# meta-init in {time.time() - t0:.1f}s")

    configs: list = []
    if args.lora:
        configs.append(
            LoRAPolicyConfig(
                rank=8,
                alpha=16,
                dropout=0.0,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            )
        )
    configs.append(
        FSDPPolicyConfig(
            cpu_offload=False,
            param_dtype="bf16",
            mixed_precision=True,
            fsdp_mode="full",
            reshard_after_forward=True,
        )
    )
    t0 = time.time()
    policy: Policy = compose_policy(pipe.diffusion, configs)
    chain = " → ".join(type(p).__name__ for p in walk_source_chain(policy))
    _r(rank, f"# compose_policy in {time.time() - t0:.1f}s — {chain}")

    t0 = time.time()
    pipe.bundle.materialize(device=device, with_aux=("vae",))
    _r(rank, f"# bundle.materialize in {time.time() - t0:.1f}s")
    policy.post_materialize_init()
    _print_state(rank, "post-materialize")

    # --- Locate decoder layers and attach sub-module probes ----------
    transformer = pipe.bundle.transformer
    if hasattr(transformer, "model") and hasattr(transformer.model, "layers"):
        layers = list(transformer.model.layers)
    elif hasattr(transformer, "layers"):
        layers = list(transformer.layers)
    else:
        raise SystemExit("Could not locate decoder layers")

    probe = SubmoduleProbe()
    n_layers, n_per_layer = probe.attach(layers)
    _r(
        rank,
        f"# attached probe: {n_layers} layers × {n_per_layer} submodules (targets={list(SUBMODULE_TARGETS)})",
    )

    # --- Rollout to get a real RolloutResp ---------------------------
    req = RolloutReq(
        sample_ids=["s0"],
        group_ids=["g0"],
        primitives={"text": Texts(texts=[args.prompt])},
        stage_params={
            "task": "t2i",
            "bot_task": args.bot_task,
            "diffusion": {
                "num_inference_steps": 1,
                "guidance_scale": args.guidance_scale,
                "height": args.height,
                "width": args.width,
                "seed": args.seed,
                "sde_indices": [0],
                "eta": args.eta,
            },
        },
    )
    torch.cuda.reset_peak_memory_stats()
    probe.set_phase("rollout_no_grad")
    t0 = time.time()
    resp = pipe.generate(req)
    _print_state(rank, f"post-generate(dt={time.time() - t0:.1f}s)")

    diff_conds = HunyuanImage3DiffusionConditions.from_dict(resp.conditions)
    params = HunyuanImage3DiffusionParams(
        num_inference_steps=1,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        seed=args.seed,
        sde_indices=[0],
        eta=args.eta,
    )
    seg = resp.rollout_traces["image"]

    # --- Phase A: replay no_grad -------------------------------------
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    probe.set_phase("A_replay_no_grad")
    t0 = time.time()
    with torch.no_grad():
        _ = pipe.diffusion.replay(diff_conds, segment=seg, params=params)
    _print_state(rank, f"post-phase-A(dt={time.time() - t0:.1f}s)")

    # --- Phase B: replay grad-enabled (training path) ----------------
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    probe.set_phase("B_replay_grad")
    try:
        t0 = time.time()
        _ = pipe.diffusion.replay(diff_conds, segment=seg, params=params)
        _print_state(rank, f"post-phase-B(dt={time.time() - t0:.1f}s)")
    except torch.cuda.OutOfMemoryError as e:
        _r(rank, f"# phase B OOM (expected after ~9 layers): {e}")
        _print_state(rank, "post-phase-B-OOM")

    probe.detach()

    # --- Reports -----------------------------------------------------
    for ph in ("rollout_no_grad", "A_replay_no_grad", "B_replay_grad"):
        pairs = _phase_pairs(probe.records, ph)
        _print_per_layer_table(rank, ph, pairs)
        _print_aggregate(rank, ph, pairs)

    if rank == 0:
        print("\n# DONE", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main() or 0)
