"""Per-layer GPU memory profile of HI3 replay in three contexts.

Phases (executed in order on the SAME materialized pipeline):

  A. ``pipe.diffusion.replay`` under ``torch.no_grad()`` — matches the
     rollout/diffuse path. Establishes the no-grad baseline.
  B. ``pipe.diffusion.replay`` WITHOUT no_grad — same forward but PyTorch
     saves activations for the backward graph. This is what
     ``DiffusionGRPO.compute_loss_and_backward`` triggers inside
     ``StageTrainStack.train_optimizer_step``.
  C. ``loss.backward()`` on the sum of replay log_probs — if (B) survived
     OOM, run backward and record post-backward memory.

Per-layer instrumentation: forward pre- and post-hooks on every
``decoder_layer`` in ``pipe.bundle.transformer.model.layers``. Each hook
records ``torch.cuda.memory_allocated()`` and ``max_memory_allocated()``.
Per-layer delta in phase B reveals how much each layer adds to the saved-
activation graph; the sum is the total activation budget for backward.

Usage (one rank for measurement clarity — FSDP all-shards costs are
identical across ranks, and single-rank is enough to characterise the
activation tax)::

    cd ~/diffusionrl && source .venv/bin/activate && \\
    PYTORCH_ALLOC_CONF=expandable_segments:True \\
    torchrun --nproc-per-node=8 --master-port=29500 \\
        scripts/profile_hi3_replay_memory.py --ckpt /dockerdata/HunyuanImage-3

Only rank 0 prints the report. Reports CSV to stdout for easy paste.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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


def _r(rank: int, s: str) -> None:
    if rank == 0:
        print(s, flush=True)


@dataclass
class HookRecord:
    """One (phase, layer, stage) snapshot."""

    phase: str
    layer: int
    stage: str  # "pre" or "post"
    mem_bytes: int
    peak_bytes: int


class LayerProbe:
    """Forward pre/post hooks on every decoder layer for memory instrumentation."""

    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.phase: Optional[str] = None
        self.records: List[HookRecord] = []
        self._handles: List[Any] = []

    def attach(self, layers) -> None:
        for i, layer in enumerate(layers):
            self._handles.append(layer.register_forward_pre_hook(self._make_pre(i)))
            self._handles.append(layer.register_forward_hook(self._make_post(i)))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def set_phase(self, name: str) -> None:
        self.phase = name

    def _record(self, layer: int, stage: str) -> None:
        if self.phase is None:
            return
        torch.cuda.synchronize()
        self.records.append(
            HookRecord(
                phase=self.phase,
                layer=layer,
                stage=stage,
                mem_bytes=int(torch.cuda.memory_allocated()),
                peak_bytes=int(torch.cuda.max_memory_allocated()),
            )
        )

    def _make_pre(self, idx: int):
        def hook(module, args):
            self._record(idx, "pre")

        return hook

    def _make_post(self, idx: int):
        def hook(module, args, output):
            self._record(idx, "post")

        return hook


def _summarize_phase(rank: int, phase: str, records: List[HookRecord], num_layers: int) -> None:
    """Print a CSV-style per-layer summary for one phase."""
    if rank != 0:
        return

    # Filter to this phase, group by (layer, stage).
    by_layer: Dict[int, Dict[str, HookRecord]] = {}
    for r in records:
        if r.phase != phase:
            continue
        by_layer.setdefault(r.layer, {})[r.stage] = r

    print(f"\n--- phase: {phase} ---", flush=True)
    print("layer,mem_before_GB,mem_after_GB,delta_GB,peak_GB", flush=True)

    deltas: List[float] = []
    for i in range(num_layers):
        rec = by_layer.get(i, {})
        pre = rec.get("pre")
        post = rec.get("post")
        if pre is None or post is None:
            continue
        delta_b = post.mem_bytes - pre.mem_bytes
        deltas.append(delta_b / _GB)
        print(
            f"{i},{pre.mem_bytes / _GB:.3f},{post.mem_bytes / _GB:.3f},{delta_b / _GB:.3f},{post.peak_bytes / _GB:.3f}",
            flush=True,
        )

    if deltas:
        total = sum(deltas)
        avg = total / len(deltas)
        print(f"# layers_measured={len(deltas)}", flush=True)
        print(f"# total_layer_delta_GB={total:.3f}", flush=True)
        print(f"# avg_per_layer_delta_GB={avg:.3f}", flush=True)
        print(f"# max_per_layer_delta_GB={max(deltas):.3f}", flush=True)
        print(f"# min_per_layer_delta_GB={min(deltas):.3f}", flush=True)


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
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Stack LoRAPolicy under FSDP — drops gradient buffer size to "
        "the adapter footprint. Phases B and C are otherwise dominated "
        "by full-model gradients.",
    )
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    _r(rank, f"# rank={rank}/{world_size} device={device}")
    _print_state(rank, "post-init")

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
    _r(
        rank,
        f"# compose_policy in {time.time() - t0:.1f}s — chain: "
        + " → ".join(type(p).__name__ for p in walk_source_chain(policy)),
    )

    t0 = time.time()
    pipe.bundle.materialize(device=device, with_aux=("vae",))
    _r(rank, f"# bundle.materialize in {time.time() - t0:.1f}s")
    policy.post_materialize_init()
    _print_state(rank, "post-materialize")

    # --- Locate decoder layers and attach probes ----------------------
    transformer = pipe.bundle.transformer
    # HF AutoModel convention: model.layers is a ModuleList of decoder layers.
    if hasattr(transformer, "model") and hasattr(transformer.model, "layers"):
        layers = list(transformer.model.layers)
    elif hasattr(transformer, "layers"):
        layers = list(transformer.layers)
    else:
        raise SystemExit("Could not locate decoder layers on the transformer")
    num_layers = len(layers)
    _r(rank, f"# located {num_layers} decoder layers (type={type(layers[0]).__name__})")

    probe = LayerProbe(rank=rank)
    probe.attach(layers)

    # --- Run a rollout to get a real segment + reconstruct conditions --
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
    _print_state(rank, "pre-generate")

    probe.set_phase("rollout_no_grad")
    t0 = time.time()
    resp = pipe.generate(req)
    rollout_dt = time.time() - t0
    _print_state(rank, f"post-generate(dt={rollout_dt:.1f}s)")

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

    # --- Phase A: replay under no_grad --------------------------------
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _print_state(rank, "pre-phase-A")

    probe.set_phase("A_replay_no_grad")
    t0 = time.time()
    with torch.no_grad():
        result_a = pipe.diffusion.replay(diff_conds, segment=seg, params=params)
    _print_state(rank, f"post-phase-A(dt={time.time() - t0:.1f}s)")
    del result_a

    # --- Phase B: replay WITHOUT no_grad (training-replay forward) ---
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _print_state(rank, "pre-phase-B")

    probe.set_phase("B_replay_grad")
    phase_b_ok = False
    loss = None
    try:
        t0 = time.time()
        result_b = pipe.diffusion.replay(diff_conds, segment=seg, params=params)
        _print_state(rank, f"post-phase-B(dt={time.time() - t0:.1f}s)")
        # Construct a scalar loss for backward (sum of log_probs).
        loss = result_b.log_probs.sum()
        phase_b_ok = True
    except torch.cuda.OutOfMemoryError as e:
        _r(rank, f"# phase B OOM: {e}")
        _print_state(rank, "post-phase-B-OOM")

    # --- Phase C: backward (if phase B made it) ----------------------
    if phase_b_ok and loss is not None:
        torch.cuda.reset_peak_memory_stats()
        _print_state(rank, "pre-phase-C")
        probe.set_phase("C_backward")
        try:
            t0 = time.time()
            loss.backward()
            _print_state(rank, f"post-phase-C(dt={time.time() - t0:.1f}s)")
        except torch.cuda.OutOfMemoryError as e:
            _r(rank, f"# phase C OOM: {e}")
            _print_state(rank, "post-phase-C-OOM")

    probe.detach()

    # --- Reports ------------------------------------------------------
    _summarize_phase(rank, "rollout_no_grad", probe.records, num_layers)
    _summarize_phase(rank, "A_replay_no_grad", probe.records, num_layers)
    _summarize_phase(rank, "B_replay_grad", probe.records, num_layers)
    if phase_b_ok:
        _summarize_phase(rank, "C_backward", probe.records, num_layers)

    if rank == 0:
        print("\n# DONE", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main() or 0)
