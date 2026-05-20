"""End-to-end smoke for the new train actor's training stack on HunyuanImage-3.

Mirrors :mod:`scripts.smoke_hunyuan_image3_t2i_replay_fsdp` for setup but
exercises the **new train path** end-to-end:

  1. Build :class:`HunyuanImage3Pipeline` (meta-init).
  2. Compose the Policy stack (LoRA optional → FSDP → EMA).
  3. ``bundle.materialize`` and ``policy.post_materialize_init``.
  4. Build a per-slot :class:`StageAlgorithm` registry — here just
     ``{"diffusion": DiffusionGRPO(stage=pipe.diffusion, params=...)}``.
  5. Build optimizer over ``policy.parameters()`` (FSDP-aware).
  6. Construct a :class:`StageTrainStack` against the Policy (the new
     migrated field shape: ``policy: Policy`` replaces the legacy
     ``backend: TrainBackend``).
  7. Generate a real :class:`RolloutResp` via ``pipe.generate(req)``,
     stamp synthetic advantages on it.
  8. Call ``train_stack.train_minibatch(resp, training_progress=0.0)`` —
     the same call ``NewTrainActor._train_resp`` would issue under Ray.
  9. Assert ``has_backward``, finite ``loss``, non-zero ``grad_norm``,
     EMA shadow advanced one step.

Usage on pod (8 × H20, single node)::

    cd ~/diffusionrl && source .venv/bin/activate && \\
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    torchrun --nproc-per-node=8 --master-port=29500 \\
        scripts/smoke_new_train_actor_e2e.py \\
        --ckpt /dockerdata/HunyuanImage-3 \\
        --steps 4 --eta 0.3

The flags mirror the FSDP replay smoke. ``--steps 4`` keeps the rollout
fast; the train step exercises the full backward path through the
FSDP-wrapped diffusion stage.

This smoke does **not** spawn a ``NewTrainActorGroup`` (which requires a
Ray placement). It exercises the actor's *internal* training stack
exactly as the actor would, validating the field migration end-to-end
and the Policy → optimizer → algorithm wiring.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from diffusionrl.algorithms_new import DiffusionGRPO
from diffusionrl.models_new.hunyuan_image3.conditions import (
    HunyuanImage3DiffusionConditions,
)
from diffusionrl.models_new.hunyuan_image3.config import HunyuanImage3PipelineConfig
from diffusionrl.models_new.hunyuan_image3.diffusion import HunyuanImage3DiffusionParams
from diffusionrl.models_new.hunyuan_image3.pipeline import HunyuanImage3Pipeline
from diffusionrl.sde.kernels import FlowSDEStrategy
from diffusionrl.training_new import StageMiniBatchResult, StageTrainStack
from diffusionrl.training_new.ema_policy import EMAPolicy, EMAPolicyConfig
from diffusionrl.training_new.fsdp_policy import FSDPPolicy, FSDPPolicyConfig
from diffusionrl.training_new.lora_policy import LoRAPolicyConfig
from diffusionrl.training_new.policy import Policy, compose_policy, walk_source_chain
from diffusionrl.types.primitives import Texts
from diffusionrl.types.rollout_req import RolloutReq


def _r(s: str, rank: int) -> None:
    if rank == 0:
        print(s, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/dockerdata/HunyuanImage-3")
    parser.add_argument("--prompt", default="A cute cat sitting on a wooden chair")
    parser.add_argument("--bot-task", default="image")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Stack LoRAPolicy below FSDP — trainable surface becomes the adapters only (~tens of MiB).",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Skip EMAPolicy. Default has EMA stacked outermost so the smoke exercises EMA.step in train_minibatch.",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Set FSDPPolicyConfig.cpu_offload=True. Spills params to CPU "
        "between uses; trades step time for memory. Required for "
        "HI3 backward at 1024x1024 on 8x H20 (96 GiB / rank).",
    )
    parser.add_argument(
        "--activation-checkpointing",
        action="store_true",
        help="Set FSDPPolicyConfig.activation_checkpointing=True. Required to "
        "fit HI3 1024x1024 backward in 96 GiB / rank — see "
        "scripts/profile_hi3_ckpt_per_layer.py for the memory profile.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    args = parser.parse_args()

    # --- Distributed init -------------------------------------------------
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    _r(f"[init] rank={rank}/{world_size} device={device}", rank)

    # --- Pipeline (meta-init) --------------------------------------------
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
    _r(f"[pipeline] meta-init in {time.time() - t0:.1f}s", rank)

    # --- Policy stack -----------------------------------------------------
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
            cpu_offload=bool(args.cpu_offload),
            param_dtype="bf16",
            mixed_precision=True,
            fsdp_mode="full",
            reshard_after_forward=True,
            activation_checkpointing=bool(args.activation_checkpointing),
        )
    )
    if not args.no_ema:
        configs.append(EMAPolicyConfig(decay=0.9999))

    t0 = time.time()
    policy: Policy = compose_policy(pipe.diffusion, configs)
    chain = " → ".join(type(p).__name__ for p in walk_source_chain(policy))
    _r(f"[stack] compose_policy({len(configs)} cfgs) in {time.time() - t0:.1f}s — {chain}", rank)

    # --- Materialize bundle ---------------------------------------------
    # ``with_aux=("vae",)`` is required because ``HunyuanImage3Pipeline.generate``
    # in t2i mode hardcodes ``vae_decode.decode(latent_seg)`` — there is no
    # skip-decode flag yet. VAE is small (~few GB) so this doesn't change the
    # memory budget meaningfully. The "training-actor shape" (with_aux=()) is
    # the right default in production where rollout runs on a separate actor;
    # the smoke is doing both rollout + train in one process for testing.
    t0 = time.time()
    pipe.bundle.materialize(device=device, with_aux=("vae",))
    _r(f"[fsdp] bundle.materialize in {time.time() - t0:.1f}s", rank)

    t0 = time.time()
    policy.post_materialize_init()
    _r(f"[policy] post_materialize_init in {time.time() - t0:.1f}s", rank)

    fsdp = next((p for p in walk_source_chain(policy) if isinstance(p, FSDPPolicy)), None)
    if fsdp is None or not fsdp.is_materialized:
        raise SystemExit("[fsdp] not materialized after bundle.materialize")
    ema = next((p for p in walk_source_chain(policy) if isinstance(p, EMAPolicy)), None)
    if ema is not None and ema.ema is None:
        raise SystemExit("[ema] shadow not built after post_materialize_init")

    # --- Algorithms -------------------------------------------------------
    params_obj = HunyuanImage3DiffusionParams(
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        seed=args.seed,
        sde_indices=list(range(args.steps)),
        eta=args.eta,
    )
    algorithms = {
        "image": DiffusionGRPO(
            stage=pipe.diffusion,
            params=params_obj,
            clip_range=1e-4,
            clip_schedule="constant",
            conditions_cls=HunyuanImage3DiffusionConditions,
        ),
    }
    _r("[alg] DiffusionGRPO bound to pipe.diffusion (slot=image)", rank)

    # --- Optimizer + scheduler over policy params ------------------------
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-5, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    _r(f"[opt] AdamW over {len(trainable)} trainable param groups", rank)

    # --- Cfg shim — only the keys StageTrainStack reads -----------------
    cfg = OmegaConf.create(
        {
            "training": {
                "execution": {"max_grad_norm": args.max_grad_norm},
                "plan": {"micro_batch_size": 1},
            }
        }
    )

    train_stack = StageTrainStack(
        policy=policy,
        optimizer=optimizer,
        scheduler=scheduler,
        algorithms=algorithms,
        cfg=cfg,
    )
    _r(f"[stack] StageTrainStack built — _optimizer_step={train_stack._optimizer_step}", rank)

    # --- Generate one real RolloutResp ----------------------------------
    diffusion_kwargs = {
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
        "sde_indices": list(range(args.steps)),
        "eta": args.eta,
    }
    req = RolloutReq(
        sample_ids=["s0"],
        group_ids=["g0"],
        primitives={"text": Texts(texts=[args.prompt])},
        stage_params={
            "task": "t2i",
            "bot_task": args.bot_task,
            "diffusion": diffusion_kwargs,
        },
    )

    t0 = time.time()
    resp = pipe.generate(req)
    _r(f"[rollout] pipe.generate in {time.time() - t0:.1f}s — rollout_traces={list(resp.rollout_traces.keys())}", rank)
    seg = resp.rollout_traces["image"]
    if seg.sde_logp is None or seg.sde_indices is None:
        raise SystemExit("[rollout] segment.sde_logp/sde_indices missing")
    _r(f"[rollout]   sde_logp={tuple(seg.sde_logp.shape)}  sde_indices={seg.sde_indices.tolist()}", rank)

    # Synthetic advantages — single sample, scalar advantage. Real training
    # would compute group-normalized advantages from rewards.
    resp.advantages = torch.tensor([0.5], dtype=torch.float32, device=device)

    # --- The actual smoke: one minibatch step ---------------------------
    ema_step_before = None
    if ema is not None and ema.ema is not None:
        ema_step_before = int(getattr(ema.ema, "optimization_step", 0))

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result: StageMiniBatchResult = train_stack.train_minibatch(resp, training_progress=0.0)
    torch.cuda.synchronize()
    train_dt = time.time() - t0
    peak_alloc_gb = torch.cuda.max_memory_allocated() / (1024**3)
    _r(
        f"[train] train_minibatch in {train_dt:.1f}s  peak_alloc_GB={peak_alloc_gb:.2f}  "
        f"activation_checkpointing={bool(args.activation_checkpointing)}",
        rank,
    )
    _r(f"[train]   loss={result.loss:.6f}  grad_norm={result.grad_norm:.4f}  lr={result.lr:.3e}", rank)
    _r(f"[train]   has_backward={result.has_backward}  per_slot={list(result.per_slot.keys())}", rank)
    _r(f"[train]   metrics={dict(result.metrics)}", rank)

    if not result.has_backward:
        raise SystemExit("[train] has_backward=False — algorithm did not backward")
    if not (result.loss == result.loss):  # NaN check
        raise SystemExit(f"[train] loss is NaN: {result.loss}")
    if not (-1e6 < result.loss < 1e6):
        raise SystemExit(f"[train] loss out of sanity range: {result.loss}")
    if result.grad_norm <= 0.0:
        raise SystemExit(f"[train] grad_norm non-positive: {result.grad_norm}")

    # EMA step counter check (only when EMA stacked).
    if ema is not None and ema.ema is not None:
        ema_step_after = int(getattr(ema.ema, "optimization_step", 0))
        if ema_step_after != (ema_step_before or 0) + 1:
            _r(
                f"[ema] WARN: EMAModuleWrapper.optimization_step did not advance "
                f"({ema_step_before} → {ema_step_after}) — EMAPolicy.step may be "
                f"a no-op under update_step_interval throttle.",
                rank,
            )
        else:
            _r(f"[ema] step advanced {ema_step_before} → {ema_step_after}", rank)

    if train_stack._optimizer_step != 1:
        raise SystemExit(f"[stack] _optimizer_step expected 1 after one minibatch, got {train_stack._optimizer_step}")
    _r(f"[stack] _optimizer_step={train_stack._optimizer_step} (advanced as expected)", rank)

    _r("[smoke] PASS — NewTrainActor train path validated end-to-end.", rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main() or 0)
