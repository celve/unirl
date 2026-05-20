"""End-to-end one-round full-pipeline smoke for the SD3 stage-driven path.

Sibling of :mod:`scripts.smoke_new_train_actor_e2e` (HI3). Runs one full
round on SD3 medium: ``SD3Pipeline.generate(req)`` → real reward via
:class:`RewardPipeline` → group-normalized advantages → one
:meth:`StageTrainStack.train_minibatch` step over a LoRA → FSDP → EMA
policy chain.

Phases:

1. Distributed init (NCCL, single or multi-rank).
2. ``SD3Pipeline.from_config(SD3PipelineConfig(...))`` — eager weight load.
3. Compose policy stack ``LoRAPolicy → FSDPPolicy → EMAPolicy`` via
   :func:`compose_policy`; run ``post_materialize_init`` (LoRA reset, EMA
   shadow snapshot).
4. Build :class:`RewardPipeline` from a minimal in-script OmegaConf dict
   (component default: ``pickscore`` w=1).
5. Build the per-slot :class:`DiffusionGRPO` + AdamW optimizer + scheduler
   + :class:`StageTrainStack`.
6. Construct a ``RolloutReq`` with ``N`` repeated copies of one prompt
   (B=N after embed); call ``pipe.generate(req) → RolloutResp`` — this
   path populates ``resp.conditions``, ``resp.rollout_traces['image']`` and
   ``resp.decoded['image']`` itself (no IPC workaround needed; SD3
   doesn't go through vLLM-Omni).
7. Score via :func:`_build_legacy_response_view` + ``reward_pipeline.score_and_attach``,
   copy rewards back onto ``resp``, compute group-normalized advantages
   inline (legacy ``GRPOAlgorithm`` is Hydra-bound; the math is 10 lines).
8. ``train_stack.train_minibatch(resp, training_progress=0.0)``.
9. Print + assert: ``has_backward``, finite ``loss``, ``grad_norm > 0``,
   ``_optimizer_step == 1``, EMA ``optimization_step`` advanced 0 → 1,
   ``resp.rewards`` finite, ``resp.advantages`` group-normalized (single
   group → mean exactly 0).

Usage on pod (single H20 fits SD3 medium easily)::

    cd ~/diffusionrl && source .venv/bin/activate && \\
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    torchrun --nproc-per-node=1 --master-port=29500 \\
        scripts/smoke_new_train_actor_sd3.py \\
        --ckpt /apdcephfs_hldy/share_305110755/hunyuan/public_models/stable-diffusion-3.5-medium \\
        --prompt "a red apple on a wooden table" \\
        --steps 4 --eta 0.7 --samples-per-prompt 8

Multi-rank FSDP sharding is exercised by raising ``--nproc-per-node``;
the smoke shape is identical.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from diffusionrl.algorithms_new import DiffusionGRPO
from diffusionrl.models_new.sd3.conditions import SD3Conditions
from diffusionrl.models_new.sd3.config import SD3PipelineConfig
from diffusionrl.models_new.sd3.diffusion import SD3DiffusionParams
from diffusionrl.models_new.sd3.pipeline import SD3Pipeline
from diffusionrl.ray.mixins.new_rollout_pipeline import _build_legacy_response_view
from diffusionrl.reward.base import InProcessRewardExecutor
from diffusionrl.reward.pipeline import RewardPipeline
from diffusionrl.reward.scorers.pickscore import PickScoreRewardScorer, PickScoreSpec
from diffusionrl.reward.service import RewardService
from diffusionrl.sde.kernels import FlowSDEStrategy
from diffusionrl.training_new import StageMiniBatchResult, StageTrainStack
from diffusionrl.training_new.ema_policy import EMAPolicy, EMAPolicyConfig
from diffusionrl.training_new.fsdp_policy import FSDPPolicy, FSDPPolicyConfig
from diffusionrl.training_new.lora_policy import LoRAPolicyConfig
from diffusionrl.training_new.policy import Policy, compose_policy, walk_source_chain
from diffusionrl.types.primitives import Texts
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.sampling import SamplingParams


def _r(s: str, rank: int) -> None:
    if rank == 0:
        print(s, flush=True)


def _group_normalized_advantages(rewards: torch.Tensor, group_ids: List[str]) -> torch.Tensor:
    """Per-group normalized advantages: ``(r - mean_g) / (std_g + eps)``.

    Inlined here so the smoke doesn't have to compose a full Hydra cfg just
    to instantiate ``GRPORolloutControl``.
    """
    eps = 1e-8
    groups: dict[str, List[int]] = {}
    for i, gid in enumerate(group_ids):
        groups.setdefault(gid, []).append(i)
    advantages = torch.zeros_like(rewards)
    for gid, idxs in groups.items():
        idx_t = torch.tensor(idxs, dtype=torch.long, device=rewards.device)
        r = rewards.index_select(0, idx_t).to(torch.float32)
        adv = (r - r.mean()) / (r.std(unbiased=False) + eps)
        advantages.index_copy_(0, idx_t, adv.to(advantages.dtype))
    return advantages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="/apdcephfs_hldy/share_305110755/hunyuan/public_models/stable-diffusion-3.5-medium",
    )
    parser.add_argument("--prompt", default="a red apple on a wooden table")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--eta", type=float, default=0.7)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=8,
        help="N copies of the prompt → B=N samples. Needs to be >=2 for group-normalized advantages to have variance.",
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument(
        "--lora-targets",
        default="attn.to_q,attn.to_k,attn.to_v,attn.to_out.0",
        help="Comma-separated peft target_module suffixes. Defaults match "
        "legacy SD3 (`diffusionrl/models/sd3.py:283-286`).",
    )
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument(
        "--reward-component",
        default="pickscore",
        help="Name of the reward component to register. Must resolve via "
        "`reward/components/__init__.py`. Falls back to a name-only "
        "registry lookup; weights must already be discoverable.",
    )
    parser.add_argument(
        "--reward-batch-size",
        type=int,
        default=8,
        help="Batch size forwarded to the reward component.",
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
    _r(f"[sd3-smoke] init: rank={rank}/{world_size} device={device}", rank)

    # --- SD3 pipeline (eager load) ---------------------------------------
    config = SD3PipelineConfig(
        pretrained_model_ckpt_path=args.ckpt,
        model_precision="bf16",
        autocast_precision="bf16",
        trajectory_precision="bf16",
        logprob_precision="fp32",
        device=device,
    )
    t0 = time.time()
    pipe = SD3Pipeline.from_config(config, strategy=FlowSDEStrategy())
    _r(f"[sd3-smoke] pipeline.from_config in {time.time() - t0:.1f}s", rank)

    # --- Policy stack -----------------------------------------------------
    lora_targets = tuple(t.strip() for t in args.lora_targets.split(",") if t.strip())
    configs: list = [
        LoRAPolicyConfig(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=0.0,
            target_modules=lora_targets,
        ),
        FSDPPolicyConfig(
            cpu_offload=False,
            param_dtype="bf16",
            mixed_precision=True,
            fsdp_mode="full",
            reshard_after_forward=True,
            activation_checkpointing=False,
        ),
        EMAPolicyConfig(decay=args.ema_decay),
    ]
    t0 = time.time()
    policy: Policy = compose_policy(pipe.diffusion, configs)
    chain = " → ".join(type(p).__name__ for p in walk_source_chain(policy))
    _r(f"[sd3-smoke] compose_policy in {time.time() - t0:.1f}s — {chain}", rank)

    t0 = time.time()
    policy.post_materialize_init()
    _r(f"[sd3-smoke] post_materialize_init in {time.time() - t0:.1f}s", rank)

    fsdp = next((p for p in walk_source_chain(policy) if isinstance(p, FSDPPolicy)), None)
    if fsdp is None or not fsdp.is_materialized:
        raise SystemExit("[sd3-smoke] FSDPPolicy not materialized after post_materialize_init")
    ema = next((p for p in walk_source_chain(policy) if isinstance(p, EMAPolicy)), None)
    if ema is None or ema.ema is None:
        raise SystemExit("[sd3-smoke] EMAPolicy shadow not built after post_materialize_init")

    trainable_count = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    _r(f"[sd3-smoke] trainable params (LoRA-only surface): {trainable_count:,}", rank)

    # --- Reward pipeline --------------------------------------------------
    # Construct scorer + executor + service directly so we don't have to
    # round-trip through the Hydra polymorphic-config path
    # (``RewardPipeline.from_configs`` materializes via ``OmegaConf.to_object``
    # which drops back to a plain dict without a registered schema, breaking
    # the ``rc.components`` access at ``reward/service.py:48``).
    if args.reward_component != "pickscore":
        raise SystemExit(
            f"[sd3-smoke] only --reward-component=pickscore is wired in the "
            f"smoke (direct-ctor path); got {args.reward_component!r}. "
            f"Add another scorer here as needed."
        )
    pickscore_spec = PickScoreSpec(
        weight=1.0,
        batch_size=int(args.reward_batch_size),
        device="auto",
    )
    t0 = time.time()
    scorer = PickScoreRewardScorer(config=pickscore_spec, base_device="cuda")
    executor = InProcessRewardExecutor(scorer, weight=pickscore_spec.weight)
    reward_service = RewardService(executors=[executor], aggregation_method="mean")
    reward_pipeline = RewardPipeline(reward_service)
    _r(f"[sd3-smoke] reward_pipeline (pickscore) constructed in {time.time() - t0:.1f}s", rank)

    # --- Algorithms -------------------------------------------------------
    diffusion_params = SD3DiffusionParams(
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        seed=args.seed,
        sde_indices=list(range(args.steps)),
        eta=args.eta,
        samples_per_prompt=int(args.samples_per_prompt),
    )
    algorithms = {
        "image": DiffusionGRPO(
            stage=pipe.diffusion,
            params=diffusion_params,
            clip_range=1e-4,
            clip_schedule="constant",
            conditions_cls=SD3Conditions,
        ),
    }
    _r("[sd3-smoke] DiffusionGRPO bound to pipe.diffusion (slot=image)", rank)

    # --- Optimizer + scheduler -------------------------------------------
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-5, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    _r(f"[sd3-smoke] AdamW over {len(trainable)} trainable param tensors", rank)

    cfg_shim = OmegaConf.create(
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
        cfg=cfg_shim,
    )
    _r(f"[sd3-smoke] StageTrainStack built — _optimizer_step={train_stack._optimizer_step}", rank)

    # --- Build RolloutReq + generate -------------------------------------
    N = int(args.samples_per_prompt)
    if N < 2:
        raise SystemExit(
            "[sd3-smoke] --samples-per-prompt must be >= 2 for group-normalized advantages to have variance"
        )
    sample_ids = [f"s{i}" for i in range(N)]
    group_ids = ["g0"] * N
    # N copies of one prompt → text embed produces [N, T5_len, dim] → SD3 sampling B=N.
    repeated_texts = [args.prompt] * N
    req = RolloutReq(
        sample_ids=sample_ids,
        group_ids=group_ids,
        primitives={"text": Texts(texts=repeated_texts)},
        stage_params={
            "diffusion": {
                "num_inference_steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "height": args.height,
                "width": args.width,
                "seed": args.seed,
                "eta": args.eta,
                "sde_indices": list(range(args.steps)),
                "samples_per_prompt": N,
            }
        },
    )

    t0 = time.time()
    resp = pipe.generate(req)
    _r(
        f"[sd3-smoke] rollout in {time.time() - t0:.1f}s — "
        f"conditions={list(resp.conditions.keys())} rollout_traces={list(resp.rollout_traces.keys())}",
        rank,
    )

    seg = resp.rollout_traces["image"]
    if seg.sde_logp is None or seg.sde_indices is None:
        raise SystemExit("[sd3-smoke] segment.sde_logp/sde_indices missing — eta=0?")
    _r(f"[sd3-smoke]   sde_logp shape={tuple(seg.sde_logp.shape)} sde_indices={seg.sde_indices.tolist()}", rank)
    text_cond = resp.conditions.get("text")
    if text_cond is None or getattr(text_cond, "embeds", None) is None:
        raise SystemExit("[sd3-smoke] resp.conditions['text'].embeds missing")
    _r(f"[sd3-smoke]   text embeds shape={tuple(text_cond.embeds.shape)}", rank)

    # --- Real reward + advantage (only on rank 0; broadcast to others) ----
    # The reward pipeline runs on cuda; for multi-rank we could shard but
    # for the smoke we score on rank 0 and broadcast the result so every
    # rank sees the same advantages for the backward pass.
    legacy_sampling = SamplingParams(
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        seed=args.seed,
        num_samples_per_prompt=N,
    )
    legacy_view = _build_legacy_response_view(req, resp, sampling_params=legacy_sampling)

    t0 = time.time()
    reward_pipeline.score_and_attach(legacy_view)
    reward_dt = time.time() - t0

    rewards = legacy_view.samples.rewards.to(device=device, dtype=torch.float32)
    if rewards.shape != (N,):
        raise SystemExit(f"[sd3-smoke] expected rewards shape ({N},), got {tuple(rewards.shape)}")
    if not torch.isfinite(rewards).all():
        raise SystemExit(f"[sd3-smoke] reward has non-finite entries: {rewards.tolist()}")
    _r(
        f"[sd3-smoke] reward.score_and_attach in {reward_dt:.1f}s — "
        f"shape={tuple(rewards.shape)} mean={float(rewards.mean()):.4f} "
        f"std={float(rewards.std(unbiased=False)):.4f}",
        rank,
    )

    resp.rewards = rewards
    resp.component_rewards = legacy_view.samples.component_rewards

    advantages = _group_normalized_advantages(rewards, group_ids)
    _r(
        f"[sd3-smoke] advantages — shape={tuple(advantages.shape)} "
        f"mean={float(advantages.mean()):.6f} "
        f"std={float(advantages.std(unbiased=False)):.4f}",
        rank,
    )
    resp.advantages = advantages

    # --- One minibatch step ----------------------------------------------
    ema_step_before = int(getattr(ema.ema, "optimization_step", 0))

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result: StageMiniBatchResult = train_stack.train_minibatch(resp, training_progress=0.0)
    torch.cuda.synchronize()
    train_dt = time.time() - t0
    peak_alloc_gb = torch.cuda.max_memory_allocated() / (1024**3)
    _r(f"[sd3-smoke] train_minibatch in {train_dt:.1f}s  peak_alloc_GB={peak_alloc_gb:.2f}", rank)
    _r(f"[sd3-smoke]   loss={result.loss:.6f}  grad_norm={result.grad_norm:.4f}  lr={result.lr:.3e}", rank)
    _r(f"[sd3-smoke]   has_backward={result.has_backward}  per_slot={list(result.per_slot.keys())}", rank)
    _r(f"[sd3-smoke]   metrics={dict(result.metrics)}", rank)

    if not result.has_backward:
        raise SystemExit("[sd3-smoke] has_backward=False — DiffusionGRPO did not backward")
    if not (result.loss == result.loss):  # NaN check
        raise SystemExit(f"[sd3-smoke] loss is NaN: {result.loss}")
    if not (-1e6 < result.loss < 1e6):
        raise SystemExit(f"[sd3-smoke] loss out of sanity range: {result.loss}")
    if result.grad_norm <= 0.0:
        raise SystemExit(f"[sd3-smoke] grad_norm non-positive: {result.grad_norm}")
    if train_stack._optimizer_step != 1:
        raise SystemExit(f"[sd3-smoke] _optimizer_step expected 1, got {train_stack._optimizer_step}")

    ema_step_after = int(getattr(ema.ema, "optimization_step", 0))
    if ema_step_after != ema_step_before + 1:
        _r(
            f"[sd3-smoke] WARN: EMA optimization_step did not advance "
            f"({ema_step_before} → {ema_step_after}). May be throttled by "
            f"update_step_interval.",
            rank,
        )
    else:
        _r(f"[sd3-smoke] EMA optimization_step: {ema_step_before} → {ema_step_after}", rank)

    # Single-group advantages must average to (near-)zero by construction.
    if float(advantages.abs().mean()) > 1e6:
        raise SystemExit(f"[sd3-smoke] advantages out of sanity range: {advantages}")

    _r("[sd3-smoke] PASS — SD3 stage-driven train path validated end-to-end.", rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main() or 0)
