"""FSDP-wrapped t2i diffuse + replay parity smoke for HunyuanImage 3.0.

Sister of :mod:`scripts.smoke_hunyuan_image3_t2i_replay`. Wraps the
diffusion stage's trainable surface (``pipeline.diffusion``) with the
new :class:`diffusionrl.training_new.fsdp_policy.FSDPPolicy` and runs the
identical diffuse → replay parity check.

Three paths exercised by CLI flags:

  * **default (full load)** — ``Pipeline.from_config(...)`` loads weights
    eagerly to CPU on every rank, then :class:`FSDPPolicy` casts +
    ``fully_shard``. Same memory footprint as the legacy smoke.

  * **``--meta-init``** — ``Pipeline.from_meta_config(...)`` builds every
    parameter (decoder + heads + vae + vit) under
    ``accelerate.init_empty_weights()``. ``bundle.materialize(device,
    with_aux=("vae",))`` then drives a single DCP-broadcast weight load
    in one collective: rank 0 reads the filtered HF state_dict; all ranks
    receive their shards. Decoder uses DTensor; heads + vae use plain
    tensors — DCP handles both transparently.

  * **``--no-load-vae``** — combined with ``--meta-init``, sets
    ``with_aux=()`` so vae and vit stay on meta entirely. Mirrors the
    training-actor shape (vae/vit not invoked in diffusion replay).
    Skips the image-decode/save step.

Same parity contract: per-step SDE log-probs from sampling must match
the replay recompute to within ``--tol-abs`` (default ``1e-2``).

Usage on pod (8x H20, single node)::

    cd ~/diffusionrl && source .venv/bin/activate && \\
    torchrun --nproc-per-node=8 --master-port=29500 \\
        scripts/smoke_hunyuan_image3_t2i_replay_fsdp.py \\
        --ckpt /dockerdata/HunyuanImage-3 \\
        --prompt "A cute cat sitting on a wooden chair" \\
        --steps 8 --guidance-scale 2.5 --eta 0.3 \\
        --output ./smoke_t2i_replay_fsdp.png

Add ``--meta-init`` for the 80B-friendly path; add ``--no-load-vae`` to
also drop VAE/ViT (training-actor shape; skips image save). Add ``--lora``
to inject peft adapters before FSDP wrap; add ``--ema`` to track an EMA
shadow over the trainable params (LoRA-only when stacked over LoRA).

For ``--lora`` (and ``--ema``) on H20-class GPUs, prefix the launch with
``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` — without it, the
MoE topkgating step in the diffuse forward OOMs on allocator
fragmentation (~27 GiB contiguous needed; budget held in fragmented
free space without expandable_segments).

The non-FSDP smoke achieves ``max_abs=4.7e-04``; under FSDP we expect
the same order of magnitude (cross-rank bf16 reduction order can add
at most ~10×, well under the ``1e-2`` tolerance). Under ``--lora`` (or
``--lora --ema``), expect bit-exact ``max_abs=0.000e+00`` because
``lora_B`` initializes to zero so adapter contribution is exactly zero
at init (and EMA shadow equals params at init, so
``use_ema_parameters()`` is a no-op for forward).
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Tuple

import torch
import torch.distributed as dist

from diffusionrl.models_new.hunyuan_image3.conditions import (
    HunyuanImage3DiffusionConditions,
)
from diffusionrl.models_new.hunyuan_image3.config import HunyuanImage3PipelineConfig
from diffusionrl.models_new.hunyuan_image3.diffusion import (
    HunyuanImage3DiffusionParams,
)
from diffusionrl.models_new.hunyuan_image3.pipeline import HunyuanImage3Pipeline
from diffusionrl.sde.kernels import FlowSDEStrategy
from diffusionrl.training_new.ema_policy import EMAPolicy, EMAPolicyConfig
from diffusionrl.training_new.fsdp_policy import FSDPPolicy, FSDPPolicyConfig
from diffusionrl.training_new.lora_policy import LoRAPolicyConfig
from diffusionrl.training_new.policy import Policy, compose_policy, walk_source_chain
from diffusionrl.types.primitives import Texts
from diffusionrl.types.rollout_req import RolloutReq


def _summarize_mismatch(label: str, sampled: torch.Tensor, replayed: torch.Tensor) -> Tuple[float, float]:
    diff = (sampled - replayed).abs()
    denom = sampled.abs().clamp_min(1e-12)
    rel = diff / denom
    max_abs = float(diff.max().item())
    max_rel = float(rel.max().item())
    print(f"[t2i-replay-fsdp] {label}:")
    print(f"  shape       = {tuple(sampled.shape)}")
    print(f"  sampled[0]  = {sampled[0].tolist()}")
    print(f"  replayed[0] = {replayed[0].tolist()}")
    print(f"  abs_diff[0] = {diff[0].tolist()}")
    print(f"  max_abs     = {max_abs:.3e}")
    print(f"  max_rel     = {max_rel:.3e}")
    return max_abs, max_rel


def _build_pipeline(
    config: HunyuanImage3PipelineConfig,
    *,
    meta_init: bool,
    rank: int,
) -> HunyuanImage3Pipeline:
    """Either full-load or meta-init the pipeline.

    Both paths return a pipeline whose decoder is FSDP-wrappable. The
    meta path leaves every parameter on meta — caller drives weight load
    via ``bundle.materialize(device, with_aux=...)`` after wrapping.
    """
    t0 = time.time()
    if meta_init:
        if rank == 0:
            print("[fsdp] meta-init pipeline (decoder + heads + aux all on meta)")
        pipe = HunyuanImage3Pipeline.from_meta_config(config, strategy=FlowSDEStrategy())
    else:
        if rank == 0:
            print(f"[fsdp] full-load pipeline from {config.pretrained_model_ckpt_path}")
        pipe = HunyuanImage3Pipeline.from_config(config, strategy=FlowSDEStrategy())
    if rank == 0:
        print(f"[fsdp] pipeline built in {time.time() - t0:.1f}s")
    return pipe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/dockerdata/HunyuanImage-3")
    parser.add_argument("--prompt", default="A cute cat sitting on a wooden chair")
    parser.add_argument(
        "--bot-task",
        default="image",
        choices=("auto", "image", "think", "recaption", "think_recaption", "img_ratio"),
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="./smoke_t2i_replay_fsdp.png")
    parser.add_argument("--tol-abs", type=float, default=1e-2)
    parser.add_argument(
        "--meta-init",
        action="store_true",
        help="Build pipeline with every parameter on meta; "
        "bundle.materialize(...) drives a single DCP-broadcast "
        "weight load after FSDP wrap. Drops CPU peak from "
        "~160 GiB × world_size to ~160 GiB × 1 (rank 0 only).",
    )
    parser.add_argument(
        "--no-load-vae",
        action="store_true",
        help="Combined with --meta-init: pass with_aux=() to "
        "bundle.materialize so vae/vit stay on meta. Mirrors "
        "the training-actor shape (vae/vit not invoked in diffusion "
        "replay). Skips the image-decode/save step.",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Stack LoRAPolicy on top of the diffusion stage before FSDP "
        "wrap. peft adapters are injected in-place; FSDP shards them "
        "with the base. After bundle.materialize, "
        "policy.post_materialize_init() walks inward to reset adapter "
        "weights (kaiming_uniform_(A), zeros_(B)). Replay parity "
        "should be unchanged since lora_B starts at zero.",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        help="Stack EMAPolicy as the outermost policy. Snapshots a shadow "
        "of the trainable params (LoRA adapters when stacked over "
        "LoRA, otherwise full base params, FSDP-sharded) in "
        "post_materialize_init. The smoke does not run an optimizer "
        "step, so the shadow stays equal to the params and "
        "use_ema_parameters() produces bit-equivalent forwards.",
    )
    args = parser.parse_args()

    if args.eta <= 0.0:
        raise SystemExit("--eta must be > 0 for replay parity (eta=0 disables SDE log-probs).")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    if rank == 0:
        print(
            f"[fsdp] world={world} rank={rank} device={device} ckpt={args.ckpt} "
            f"meta_init={args.meta_init} no_load_vae={args.no_load_vae}"
        )

    # Build the pipeline (full-load or meta-init).
    config = HunyuanImage3PipelineConfig(
        pretrained_model_ckpt_path=args.ckpt,
        model_precision="bf16",
        autocast_precision="bf16",
        trajectory_precision="bf16",
        logprob_precision="fp32",
        device=device,
    )
    pipe = _build_pipeline(config, meta_init=args.meta_init, rank=rank)

    # Build the policy stack via compose_policy. Order = inside-out: the
    # first config in ``configs`` becomes the *innermost* policy (closest
    # to the Stage), the last becomes the outermost. peft injection and
    # ``fully_shard`` mutate the underlying nn.Module in place, so all
    # policies in the stack share the same model object and only differ
    # in which surface they expose. EMA is intentionally outermost — its
    # shadow observes the FSDP-sharded view of trainable params.
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
    if args.ema:
        configs.append(EMAPolicyConfig(decay=0.9999))

    t0 = time.time()
    policy: Policy = compose_policy(pipe.diffusion, configs)
    if rank == 0:
        chain = " → ".join(type(p).__name__ for p in walk_source_chain(policy))
        print(f"[stack] compose_policy({len(configs)} configs) done in {time.time() - t0:.1f}s — chain: {chain}")

    # Single-call materialization: allocates per-rank storage for the
    # FSDP-wrapped decoder + wrapper-level heads + opt-in vae, then DCP-
    # broadcasts weights from rank 0 in one collective. ``with_aux=()``
    # leaves vae / vit on meta (training-actor shape).
    if args.meta_init:
        with_aux: Tuple[str, ...] = () if args.no_load_vae else ("vae",)
        t0 = time.time()
        if rank == 0:
            print(f"[fsdp] bundle.materialize start (with_aux={with_aux}, ckpt={args.ckpt}) ...")
        pipe.bundle.materialize(device=device, with_aux=with_aux)
        if rank == 0:
            print(f"[fsdp] bundle.materialize done in {time.time() - t0:.1f}s")

        # Walk the source chain inward, letting each policy run any
        # post-materialize state init it owns. For LoRAPolicy this is
        # the adapter reset (kaiming_uniform_(A), zeros_(B)). FSDPPolicy
        # itself is a no-op here (default delegation hits the LoRA layer
        # if present, else terminates at the Stage).
        t0 = time.time()
        policy.post_materialize_init()
        if rank == 0:
            print(f"[policy] post_materialize_init done in {time.time() - t0:.1f}s")

        # ``is_materialized`` is FSDPPolicy-specific. With EMA stacked
        # outermost, ``policy`` is an EMAPolicy and doesn't expose it —
        # walk the chain to find the FSDPPolicy and ask there.
        fsdp_policy = next(
            (p for p in walk_source_chain(policy) if isinstance(p, FSDPPolicy)),
            None,
        )
        if fsdp_policy is not None and not fsdp_policy.is_materialized:
            raise SystemExit(
                "[fsdp] FSDPPolicy.is_materialized is False after "
                "bundle.materialize — some FSDP shard is still on meta?"
            )

        # When EMAPolicy is in the stack, sanity check that its shadow
        # was actually built during the post_materialize_init walk.
        ema_policy = next(
            (p for p in walk_source_chain(policy) if isinstance(p, EMAPolicy)),
            None,
        )
        if ema_policy is not None:
            if ema_policy.ema is None:
                raise SystemExit(
                    "[ema] EMAPolicy.ema is None after post_materialize_init — shadow snapshot didn't run?"
                )
            if rank == 0:
                print(
                    f"[ema] shadow built — {len(ema_policy.ema.ema_parameters)} "
                    f"shadow tensors (decay={ema_policy.config.decay})"
                )

    # Build the rollout request.
    sde_indices = list(range(args.steps))
    diffusion_kwargs = {
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
        "sde_indices": sde_indices,
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

    if rank == 0:
        print(
            f"[t2i-replay-fsdp] prompt={args.prompt!r}  bot_task={args.bot_task}  "
            f"steps={args.steps}  size={args.height}x{args.width}  "
            f"eta={args.eta}  cfg={args.guidance_scale}"
        )

    # --- 1. Diffuse ---------------------------------------------------
    t0 = time.time()
    resp = pipe.generate(req)
    diffuse_dt = time.time() - t0
    seg = resp.rollout_traces["image"]
    if rank == 0:
        print(
            f"[t2i-replay-fsdp] diffuse+decode in {diffuse_dt:.1f}s  "
            f"latents={tuple(seg.latents.shape)}  "
            f"sde_logp={tuple(seg.sde_logp.shape)}  "
            f"sde_indices={seg.sde_indices.tolist()}"
        )
    if seg.sde_logp is None or seg.sde_indices is None:
        raise SystemExit(
            "[t2i-replay-fsdp] segment.sde_logp / sde_indices missing — eta or sde_indices got overridden somewhere?"
        )

    # --- 2. Save sanity image (rank 0; only when VAE is loaded) ------
    if rank == 0 and not args.no_load_vae and "image" in resp.decoded:
        out_pixels = resp.decoded["image"].pixels[0]
        from torchvision.transforms.functional import to_pil_image

        to_pil_image(out_pixels.clamp(0, 1).float().cpu()).save(args.output)
        print(f"[t2i-replay-fsdp] saved sampled image -> {args.output}")
    elif rank == 0:
        print("[t2i-replay-fsdp] skipping image save (no VAE materialized)")

    # --- 3. Replay ---------------------------------------------------
    diff_conds = HunyuanImage3DiffusionConditions.from_dict(resp.conditions)
    params = HunyuanImage3DiffusionParams(**diffusion_kwargs)

    t0 = time.time()
    with torch.no_grad():
        replay_result = policy.replay(diff_conds, segment=seg, params=params)
    replay_logp = replay_result.log_probs
    replay_dt = time.time() - t0
    if rank == 0:
        print(
            f"[t2i-replay-fsdp] replay all steps in {replay_dt:.1f}s "
            f"(per-step ~{replay_dt / max(1, args.steps):.2f}s)  "
            f"replay_logp={tuple(replay_logp.shape)}"
        )

    # --- 4. Compare full-replay parity on rank 0 ---------------------
    # ALL ranks must participate in any further policy.replay call (it
    # triggers FSDP collectives). The print/compare only happens on rank 0.
    max_abs = 0.0
    max_rel = 0.0
    if rank == 0:
        sampled = seg.sde_logp.detach().to(replay_logp.dtype).to(replay_logp.device)
        max_abs, max_rel = _summarize_mismatch("full SDE replay", sampled, replay_logp)
        full_verdict = "PASS" if max_abs <= args.tol_abs else "FAIL"
        print(
            f"[t2i-replay-fsdp] full-replay {full_verdict} — "
            f"max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
            f"(tol_abs={args.tol_abs:.3e})"
        )

    # Free intermediate state before the subset replay; the prior full
    # replay's reserved-but-unallocated fragments push rank 0 over the
    # 96 GiB H20 budget without this.
    del replay_logp, replay_result
    torch.cuda.empty_cache()
    dist.barrier()

    # --- 5. Subset replay (all ranks, best-effort) -------------------
    sub_abs = 0.0
    sub_rel = 0.0
    sub_ok = False
    if args.steps >= 4:
        subset = [1, args.steps - 1]
        try:
            with torch.no_grad():
                sub_result = policy.replay(diff_conds, segment=seg, params=params, step_indices=subset)
            sub_logp = sub_result.log_probs
            if rank == 0:
                sub_sampled = sampled[:, [sde_indices.index(i) for i in subset]]
                sub_abs, sub_rel = _summarize_mismatch(f"subset replay step_indices={subset}", sub_sampled, sub_logp)
            sub_ok = True
        except torch.cuda.OutOfMemoryError as e:
            if rank == 0:
                print(f"[t2i-replay-fsdp] subset replay OOM (skipping): {e}")
            torch.cuda.empty_cache()

    if rank == 0:
        if sub_ok:
            print(
                f"[t2i-replay-fsdp] subset-replay {'PASS' if sub_abs <= args.tol_abs else 'FAIL'} — "
                f"max_abs={sub_abs:.3e} max_rel={sub_rel:.3e}"
            )
        if max_abs > args.tol_abs:
            print("[t2i-replay-fsdp] FAIL")
        else:
            print(
                f"[t2i-replay-fsdp] PASS — full max_abs={max_abs:.3e}, "
                f"subset max_abs={sub_abs:.3e}{' (skipped due to OOM)' if not sub_ok else ''}"
            )

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0 and max_abs > args.tol_abs:
        sys.exit(1)


if __name__ == "__main__":
    main()
