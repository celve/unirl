#!/usr/bin/env python
"""diffusionRL training entry point (Hydra-native).

Spawns :class:`RolloutActorGroup` and :class:`TrainActorGroup` and drives
the rollout / train / weight-sync loop via :class:`RolloutPipeline`.

Dual-mode sampling, gated on
:func:`diffusionrl.config.validation.is_direct_sampling`:

- **Separate sampling** (default; ``cfg.rollout.engine: vllm_omni`` /
  ``sglang`` / etc.): rollout runs on sibling :class:`RolloutActor`
  instances; trainer→rollout weight sync via the configured ``cfg.sync``
  preset (NCCL broadcast, IPC, ...).
- **Direct sampling** (``cfg.rollout.engine: trainside``): the
  FSDP-wrapped training Policy IS the sampler;
  :meth:`RolloutActorGroup.from_train_group` adopts the train actor
  handles. No proxy actor, no weight sync, no offload dance.

Scope: rollout → train → weight-sync loop with structured wandb logging
(when enabled), media preview, and PhaseTimings. Periodic eval and
periodic save / resume are deferred follow-ups.

Usage::

    # vllm-omni separate sampling with NCCL weight sync (production):
    python -m diffusionrl.train \\
        +experiment=flowgrpo_fast_sd3

    # Trainside direct sampling (no sync needed):
    python -m diffusionrl.train \\
        +experiment=flowgrpo_fast_sd3_trainside
"""

from __future__ import annotations

import contextlib
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

# Populate Hydra's ConfigStore with every @register_config leaf before
# @hydra.main triggers composition.
from diffusionrl.config import register_all_configs

register_all_configs()

logger = logging.getLogger(__name__)


def _run_cross_component_validators(cfg: DictConfig) -> None:
    """Cross-component invariants (dual-mode).

    Calls the dynamic-dotpath, LoRA-target-modules and training-batch-geometry
    validators that apply to both modes, plus the three direct-sampling /
    weight-sync / offload contracts that gate on ``is_direct_sampling(cfg)``.
    """
    from diffusionrl.config.validation import (
        validate_dynamic_dotpaths,
        validate_lora_target_modules,
        validate_offload_contract,
        validate_training_batch_geometry,
        validate_weight_sync_contract,
    )

    validate_dynamic_dotpaths(cfg)
    validate_lora_target_modules(cfg)
    validate_training_batch_geometry(cfg)
    validate_weight_sync_contract(cfg)
    validate_offload_contract(cfg)


def _should_use_ema_rollout(cfg: DictConfig) -> bool:
    """Check whether any configured algorithm requires EMA weights during rollout.

    Resolves each ``cfg.algorithms.<slot>._target_`` to its class and reads
    the ``requires_ema_rollout`` class attribute. Returns True if ANY
    algorithm declares it (off-policy / NFT); False otherwise (on-policy / GRPO).
    """
    for slot, alg_node in cfg.algorithms.items():
        target = str(alg_node.get("_target_") or "")
        if not target:
            continue
        try:
            alg_cls = hydra.utils.get_class(target)
        except Exception as exc:
            logger.warning(
                "EMA rollout check: failed to resolve _target_=%r for slot %r: %s. "
                "Assuming on-policy (no EMA rollout).",
                target,
                slot,
                exc,
            )
            continue
        if getattr(alg_cls, "requires_ema_rollout", False):
            logger.info("EMA rollout policy: enabled (off-policy algorithm in slot %r)", slot)
            return True
    logger.info("EMA rollout policy: disabled (on-policy)")
    return False


def train(cfg: DictConfig) -> None:
    """Synchronous training entrypoint."""
    debug_mode = str(cfg.debug.get("mode") or "none").strip().lower()
    if debug_mode and debug_mode != "none":
        raise NotImplementedError(f"debug.mode={debug_mode!r} is not supported (the argparse debug runner is retired).")

    from diffusionrl.config.instantiate import build, validate
    from diffusionrl.config.validation import is_direct_sampling

    # Fail-fast schema check on the driver before any Ray work.
    validate(cfg)
    _run_cross_component_validators(cfg)

    import ray

    from diffusionrl.ray.group.rollout import RolloutActorGroup
    from diffusionrl.ray.group.train import TrainActorGroup
    from diffusionrl.rollout.pipeline import RolloutPipeline
    from diffusionrl.utils import configure_logger, load_function, set_seed
    from diffusionrl.utils.timing import PhaseTimings
    from diffusionrl.utils.wandb_logger import (
        aggregate_stage_results,
        init_logger,
    )
    from diffusionrl.utils.wandb_metrics import compute_rollout_resp_metrics

    configure_logger()
    set_seed(int(cfg.run.seed))

    direct_sampling = is_direct_sampling(cfg)
    sync_cfg = cfg.get("sync")
    # Weight sync is only meaningful for separate sampling — in direct mode
    # the trainer Policy IS the sampler, so trainer→rollout sync is a no-op.
    sync_enabled = (sync_cfg is not None) and (not direct_sampling)
    sync_target = str(sync_cfg.get("_target_") or "") if sync_enabled else ""

    if sync_enabled and sync_target.endswith("UpdateWeightFromCheckpoint"):
        raise NotImplementedError(
            "train does not yet support sync=checkpoint_path "
            "(no driver-side export_weights_to_path hook on the train actor)."
        )

    control_algorithm = build(cfg.algorithm)
    sampling_spec = OmegaConf.to_object(cfg.sampling)

    logger.info("Starting diffusionRL training...")
    logger.info(
        "Sampling mode: %s",
        "direct (trainside)" if direct_sampling else "separate",
    )
    logger.info("Model: %s", cfg.model.pretrained_model_ckpt_path)
    logger.info("Algorithm: %s", cfg.algorithm._target_)
    logger.info("Rollout engine: %s", cfg.rollout.engine._target_)
    logger.info(
        "Offload train: %s, Offload rollout: %s",
        cfg.training.execution.offload_train,
        cfg.training.execution.offload_rollout,
    )
    logger.info("Weight sync: %s", sync_target if sync_enabled else "disabled")

    # Initialize Ray. Auto-connects when RAY_ADDRESS is set or when a
    # local cluster is already running (detected via /tmp/ray/).
    if not ray.is_initialized():
        import os
        from pathlib import Path

        _has_existing = os.environ.get("RAY_ADDRESS") or Path("/tmp/ray/ray_current_cluster").exists()
        if _has_existing:
            ray.init()
        else:
            _max_cpus = min(os.cpu_count() or 64, 64)
            ray.init(num_cpus=_max_cpus)

    wandb_logger = None
    rollout_group = None
    train_group = None
    placement = None

    try:
        # 0. W&B init (parity with legacy train.py; ``init_logger`` is a no-op
        #    when ``report_to_wandb=false`` so the call is always safe).
        if cfg.logging.report_to_wandb and cfg.logging.project_name:
            raw_tags = cfg.logging.tags
            if raw_tags:
                if isinstance(raw_tags, str):
                    raw_tags = raw_tags.split(",")
                wandb_tags = [t.strip() for t in raw_tags if t.strip()]
            else:
                wandb_tags = None
            wandb_logger = init_logger(
                project=str(cfg.logging.project_name),
                run_name=cfg.logging.run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                log_dir=cfg.logging.logging_dir,
                rank=0,
                tags=wandb_tags,
                entity=cfg.logging.entity or None,
                require_success=True,
            )
            if wandb_logger.initialized:
                logger.info(
                    "WandB initialized: project=%s, run=%s",
                    cfg.logging.project_name,
                    cfg.logging.run_name,
                )

        # 1. Resource allocation
        placement = build(cfg.placement)
        logger.info("Placement ready: %s", placement.config)

        # 2. Driver-side data source + sampling defaults
        data_source_cls = load_function(str(cfg.run.data_source_dotpath))
        data_source = data_source_cls(cfg)
        prompt_batch_size = int(cfg.algorithm.prompts_per_rollout)
        samples_per_prompt = int(getattr(control_algorithm, "samples_per_prompt", 1))
        logger.info(
            "Driver-side rollout config: prompt_batch_size=%s samples_per_prompt=%s",
            prompt_batch_size,
            samples_per_prompt,
        )
        rollout_pipeline = RolloutPipeline()
        # PPO multi-epoch: re-train on the same RolloutResp for
        # ``cfg.training.plan.num_updates_per_batch`` optimizer steps. π_old
        # is fixed by ``segment.sde_logp`` (captured during rollout) so
        # every replay step sees the same anchor.
        num_updates_per_batch = int(cfg.training.plan.get("num_updates_per_batch", 1))
        logger.info("Rollout cadence: num_updates_per_batch=%d", num_updates_per_batch)

        # 3. Spawn actor groups.
        #    Direct sampling: the train actor IS the rollout actor, so the
        #    train group must be built first; ``from_train_group`` then
        #    adopts its handles without spawning anything new.
        #    Separate sampling: build rollout_group, optionally sleep, then
        #    build train_group, then wake_up — the colocate offload dance
        #    has the rollout group sleep before the train group spawns so
        #    the GPU bundles aren't double-claimed.
        if direct_sampling:
            train_group = TrainActorGroup(cfg=cfg, placement=placement)
            rollout_group = RolloutActorGroup.from_train_group(train_group, cfg=cfg)
        else:
            rollout_group = RolloutActorGroup(cfg=cfg, placement=placement)
            logger.info("[BOOTSTRAP] Rollout ready.")
            if cfg.training.execution.offload_rollout:
                logger.info("[BOOTSTRAP] Sleeping rollout...")
                rollout_group.sleep()
                logger.info("[BOOTSTRAP] Rollout sleeping done.")
            logger.info("[BOOTSTRAP] Creating train group...")
            train_group = TrainActorGroup(cfg=cfg, placement=placement)
            logger.info("[BOOTSTRAP] Train group created.")
            if cfg.training.execution.offload_rollout:
                if cfg.training.execution.offload_train:
                    logger.info("[BOOTSTRAP] Offloading train before wake...")
                    train_group.offload()
                logger.info("[BOOTSTRAP] Waking rollout...")
                rollout_group.wake_up()
                logger.info("[BOOTSTRAP] Rollout waked up.")

        # 4. Optional weight-sync setup (separate sampling only).
        if sync_enabled:
            param_name_prefix = str(cfg.model.get("weight_sync_param_name_prefix", "") or "")
            # OmegaConf DictConfig doesn't survive Ray serialization cleanly;
            # convert to plain Python dicts/lists recursively.
            _raw_packed = cfg.model.get("weight_sync_packed_modules", None)
            if _raw_packed:
                packed_modules = OmegaConf.to_container(_raw_packed, resolve=True)
            else:
                packed_modules = {}
            train_group.setup_weight_sync(
                sync_cfg=sync_cfg,
                placement_cfg=placement.config,
                rollout_runtime=rollout_group,
                param_name_prefix=param_name_prefix,
                packed_modules=packed_modules,
            )
            logger.info(
                "[BOOTSTRAP] Weight sync configured: %s (param_name_prefix=%r, packed_modules=%d keys)",
                sync_target,
                param_name_prefix,
                len(packed_modules),
            )

        logger.info(
            "Bootstrap complete; entering main loop (rollouts %d..%d)",
            int(cfg.resume.start_rollout_id),
            int(cfg.run.num_rollouts),
        )

        wandb_media_enabled = wandb_logger is not None and bool(cfg.logging.log_media)
        wandb_media_max_items = max(1, int(cfg.logging.media_max_items or 0))

        # Optimizer-step counter under ``train/step``. Starts at 0; V3 resume
        # work will thread checkpoint-saved values through here.
        global_optimizer_step = 0

        _use_ema_rollout = _should_use_ema_rollout(cfg) if direct_sampling else False

        _rollout_is_sleeping = False

        # 5. Main loop
        for rollout_id in range(
            int(cfg.resume.start_rollout_id),
            int(cfg.run.num_rollouts),
        ):
            timings = PhaseTimings()
            logger.info("[LOOP r=%d] Start.", rollout_id)

            # Wake rollout if it was sleeping (from previous iter's train phase).
            if cfg.training.execution.offload_rollout and _rollout_is_sleeping:
                if cfg.training.execution.offload_train:
                    logger.info("[LOOP r=%d] Offloading train before rollout wake...", rollout_id)
                    train_group.offload()
                logger.info("[LOOP r=%d] Waking rollout...", rollout_id)
                rollout_group.wake_up()
                _rollout_is_sleeping = False
                logger.info("[LOOP r=%d] Rollout awake.", rollout_id)

            # --- Rollout: 4 direct phase calls (skip convert_training_data) ---
            # EMA rollout gating: on-policy algorithms (GRPO) MUST sample
            # with base weights so the importance ratio equals 1 on the
            # first training step. Off-policy / forward-process algorithms
            # (NFT) intentionally use EMA-smoothed weights for higher-
            # quality trajectories. The flag ``_use_ema_rollout`` is
            # resolved once before the loop from the configured algorithms'
            # ``requires_ema_rollout`` class attribute.
            with timings.measure("rollout"):
                ema_ctx = (
                    train_group.use_eval_ema() if (direct_sampling and _use_ema_rollout) else contextlib.nullcontext()
                )
                with ema_ctx:
                    prompts = rollout_pipeline.load_prompts(
                        data_source=data_source,
                        prompt_batch_size=prompt_batch_size,
                        samples_per_prompt=samples_per_prompt,
                        init_same_noise=bool(getattr(sampling_spec, "init_same_noise", False)),
                    )
                    # Pre-compute the per-sample x_T tensor on the driver so
                    # rollouts in the same GRPO group draw distinct noise (and
                    # rollout-side sampling matches what trainer replay sees in
                    # ``segment.latents[:, 0, ...]``). For modalities we haven't
                    # wired this for, returns ``None`` and the rollout engine
                    # falls back to its internal RNG.
                    from diffusionrl.rollout.pipeline import compute_initial_noise_for_request

                    initial_noise = compute_initial_noise_for_request(
                        cfg=cfg,
                        prompts=prompts,
                        sampling_spec=sampling_spec,
                        samples_per_prompt=samples_per_prompt,
                        rollout_id=rollout_id,
                    )
                    req, _sde_indices = rollout_pipeline.plan_requests(
                        prompts=prompts,
                        sampling_spec=sampling_spec,
                        samples_per_prompt=samples_per_prompt,
                        control_algorithm=control_algorithm,
                        rollout_id=rollout_id,
                        collect_media_preview=wandb_media_enabled,
                        media_max_items=wandb_media_max_items,
                        initial_latents=initial_noise,
                    )
                    responses = rollout_pipeline.exec_request(
                        req=req,
                        rollout_group=rollout_group,
                        samples_per_prompt=samples_per_prompt,
                    )
                    rollout_resp = rollout_pipeline.aggregate(responses=responses)
            sample_count = int(rollout_resp.batch_size)
            reward_compute_s = float(rollout_resp.reward_compute_s)

            # --- Colocate offload dance: sleep rollout before train ---
            logger.info("[LOOP r=%d] Rollout done (%.1fs).", rollout_id, timings.get("rollout"))
            if cfg.training.execution.offload_rollout:
                logger.info("[LOOP r=%d] Sleeping rollout for train...", rollout_id)
                rollout_group.sleep()
                _rollout_is_sleeping = True

            # --- Train: feed RolloutResp directly (group does balanced split) ---
            with timings.measure("train"):
                if cfg.training.execution.offload_train:
                    logger.info("[LOOP r=%d] Onloading train...", rollout_id)
                    train_group.onload()
                # per_update_results: List[List[TrainOptimizerStepResult]]
                #   outer = per optimizer step, inner = per actor
                per_update_results = train_group.train(rollout_id, rollout_resp)

            # --- Sync, then offload ---
            #
            # IMPORTANT ordering: sync_weights_to_rollout() reads from the
            # train actors' live parameters (CUDA IPC / NCCL broadcast /
            # checkpoint transport — all source-on-GPU). If we offload
            # train BEFORE sync, the source parameters are on CPU and the
            # sync either degrades to a slower path or transmits stale /
            # wrong values silently. Sync first (with both train and
            # rollout on-load), then offload both for the rollout phase
            # of the next iter.
            #
            # Old order (broken):
            #   1. train.offload()    ← source on CPU
            #   2. rollout.wake_up()
            #   3. train.sync_to_rollout()   ← reads from CPU, may degrade
            #   4. rollout.sleep()
            #
            # New order:
            #   1. rollout.wake_up()  (if offload_rollout)
            #   2. train.sync_to_rollout()   ← source on GPU, fast path
            #   3. rollout.sleep()    (if offload_rollout)
            #   4. train.offload()    (if offload_train) — after sync done
            logger.info("[LOOP r=%d] Train done (%.1fs).", rollout_id, timings.get("train"))

            if sync_enabled and (rollout_id + 1) % int(cfg.run.weight_sync_interval) == 0:
                # Offload train before sync so rollout wake_up has GPU room.
                # sync_weights_to_rollout → _to_full_tensor will .cuda() each
                # LoRA tensor back individually (few MB) for the all_gather +
                # ZMQ send, so this is safe even with train offloaded.
                if cfg.training.execution.offload_train:
                    train_group.offload()
                else:
                    train_group.clear_memory()

                with timings.measure("sync"):
                    if cfg.training.execution.offload_rollout:
                        logger.info("[LOOP r=%d] Waking rollout for sync...", rollout_id)
                        rollout_group.wake_up()
                        _rollout_is_sleeping = False
                    train_group.sync_weights_to_rollout()
                    logger.info("[LOOP r=%d] Sync done.", rollout_id)
                    # Sleep rollout after sync (will be waked at next iter start).
                    if cfg.training.execution.offload_rollout:
                        rollout_group.sleep()
                        _rollout_is_sleeping = True
            else:
                # No sync this iter — still offload train for next rollout.
                if cfg.training.execution.offload_train:
                    train_group.offload()
                else:
                    train_group.clear_memory()

            # --- Metric aggregation + console scalar summary ---
            # Flatten all per-update × per-actor results for the rollout-level summary.
            all_results = [r for update in per_update_results for r in update]
            n = max(1, len(all_results))
            mean_loss = float(sum(r.loss for r in all_results) / n)
            mean_grad = float(sum(r.grad_norm for r in all_results) / n)
            mean_lr = float(sum(r.lr for r in all_results) / n) if all_results else 0.0
            reward_mean = float(rollout_resp.rewards.mean().item()) if rollout_resp.rewards is not None else 0.0
            rollout_s = timings.get("rollout")
            train_s = timings.get("train")
            sync_s = timings.get("sync")
            step_s = timings.total()
            logger.info(
                "rollout=%d samples=%d reward_mean=%.4f loss=%.4f grad_norm=%.4f lr=%.2e "
                "phase=[rollout=%.2fs train=%.2fs sync=%.2fs reward_compute=%.2fs] step=%.2fs",
                rollout_id,
                sample_count,
                reward_mean,
                mean_loss,
                mean_grad,
                mean_lr,
                rollout_s,
                train_s,
                sync_s,
                reward_compute_s,
                step_s,
            )

            # --- W&B logging (no-op when wandb_logger is None) ---
            if wandb_logger is not None:
                rollout_metrics = compute_rollout_resp_metrics(resp=rollout_resp)

                # rollout/ panel — averaged training scalars + reward stats.
                rollout_training_metrics = aggregate_stage_results(all_results)
                wandb_logger.log_rollout(
                    rollout_id,
                    {**rollout_training_metrics, **rollout_metrics},
                )

                # train/ panel — one entry PER optimizer step so ratio/loss
                # curves have full num_updates_per_batch granularity.
                for per_actor_results in per_update_results:
                    step_count = sum(int(bool(r.has_backward)) for r in per_actor_results)
                    if step_count > 0:
                        global_optimizer_step += 1
                        step_metrics = aggregate_stage_results(per_actor_results)
                        wandb_logger.log_step(global_optimizer_step, step_metrics)

                # perf/ panel — driver-side timings + actor-reported
                # reward_compute_s (which the rollout-phase guard can't see).
                perf = timings.as_perf_dict(samples=sample_count)
                perf["reward_compute_s"] = reward_compute_s
                wandb_logger.log_perf(rollout_id, perf)

                if sync_s > 0:
                    wandb_logger.log_with_step(
                        step_key="rollout/step",
                        step=rollout_id,
                        metrics={"sync/elapsed_s": sync_s},
                    )

                if wandb_media_enabled and rollout_resp.media_preview is not None:
                    wandb_logger.log_generated_media(rollout_id, rollout_resp.media_preview)
    finally:
        if sync_enabled and train_group is not None:
            try:
                train_group.teardown_weight_sync()
            except Exception:
                logger.exception("Weight-sync teardown failed")
        # Direct sampling: rollout_group adopts train_group handles, so
        # disposing rollout_group would also dispose the train actors. Only
        # dispose train_group; rollout_group has no separate state.
        if not direct_sampling and rollout_group is not None:
            rollout_group.dispose()
        if train_group is not None:
            train_group.dispose()
        if placement is not None:
            try:
                placement.destroy()
            except Exception:
                logger.exception("Placement teardown failed")
        if wandb_logger is not None:
            wandb_logger.finish()

    logger.info("Training complete!")


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: DictConfig) -> None:
    from diffusionrl.config.instantiate import freeze
    from diffusionrl.config.polymorphic import expand_polymorphic_lists

    OmegaConf.resolve(cfg)
    expand_polymorphic_lists(cfg)
    freeze(cfg)
    train(cfg)


if __name__ == "__main__":
    main()
