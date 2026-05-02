#!/usr/bin/env python
"""diffusionRL training entry point (Hydra-native).

Spawns ``TrainActor`` and ``RolloutActor`` handles directly from a Hydra
``DictConfig`` and drives the rollout loop via ``RolloutPipeline``.

Usage:
    python -m diffusionrl.train algorithm=grpo model=flux ...
"""

from __future__ import annotations

import logging
import time

import hydra
from omegaconf import DictConfig, OmegaConf

# Populate Hydra's ConfigStore with every @register_config leaf before
# @hydra.main triggers composition. Done as an explicit call (not a side
# effect of importing diffusionrl.config) so Ray workers and other importers
# don't trigger the package-wide walk, which can create circular imports.
from diffusionrl.config import register_all_configs

register_all_configs()

logger = logging.getLogger(__name__)


def _run_cross_component_validators(cfg: DictConfig) -> None:
    """Cross-component invariants that span multiple resolved sections.

    Runs on the driver after ``validate(cfg)`` has materialized every
    registered leaf. Each helper raises ``ValueError`` with a single-line
    message on failure.
    """
    from diffusionrl.config.validation import (
        validate_dynamic_dotpaths,
        validate_lora_target_modules,
        validate_offload_contract,
        validate_rollout_layout,
        validate_sampling_chunk_geometry,
        validate_training_actor_sampling_mode,
        validate_training_batch_geometry,
        validate_weight_sync_contract,
    )

    validate_dynamic_dotpaths(cfg)
    # LoRA target materializer must run before downstream validators / Ray
    # bootstrap so PEFT and SGLang share one resolved target list.
    validate_lora_target_modules(cfg)
    validate_training_actor_sampling_mode(cfg)
    validate_training_batch_geometry(cfg)
    validate_sampling_chunk_geometry(cfg)
    validate_weight_sync_contract(cfg)
    validate_rollout_layout(cfg)
    validate_offload_contract(cfg)


def train(cfg: DictConfig) -> None:
    """Synchronous training entrypoint, cfg-native."""
    debug_mode = str(cfg.debug.get("mode") or "none").strip().lower()
    if debug_mode and debug_mode != "none":
        raise NotImplementedError(
            f"debug.mode={debug_mode!r} is not supported on the cfg-native "
            "train entry (the argparse debug runner is retired)."
        )

    from diffusionrl.config.instantiate import build, validate

    # Fail-fast schema check on the driver before any Ray work.
    validate(cfg)
    _run_cross_component_validators(cfg)

    import ray

    from diffusionrl.config.validation import is_direct_sampling
    from diffusionrl.distributed.transfer_queue import TransferQueueRuntime
    from diffusionrl.ray.group import RolloutActorGroup, TrainActorGroup
    from diffusionrl.rollout.pipeline import RolloutPipeline, build_media_preview
    from diffusionrl.utils import configure_logger, load_function, set_seed
    from diffusionrl.utils.train_utils import (
        collect_rollout_batch_metrics,
        maybe_restore_start_rollout_id_from_checkpoint,
        should_eval,
        should_log,
        should_save,
    )
    from diffusionrl.utils.wandb_logger import aggregate_metrics, init_logger
    from diffusionrl.utils.wandb_metrics import compute_rollout_batch_metrics

    configure_logger()
    set_seed(int(cfg.run.seed))

    # Derived inline from cfg — no intermediate launch/info wrappers.
    sync_cfg = cfg.get("sync")
    sync_enabled = sync_cfg is not None
    sync_target = str(sync_cfg.get("_target_") or "") if sync_enabled else ""
    training_actor_sampling_mode = is_direct_sampling(cfg)

    if sync_enabled and sync_target.endswith("UpdateWeightFromCheckpoint"):
        raise NotImplementedError(
            "train does not yet support sync=checkpoint_path "
            "(would call training_runtime.export_weights_to_path which the new TrainActor lacks)."
        )

    control_algorithm = build(cfg.algorithm)
    training_plan = OmegaConf.to_object(cfg.training.plan)
    topology = OmegaConf.to_object(cfg.training.topology)
    sampling_spec = OmegaConf.to_object(cfg.sampling)

    # Driver-side logging derivations; bootstrap re-derives its own copies.
    from diffusionrl.ray.group.train import _backend_name_from_cfg

    backend_name = _backend_name_from_cfg(cfg)

    logger.info("Starting diffusionRL training...")
    logger.info("Model: %s", cfg.model.pretrained_model_ckpt_path)
    logger.info("Algorithm: %s", cfg.algorithm._target_)
    logger.info("Mode: %s", "direct_sampling" if training_actor_sampling_mode else "separate")
    logger.info(
        "Offload train: %s, Offload rollout: %s",
        cfg.training.execution.offload_train,
        cfg.training.execution.offload_rollout,
    )
    logger.info("Weight sync: %s", sync_target if sync_enabled else "disabled")
    logger.info(
        "Periodic controls: save_steps=%s eval_steps=%s logging_steps=%s",
        cfg.resume.save_steps,
        cfg.evaluation.eval_steps,
        cfg.logging.logging_steps,
    )
    logger.info(
        "Debug flags: mode=%s save_intermediates=%s save_dir=%s",
        debug_mode,
        bool(cfg.debug.save_intermediates),
        cfg.debug.save_dir,
    )

    # Initialize Ray. Auto-connects via RAY_ADDRESS when set, else local.
    if not ray.is_initialized():
        import os

        if os.environ.get("RAY_ADDRESS"):
            # Connecting to an existing cluster — Ray rejects num_cpus/num_gpus
            # in this mode. The head allocator already provisioned resources.
            ray.init()
        else:
            # Local cluster: cap num_cpus to avoid slow startup on high-core
            # machines (e.g. 384 cores) where Ray pre-starts one worker per CPU,
            # causing connect timeouts.
            _max_cpus = min(os.cpu_count() or 64, 64)
            ray.init(num_cpus=_max_cpus)

    wandb_logger = None
    rollout_group = None
    train_group = None
    placement = None

    try:
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
        rollout_on_gpu = True

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

        # 3. Bootstrap rollout group
        if not training_actor_sampling_mode:
            rollout_group = RolloutActorGroup.bootstrap(
                cfg=cfg,
                placement=placement,
            )

        # 4. Optional pre-offload to relieve memory pressure before train init
        if not training_actor_sampling_mode and cfg.training.execution.offload_rollout:
            rollout_group.sleep()
            rollout_on_gpu = False

        # 5. Bootstrap training group
        train_group, master_addr, master_port = TrainActorGroup.bootstrap(
            cfg=cfg,
            placement=placement,
            colocate=bool(cfg.placement.colocate),
            colocate_gpu_fraction=float(cfg.placement.colocate_gpu_fraction),
        )
        logger.info(
            "Resolved training distributed master: addr=%s port=%s world_size=%d",
            master_addr,
            master_port,
            train_group.num_actors,
        )

        if cfg.resume.resume_from_checkpoint:
            ckpt_path = str(cfg.resume.resume_from_checkpoint)
            train_group.load_checkpoint(ckpt_path)
            logger.info("Checkpoint loaded: %s", ckpt_path)
            restored_rollout_id = maybe_restore_start_rollout_id_from_checkpoint(
                cfg,
                ckpt_path,
            )
            if restored_rollout_id is not None:
                logger.info(
                    "Auto-set start_rollout_id=%s from checkpoint path.",
                    restored_rollout_id,
                )

        expected_global_batch_size = int(training_plan.global_batch_size)
        train_backend_info = {
            "name": backend_name,
            "topology": topology.as_dict(),
            "training_plan": training_plan.as_dict(),
        }
        logger.info("Training backend: %s", train_backend_info)
        logger.info("Expected global batch size: %s", expected_global_batch_size)

        # 6. Initial weight broadcast skipped (FSDP2 deterministic per-rank init)
        logger.info("Initial update_weights() skipped — FSDP2 TrainActor inits deterministically per rank")

        # 7. Post-init offload dance
        if not training_actor_sampling_mode and cfg.training.execution.offload_rollout:
            if cfg.training.execution.offload_train:
                train_group.offload()
            rollout_group.wake_up()
            rollout_on_gpu = True

        # 8. Weight sync setup
        if not training_actor_sampling_mode and sync_enabled:
            train_group.setup_weight_sync(
                sync_cfg=sync_cfg,
                placement_cfg=placement.config,
                rollout_runtime=rollout_group,
            )

        # 8b. Resolve pipeline dispatch targets
        if training_actor_sampling_mode:
            pipeline_group = train_group
        else:
            pipeline_group = rollout_group

        # 8c. Initialize TransferQueue (no-op when cfg.transfer_queue is absent)
        tq_runtime = TransferQueueRuntime().install()
        tq_handoffs = tq_runtime.init(cfg)
        if tq_handoffs is not None:
            controller_handoff, actor_handoff = tq_handoffs
            tq_runtime.create_client("controller", controller_handoff, sync=True)
            if not training_actor_sampling_mode:
                tq_runtime.init_remote_actor_clients(rollout_group.get_actors(), actor_handoff)
            tq_runtime.init_remote_actor_clients(train_group.get_actors(), actor_handoff)

        # 9. Main training loop
        wandb_media_enabled = wandb_logger is not None and bool(cfg.logging.log_media)
        wandb_media_max_items = max(1, int(cfg.logging.media_max_items))
        global_optimizer_step = 0
        for rollout_id in range(int(cfg.resume.start_rollout_id), int(cfg.run.num_rollouts)):
            step_start_t = time.perf_counter()
            sync_phase_s = 0.0
            eval_phase_s = 0.0
            should_log_step = should_log(rollout_id, cfg)

            # === Reset TransferQueue zero-copy buffer free (no-op when disabled) ===
            if cfg.get("transfer_queue") is not None:
                tq_runtime.reset_zero_copy_buffer_free()
                tq_runtime.reset_actors_zero_copy_buffer_free(train_group.get_actors())
                if not training_actor_sampling_mode:
                    tq_runtime.reset_actors_zero_copy_buffer_free(rollout_group.get_actors())

            # === Periodic Eval (before rollout/train of this step) ===
            if should_eval(rollout_id, cfg):
                eval_phase_start_t = time.perf_counter()
                if not training_actor_sampling_mode:
                    if cfg.training.execution.offload_rollout and not rollout_on_gpu:
                        rollout_group.wake_up()
                        rollout_on_gpu = True

                def _run_eval():
                    return rollout_pipeline.run_eval(
                        rollout_group=pipeline_group,
                        data_source=data_source,
                        prompt_batch_size=prompt_batch_size,
                        samples_per_prompt=samples_per_prompt,
                        sampling_spec=sampling_spec,
                        evaluation_settings=cfg.evaluation,
                        rollout_id=rollout_id,
                        collect_media_preview=wandb_media_enabled,
                        media_max_items=wandb_media_max_items,
                    )

                if training_actor_sampling_mode:
                    with train_group.use_eval_ema():
                        eval_metrics = _run_eval()
                else:
                    eval_metrics = _run_eval()
                eval_phase_s = time.perf_counter() - eval_phase_start_t
                logger.info(
                    "Eval at %s: mean_reward=%.4f std_reward=%.4f (%.3fs)",
                    rollout_id,
                    eval_metrics.get("mean_reward", 0.0),
                    eval_metrics.get("std_reward", 0.0),
                    eval_phase_s,
                )
                if wandb_logger:
                    # Pop media_preview before log_eval so only scalar metrics
                    # (mean/std reward, num_samples, rollout_id) go through the
                    # ``eval/`` numeric panel; images are logged separately.
                    eval_media_preview = eval_metrics.pop("media_preview", None)
                    wandb_logger.log_eval(rollout_id, eval_metrics)
                    if eval_media_preview:
                        wandb_logger.log_generated_media(
                            rollout_id,
                            eval_media_preview,
                            key="eval/generated_media",
                        )

            # === PHASE A: Rollout ===
            rollout_phase_start_t = time.perf_counter()
            if not training_actor_sampling_mode:
                if cfg.training.execution.offload_rollout and not rollout_on_gpu:
                    rollout_group.wake_up()
                    rollout_on_gpu = True

            collect_rollout_preview = bool(should_log_step and wandb_media_enabled)
            training_batch, sample_count, rollout_response = rollout_pipeline.run_once(
                rollout_group=pipeline_group,
                data_source=data_source,
                prompt_batch_size=prompt_batch_size,
                samples_per_prompt=samples_per_prompt,
                sampling_spec=sampling_spec,
                control_algorithm=control_algorithm,
                rollout_id=rollout_id,
                collect_media_preview=collect_rollout_preview,
                media_max_items=wandb_media_max_items,
            )
            rollout_phase_s = time.perf_counter() - rollout_phase_start_t
            reward_compute_s = float(rollout_response.samples.reward_compute_s)

            # Even when no preview was requested, the response is now small
            # (actors drop decoded_images after scoring), so we keep the
            # reference for the logging path below. No eager ``del`` needed.

            if not training_actor_sampling_mode and cfg.training.execution.offload_rollout:
                rollout_group.sleep()
                rollout_on_gpu = False

            # === PHASE B: Training ===
            train_phase_start_t = time.perf_counter()
            if cfg.training.execution.offload_train:
                train_group.onload()

            batch_ref = ray.put(training_batch)
            results = train_group.train(rollout_id, training_batch)
            metrics = [r.to_legacy_metric_dict() for r in results]
            train_phase_s = time.perf_counter() - train_phase_start_t

            # Periodic save (collective; before offload so weights are still on GPU)
            if should_save(rollout_id, cfg):
                save_path = f"{cfg.resume.output_dir}/checkpoint-{rollout_id}"
                train_group.save_model(save_path)
                logger.info("Checkpoint saved: %s", save_path)

            # === Clear TransferQueue Partition Data (no-op when disabled) ===
            if cfg.get("transfer_queue") is not None:
                tq_runtime.clear_partition()

            # === PHASE C: Offload + Weight Sync ===
            if cfg.training.execution.offload_train:
                train_group.offload()
            else:
                train_group.clear_memory()

            if (
                not training_actor_sampling_mode
                and sync_enabled
                and (rollout_id + 1) % int(cfg.run.weight_sync_interval) == 0
            ):
                sync_phase_start_t = time.perf_counter()
                rollout_group.wake_up()
                train_group.sync_weights_to_rollout()
                sync_phase_s = time.perf_counter() - sync_phase_start_t
                rollout_on_gpu = True

            # === Per-optimizer-step wandb logging ===
            if wandb_logger and metrics:
                per_step_list = metrics[0].get("_per_optimizer_step_metrics", [])
                for per_step_m in per_step_list:
                    if per_step_m.get("has_backward", False):
                        global_optimizer_step += 1
                        wandb_step_m = {k: v for k, v in per_step_m.items() if k != "has_backward"}
                        wandb_logger.log_step(global_optimizer_step, wandb_step_m)

            if should_log_step:
                avg_loss = sum(m.get("loss", 0) for m in metrics) / max(len(metrics), 1)
                step_time_s = time.perf_counter() - step_start_t
                logger.info(
                    "Rollout %s: loss=%.4E rollout=%.3fs reward=%.3fs train=%.3fs sync=%.3fs eval=%.3fs step=%.3fs",
                    rollout_id,
                    avg_loss,
                    rollout_phase_s,
                    reward_compute_s,
                    train_phase_s,
                    sync_phase_s,
                    eval_phase_s,
                    step_time_s,
                )
                if wandb_logger:
                    aggregated = aggregate_metrics(metrics)
                    aggregated["loss"] = avg_loss
                    wandb_logger.log_rollout(rollout_id, aggregated)

                    rollout_metrics = collect_rollout_batch_metrics(
                        ray_module=ray,
                        batch_ref=batch_ref,
                        compute_rollout_batch_metrics_fn=compute_rollout_batch_metrics,
                    )
                    if rollout_metrics:
                        wandb_logger.log_rollout(rollout_id, rollout_metrics)

                    if wandb_media_enabled:
                        # Primary path: actors populated ``samples.media_preview``
                        # already (capped in ``RolloutPipeline.aggregate``). Fall
                        # back to ``build_media_preview`` only if a legacy code
                        # path left it unset.
                        media_preview = build_media_preview(
                            rollout_response,
                            max_items=wandb_media_max_items,
                        )
                        if media_preview is not None:
                            wandb_logger.log_generated_media(rollout_id, media_preview)
                        del rollout_response

                    perf_metrics = {
                        "rollout_phase_s": rollout_phase_s,
                        "reward_compute_s": reward_compute_s,
                        "train_phase_s": train_phase_s,
                        "sync_phase_s": sync_phase_s,
                        "eval_phase_s": eval_phase_s,
                        "step_time_s": step_time_s,
                        "samples_per_rollout": float(sample_count),
                        "samples_per_s": (
                            float(sample_count) / float(step_time_s) if step_time_s > 0 and sample_count > 0 else 0.0
                        ),
                    }
                    wandb_logger.log_perf(rollout_id, perf_metrics)

                    if sync_phase_s > 0:
                        wandb_logger.log_with_step(
                            step_key="rollout/step",
                            step=rollout_id,
                            metrics={"sync/elapsed_s": sync_phase_s},
                        )
    finally:
        if not training_actor_sampling_mode and sync_enabled and train_group is not None:
            try:
                train_group.teardown_weight_sync()
            except Exception:
                logger.exception("Weight-sync teardown failed")
        if rollout_group is not None:
            rollout_group.dispose()
        if train_group is not None:
            train_group.dispose()
        if placement is not None:
            try:
                placement.destroy()
            except Exception:
                logger.exception("Placement teardown failed")
        if wandb_logger:
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
