#!/usr/bin/env python
"""
diffusionrl async training entrypoint (separate mode only).

Usage:
    python -m diffusionrl.train_async --config scripts/example_flux_dancegrpo_sglang_separate.yaml

This overlaps rollout and training with explicit synchronization boundaries:
- launch rollout N+1 while training on rollout N
- periodically synchronize weights with explicit boundary
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from diffusionrl.algorithms.construction import create_algorithm_from_init_payload
from diffusionrl.cmdline.parse_args import parse_args_with_derived_config
from diffusionrl.cmdline.resolution import build_launch_config
from diffusionrl.cmdline.schema import build_derived_config_view
from diffusionrl.cmdline.validation import validate_async_training_runner
from diffusionrl.config import DerivedConfig
from diffusionrl.rollout.service_interface import compute_dataset_step_info
from diffusionrl.utils.train_utils import (
    collect_rollout_batch_metrics,
    maybe_restore_start_rollout_id_from_checkpoint,
    should_eval,
    should_log,
    should_save,
)
from diffusionrl.utils.wandb_logger import aggregate_metrics
from diffusionrl.utils.wandb_metrics import build_buffer_metrics, build_sync_metrics

if TYPE_CHECKING:
    from diffusionrl.distributed.weight_sync import WeightSyncCoordinator

logger = logging.getLogger(__name__)


"""
Main control-plane path (async mode):
parse_args -> build_launch_config -> create_placement_groups_from_launch
-> create_rollout_services -> create_training_actor_group
-> prepare RolloutRequest(s) -> async_generate
-> resolve requests -> rollout_buffer.push/pop -> training_group.train -> weight_sync.sync
"""

@dataclass(frozen=True)
class InflightRollout:
    """A rollout launched but not yet consumed."""

    rollout_id: int
    future: Any


@dataclass(frozen=True)
class Rollout:
    """A rollout-scoped future result resolved from an inflight future."""

    rollout_id: int
    result: Any


@dataclass(frozen=True)
class PreparedRolloutPlan:
    """One rollout request plan visible to the async driver."""

    context: Any
    batch: Dict[str, Any]
    request_batches: List[Tuple[int, Any]]


@dataclass(frozen=True)
class InflightPreparedRollout:
    """One launched rollout plan whose sampling requests are still inflight."""

    plan: PreparedRolloutPlan
    inflight_requests: Any

class AsyncPipelineRuntime:
    """Minimal producer-consumer state for the async training loop."""

    def __init__(
        self,
        *,
        max_inflight_rollouts: int = 1,
        initial_rollout_id: int = 0,
    ) -> None:
        del initial_rollout_id
        if max_inflight_rollouts < 1:
            raise ValueError(
                f"max_inflight_rollouts must be >= 1, got {max_inflight_rollouts}"
            )

        self.max_inflight_rollouts = int(max_inflight_rollouts)
        self._inflight: Dict[int, InflightRollout] = {}

    @property
    def inflight_count(self) -> int:
        return len(self._inflight)

    def can_launch(self) -> bool:
        return self.inflight_count < self.max_inflight_rollouts

    def launch_rollout(
        self,
        rollout_id: int,
        future: Any,
    ) -> InflightRollout:
        rid = int(rollout_id)
        if not self.can_launch():
            raise RuntimeError(
                "Async inflight queue full: "
                f"inflight={self.inflight_count}, max_inflight_rollouts={self.max_inflight_rollouts}"
            )
        if rid in self._inflight:
            raise RuntimeError(f"Rollout {rid} is already inflight")

        inflight = InflightRollout(rollout_id=rid, future=future)
        self._inflight[rid] = inflight
        return inflight

    def resolve_next_rollout(self, resolver: Callable[[Any], Any]) -> Rollout:
        if not self._inflight:
            raise RuntimeError("No inflight rollout to resolve")

        rid = min(self._inflight.keys())
        inflight = self._inflight.pop(rid)
        result = resolver(inflight.future)
        return Rollout(
            rollout_id=inflight.rollout_id,
            result=result,
        )

    def assert_no_inflight_for_weight_sync(self) -> None:
        if self._inflight:
            pending = sorted(self._inflight.keys())
            raise RuntimeError(
                "Weight sync requires empty inflight queue, but found pending rollouts: "
                f"{pending}"
            )


def train_async_loop(  # [PUBLIC-API → train()] async core loop
    *,
    args,
    rollout_services,
    rollout_eval_function,
    rollout_reward_hook,
    rollout_buffer,
    training_group,
    training_runtime,
    rollout_runtime,
    control_algorithm,
    wandb_logger: Optional[Any],
    should_save_fn: Callable[[int, Any], bool],
    should_eval_fn: Callable[[int, Any], bool],
    should_log_fn: Callable[[int, Any], bool],
    collect_rollout_batch_metrics_fn: Callable[[Any], dict],
    weight_sync: "WeightSyncCoordinator",
) -> None:
    """Asynchronous train loop with rollout/train overlap."""
    import ray

    from diffusionrl.rollout.base_types import RolloutContext, RolloutFunctionResult
    from diffusionrl.rollout.default_rollout import (
        finalize_default_rollout,
        prepare_default_rollout_plan,
    )
    from diffusionrl.rollout.primitives import (
        build_sampler_output_validator,
        launch_request_batches_async,
        resolve_request_batches_async,
    )

    logger.info("Starting async pipeline loop (separate mode)")
    rollout_update_interval = args.sync.rollout_update_interval
    rollout_on_gpu = True
    runtime = AsyncPipelineRuntime(
        max_inflight_rollouts=args.rollout.max_inflight_rollouts,
        initial_rollout_id=args.rollout.start_rollout_id,
    )
    next_rollout_to_launch = args.rollout.start_rollout_id

    def _sync_boundary_for(rollout_id: int) -> int:
        """Largest rollout id allowed before next weight sync boundary."""
        boundary = (
            (int(rollout_id) // rollout_update_interval) + 1
        ) * rollout_update_interval - 1
        return min(boundary, int(args.rollout.num_rollout) - 1)

    def _launch_rollout(rollout_id: int) -> None:
        if not runtime.can_launch():
            raise RuntimeError(
                f"Cannot launch rollout {rollout_id}: inflight queue is full "
                f"(inflight={runtime.inflight_count}, "
                f"max_inflight_rollouts={runtime.max_inflight_rollouts})"
            )
        should_log_rollout = should_log_fn(rollout_id, args)
        services = rollout_services
        context = RolloutContext(
            rollout_id=int(rollout_id),
            sde_indices=control_algorithm.resolve_rollout_sde_indices(
                current_step=int(rollout_id),
            ),
            collect_media_preview=bool(should_log_rollout and wandb_media_enabled),
            media_max_items=wandb_media_max_items,
        )
        batch, request_batches = prepare_default_rollout_plan(
            services=services,
        )
        plan = PreparedRolloutPlan(
            context=context,
            batch=batch,
            request_batches=request_batches,
        )

        def _launch_sampling_request(request):
            launched = services.launch_sampling_request(
                request=request,
                actor_group=rollout_services.get_sampling_group(),
                sde_indices=plan.context.sde_indices,
                requirements=services.sampling_requirements,
                sampling_overrides={
                    "collect_media_preview": bool(plan.context.collect_media_preview),
                },
            )
            return launched.request, launched

        inflight_rollout = InflightPreparedRollout(
            plan=plan,
            inflight_requests=launch_request_batches_async(
                request_batches=plan.request_batches,
                rollout_id=plan.context.rollout_id,
                launch_sampling_request=_launch_sampling_request,
            ),
        )
        runtime.launch_rollout(
            rollout_id,
            inflight_rollout,
        )

    def _fill_inflight_window(current_rollout: int) -> None:
        nonlocal next_rollout_to_launch
        if next_rollout_to_launch >= int(args.rollout.num_rollout):
            return

        launch_limit = _sync_boundary_for(current_rollout)
        while runtime.can_launch() and next_rollout_to_launch <= launch_limit:
            _launch_rollout(next_rollout_to_launch)
            next_rollout_to_launch += 1

    def _ensure_rollout_on_gpu() -> None:
        nonlocal rollout_on_gpu
        if (
            bool(args.ray.offload_rollout)
            and rollout_runtime is not None
            and not rollout_on_gpu
        ):
            rollout_runtime.wake_up()
            rollout_on_gpu = True

    wandb_media_enabled = wandb_logger is not None and args.logging.log_media
    wandb_media_max_items = max(1, int(args.logging.media_max_items))

    global_optimizer_step = 0
    for rollout_id in range(args.rollout.start_rollout_id, args.rollout.num_rollout):
        step_start_t = time.perf_counter()
        sync_result = None
        sync_phase_s = 0.0
        eval_phase_s = 0.0

        rollout_phase_start_t = time.perf_counter()
        _fill_inflight_window(rollout_id)

        resolved = runtime.resolve_next_rollout(lambda future: future)
        if resolved.rollout_id != rollout_id:
            raise RuntimeError(
                f"Async rollout ordering violated: expected rollout_id={rollout_id}, "
                f"got {resolved.rollout_id}"
            )
        inflight_rollout = resolved.result
        services = rollout_services
        request, sampler_outputs = resolve_request_batches_async(
            inflight_rollout=inflight_rollout.inflight_requests,
            resolve_sampling_request=lambda launched: services.resolve_launched_sampling_request(
                launched_request=launched,
            ),
            validate_sampler_outputs=build_sampler_output_validator(
                requirements=services.sampling_requirements,
                validation_config=services.sampler_validation_config,
            ),
        )
        plan = inflight_rollout.plan
        rollout_result = finalize_default_rollout(
            services=services,
            reward_hook=rollout_reward_hook,
            context=plan.context,
            batch=plan.batch,
            request_batches=plan.request_batches,
            request=request,
            sampler_outputs=sampler_outputs,
        )
        if not isinstance(rollout_result, RolloutFunctionResult):
            raise TypeError(
                "Default rollout finalizer must return RolloutFunctionResult. "
                f"Got type={type(rollout_result)}"
            )
        rollout_metadata = dict(rollout_result.metadata or {})
        advantages = services.compute_advantages(
            rewards=rollout_result.rewards,
            group_ids=rollout_result.request.meta.get("group_ids"),
            component_rewards=rollout_result.component_rewards,
        )
        training_batch = services.algorithm.assemble_training_batch(
            request=rollout_result.request,
            sampler_outputs=rollout_result.sampler_outputs,
            rewards=rollout_result.rewards,
            advantages=advantages,
            sde_indices=plan.context.sde_indices,
        )
        push_result = ray.get(
            rollout_buffer.push.remote(
                rollout_id=int(rollout_id),
                train_data=training_batch,
                metadata=dict(rollout_metadata or {}),
            )
        )
        if not push_result.get("accepted", False):
            raise RuntimeError(
                f"Rollout buffer rejected rollout_id={rollout_id}: {push_result.get('error')}"
            )
        rollout_payload = ray.get(
            rollout_buffer.pop_training_data.remote(
                expected_rollout_id=rollout_id,
            )
        )
        training_data_handle = rollout_payload.training_data
        rollout_metadata = dict(rollout_payload.metadata or {})
        sample_count = int(rollout_payload.sample_count or 0)
        rollout_phase_s = time.perf_counter() - rollout_phase_start_t

        should_sync = (rollout_id + 1) % rollout_update_interval == 0

        train_phase_start_t = time.perf_counter()
        metrics = training_group.train(rollout_id, training_data_handle)
        train_phase_s = time.perf_counter() - train_phase_start_t

        if should_save_fn(rollout_id, args):
            save_path = f"{args.rollout.output_dir}/checkpoint-{rollout_id}"
            training_runtime.save_model(save_path)
            logger.info("[async] Checkpoint saved: %s", save_path)

        if should_sync:
            runtime.assert_no_inflight_for_weight_sync()
            sync_phase_start_t = time.perf_counter()
            sync_result = weight_sync.sync(rollout_id=rollout_id)
            sync_phase_s = time.perf_counter() - sync_phase_start_t
            rollout_on_gpu = True

        if should_eval_fn(rollout_id, args):
            eval_phase_start_t = time.perf_counter()
            _ensure_rollout_on_gpu()
            eval_metrics = rollout_eval_function(
                services=rollout_services,
                reward_hook=rollout_reward_hook,
                rollout_id=int(rollout_id),
            )
            eval_phase_s = time.perf_counter() - eval_phase_start_t
            logger.info(
                "[async] Eval at %s: mean_reward=%.4f",
                rollout_id,
                eval_metrics["mean_reward"],
            )
            if wandb_logger is not None:
                wandb_logger.log_eval(rollout_id, eval_metrics)

        if wandb_logger is not None and metrics:
            per_step_list = metrics[0].get("_per_optimizer_step_metrics", [])
            for per_step_m in per_step_list:
                if per_step_m.get("has_backward", False):
                    global_optimizer_step += 1
                    wandb_step_m = {
                        k: v for k, v in per_step_m.items()
                        if k != "has_backward"
                    }
                    wandb_logger.log_step(global_optimizer_step, wandb_step_m)

        should_log_step = should_log_fn(rollout_id, args)
        if should_log_step:
            avg_loss = sum(m.get("loss", 0) for m in metrics) / max(len(metrics), 1)
            step_time_s = time.perf_counter() - step_start_t
            logger.info(
                "[async] Rollout %s: loss=%.4f rollout=%.3fs train=%.3fs sync=%.3fs eval=%.3fs step=%.3fs",
                rollout_id,
                avg_loss,
                rollout_phase_s,
                train_phase_s,
                sync_phase_s,
                eval_phase_s,
                step_time_s,
            )

            if wandb_logger is not None:
                aggregated = aggregate_metrics(metrics)
                aggregated["loss"] = avg_loss
                wandb_logger.log_rollout(rollout_id, aggregated)

                rollout_metrics = collect_rollout_batch_metrics_fn(training_data_handle)
                if rollout_metrics:
                    wandb_logger.log_rollout(rollout_id, rollout_metrics)
                media_preview = rollout_metadata.get("wandb_media_preview")
                if media_preview:
                    wandb_logger.log_generated_media(rollout_id, media_preview)

                perf_metrics = {
                    "rollout_phase_s": rollout_phase_s,
                    "train_phase_s": train_phase_s,
                    "sync_phase_s": sync_phase_s,
                    "eval_phase_s": eval_phase_s,
                    "step_time_s": step_time_s,
                    "samples_per_rollout": float(sample_count),
                    "samples_per_s": (
                        float(sample_count) / float(step_time_s)
                        if step_time_s > 0 and sample_count > 0
                        else 0.0
                    ),
                }
                wandb_logger.log_perf(rollout_id, perf_metrics)

                if sync_result is not None:
                    sync_metrics = build_sync_metrics(sync_result)
                    if sync_metrics:
                        wandb_logger.log_with_step(
                            step_key="rollout/step",
                            step=rollout_id,
                            metrics=sync_metrics,
                        )

                try:
                    buffer_stats = ray.get(rollout_buffer.get_stats.remote())
                except Exception as exc:
                    logger.warning("[async] Failed to fetch rollout buffer stats: %s", exc)
                else:
                    buffer_metrics = build_buffer_metrics(buffer_stats)
                    if buffer_metrics:
                        wandb_logger.log_with_step(
                            step_key="rollout/step",
                            step=rollout_id,
                            metrics=buffer_metrics,
                        )


def train(args, *, derived_config: DerivedConfig):  # [PUBLIC-API → main()] async entrypoint
    """Asynchronous training entrypoint."""
    validate_async_training_runner(
        args,
        rollout_info=derived_config.rollout_info,
    )

    import ray

    from diffusionrl.distributed.weight_sync import create_weight_sync
    from diffusionrl.ray.buffer_actor import create_buffer_actor
    from diffusionrl.ray.group_factory import create_rollout_actor_group, create_training_actor_group
    from diffusionrl.ray.group_runtime import RolloutGroupRuntime, TrainingGroupRuntime
    from diffusionrl.ray.placement_group import create_placement_groups_from_launch
    from diffusionrl.rollout.factory import (
        DEFAULT_EVAL_FUNCTION_PATH,
        DEFAULT_REWARD_HOOK_PATH,
        DEFAULT_ROLLOUT_FUNCTION_PATH,
        create_rollout_services,
    )
    from diffusionrl.utils import configure_logger, load_function, set_seed
    from diffusionrl.utils.wandb_logger import init_logger
    from diffusionrl.utils.wandb_metrics import compute_rollout_batch_metrics

    configure_logger()
    set_seed(args.seed)

    debug_mode = args.debug.mode
    launch_config = build_launch_config(
        args,
        derived_config=derived_config,
    )
    algorithm_init_payload = launch_config.algorithm_init_payload
    control_algorithm = create_algorithm_from_init_payload(algorithm_init_payload)
    rollout_info = launch_config.rollout_info
    training_actor_sampling_mode = rollout_info.training_actor_sampling_mode
    sync_mode = rollout_info.sync_protocol
    rollout_mode_name = rollout_info.mode

    if training_actor_sampling_mode:
        raise ValueError(
            "train_async.py requires a dedicated rollout engine and does not support training-actor direct sampling."
        )

    logger.info("Starting diffusionRL async training...")
    logger.info("Model: %s", args.model.pretrained_model_ckpt_path)
    logger.info("Algorithm: %s", algorithm_init_payload.component_dotpath)
    logger.info("Mode: %s", rollout_mode_name)
    logger.info("Weight sync mode: %s", sync_mode)
    logger.info(
        "Async controls: max_inflight_rollouts=%s rollout_update_interval=%s",
        args.rollout.max_inflight_rollouts,
        args.sync.rollout_update_interval,
    )
    logger.info(
        "Periodic controls: save_steps=%s eval_steps=%s logging_steps=%s",
        args.rollout.save_steps,
        args.evaluation.eval_steps,
        args.logging.logging_steps,
    )
    logger.info(
        "Debug flags: mode=%s save_intermediates=%s save_dir=%s",
        debug_mode,
        bool(args.debug.save_intermediates),
        args.debug.save_dir,
    )

    if not ray.is_initialized():
        if args.ray.ray_address:
            ray.init(address=args.ray.ray_address, ignore_reinit_error=True)
        else:
            ray.init()

    wandb_logger = None
    rollout_services = None
    rollout_function_dotpath = ""
    rollout_eval_function = None
    rollout_reward_hook = None
    rollout_buffer = None
    rollout_group = None
    rollout_runtime = None
    training_group = None
    training_runtime = None
    weight_sync = create_weight_sync(launch_config, mode=sync_mode)

    try:
        if args.logging.report_to_wandb and args.logging.project_name:
            wandb_tags = args.logging.tags
            wandb_logger = init_logger(
                project=args.logging.project_name,
                run_name=args.logging.run_name,
                config=build_derived_config_view(
                    args.to_dotted_dict(),
                    derived_config=derived_config,
                ),
                log_dir=args.logging.logging_dir,
                rank=0,
                tags=wandb_tags,
                entity=args.logging.entity or None,
                require_success=True,
            )
            if wandb_logger.initialized:
                logger.info(
                    "WandB initialized: project=%s, run=%s",
                    args.logging.project_name,
                    args.logging.run_name,
                )

        pgs = create_placement_groups_from_launch(launch_config)
        logger.info("Placement groups created")

        rollout_services = create_rollout_services(
            launch_config=launch_config,
        )
        dataset_step_info = compute_dataset_step_info(
            data_source=rollout_services.data_source,
            prompts_per_rollout=rollout_services.prompt_batch_size,
        )
        rollout_function_dotpath = args.rollout_function_dotpath or DEFAULT_ROLLOUT_FUNCTION_PATH
        rollout_eval_function = load_function(args.eval_function_dotpath or DEFAULT_EVAL_FUNCTION_PATH)
        rollout_reward_hook = load_function(args.reward_hook_dotpath or DEFAULT_REWARD_HOOK_PATH)
        logger.info("Rollout services created")
        
        if rollout_function_dotpath != DEFAULT_ROLLOUT_FUNCTION_PATH:
            raise ValueError(
                "train_async.py currently requires the default request-centric rollout function "
                "so the driver can overlap request launch and training. "
                f"Got custom rollout_function_dotpath={rollout_function_dotpath!r}."
            )
        if dataset_step_info.get("num_prompts", 0) > 0:
            logger.info(
                "Dataset step info: num_prompts=%s prompts_per_rollout=%s "
                "estimated_steps_per_dataset_pass=%s steps_before_reset=%s",
                dataset_step_info.get("num_prompts"),
                dataset_step_info.get("prompts_per_rollout"),
                dataset_step_info.get("estimated_steps_per_dataset_pass"),
                dataset_step_info.get("steps_before_reset"),
            )
            if not dataset_step_info.get("exact_dataset_pass_per_cycle", False):
                logger.warning(
                    "Inexact dataset pass: %s prompts will be dropped per cycle "
                    "(drop_last=%s).",
                    dataset_step_info.get("remainder_prompts"),
                    dataset_step_info.get("drop_last"),
                )

        rollout_pgs = pgs.get("rollout")
        if rollout_pgs is None:
            raise ValueError("Missing rollout placement-group allocation.")
        rollout_group = create_rollout_actor_group(launch_config, rollout_pgs)
        rollout_runtime = RolloutGroupRuntime.from_group(rollout_group)
        rollout_services.attach_sampling_group(rollout_group)
        logger.info("Rollout actor group created and attached to rollout services")

        training_pgs = pgs.get("training")
        if training_pgs is None:
            raise ValueError("Missing training placement-group allocation.")
        training_group = create_training_actor_group(
            launch_config,
            training_pgs,
        )
        training_runtime = TrainingGroupRuntime.from_group(training_group)
        resume_from_checkpoint = args.training.resume_from_checkpoint
        if resume_from_checkpoint:
            training_runtime.load_checkpoint(resume_from_checkpoint)
            logger.info("Checkpoint loaded: %s", resume_from_checkpoint)
            restored_rollout_id = maybe_restore_start_rollout_id_from_checkpoint(
                args,
                resume_from_checkpoint,
            )
            if restored_rollout_id is not None:
                logger.info(
                    "Auto-set start_rollout_id=%s from checkpoint path.",
                    restored_rollout_id,
                )
        train_backend_info = training_runtime.get_train_backend_info()
        logger.info("Training actor group created")
        if train_backend_info:
            logger.info("Training backend: %s", train_backend_info)

        training_runtime.update_weights()
        logger.info("Initial weights synchronized")

        rollout_buffer = create_buffer_actor(args)
        logger.info("Rollout buffer actor created")

        weight_sync.setup(
            training_runtime=training_runtime,
            rollout_runtime=rollout_runtime,
        )

        train_async_loop(
            args=args,
            rollout_services=rollout_services,
            rollout_eval_function=rollout_eval_function,
            rollout_reward_hook=rollout_reward_hook,
            rollout_buffer=rollout_buffer,
            training_group=training_group,
            training_runtime=training_runtime,
            rollout_runtime=rollout_runtime,
            control_algorithm=control_algorithm,
            wandb_logger=wandb_logger,
            should_save_fn=should_save,
            should_eval_fn=should_eval,
            should_log_fn=should_log,
            collect_rollout_batch_metrics_fn=lambda batch_ref: collect_rollout_batch_metrics(
                ray_module=ray,
                batch_ref=batch_ref,
                compute_rollout_batch_metrics_fn=compute_rollout_batch_metrics,
            ),
            weight_sync=weight_sync,
        )
    finally:
        weight_sync.teardown()
        if rollout_buffer is not None:
            try:
                ray.get(rollout_buffer.dispose.remote())
            finally:
                ray.kill(rollout_buffer)
        if rollout_services is not None:
            rollout_services.dispose()
        if rollout_group is not None:
            rollout_group.dispose()
        if training_group is not None:
            training_group.dispose()
        if wandb_logger is not None:
            wandb_logger.finish()

    logger.info("Async training complete!")


def main(argv=None):  # [PUBLIC-API → __main__] async CLI entrypoint
    args, derived_config = parse_args_with_derived_config(argv)
    train(args, derived_config=derived_config)


if __name__ == "__main__":
    main()
