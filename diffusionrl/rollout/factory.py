"""Driver-side rollout service construction."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from diffusionrl.algorithms.construction import instantiate_algorithm_from_config
from diffusionrl.config.launch_resolution import LaunchConfig
from diffusionrl.config.resolution import collect_sampling_requirements, derive_rollout_topology
from diffusionrl.reward.factory import create_driver_reward_executor
from diffusionrl.reward.schema import RewardSchema
from diffusionrl.rollout.service_interface import RolloutServices, compute_dataset_step_info
from diffusionrl.utils import load_function

DEFAULT_ROLLOUT_FUNCTION_PATH = "diffusionrl.rollout.default_rollout.generate_rollout"
DEFAULT_EVAL_FUNCTION_PATH = "diffusionrl.rollout.default_rollout.evaluate_rollout"
DEFAULT_REWARD_HOOK_PATH = "diffusionrl.rollout.default_rollout.score_rewards_hook"


def create_rollout_services(
    args,
    *,
    reward_pg_result: Optional[Any] = None,
    launch_config: LaunchConfig,
) -> Tuple[RolloutServices, Dict[str, Any]]:
    """Create rollout services and return dataset-step info for the driver."""
    if not isinstance(launch_config, LaunchConfig):
        raise ValueError(
            "create_rollout_services requires LaunchConfig to be built by the driver."
        )

    algorithm = instantiate_algorithm_from_config(dict(launch_config.algorithm_config))
    sampling_requirements = collect_sampling_requirements(algorithm=algorithm)
    sampling_config = dict(launch_config.training_sampling_config)
    rollout_topology = derive_rollout_topology(args)
    reward_schema = RewardSchema.from_args(args)
    reward_service = create_driver_reward_executor(
        reward_schema,
        reward_pg_result=reward_pg_result,
    )
    if reward_service is None and not reward_schema.uses_sampling_actor_execution:
        raise RuntimeError("Driver failed to initialize driver-side reward service.")

    try:
        data_source_cls = load_function(args.data_source_path)
        data_source = data_source_cls(args)
    except Exception as exc:
        raise RuntimeError(f"Failed to load data source: {exc}") from exc

    reward_scoring_mode = (
        "precomputed"
        if reward_schema.uses_sampling_actor_execution
        else "service"
    )
    prompt_batch_size = int(
        getattr(algorithm, "prompts_per_rollout", args.algorithm.prompts_per_rollout)
    )
    sampler_validation_config = algorithm.get_sampler_validation_config(args=args)
    if not isinstance(sampler_validation_config, dict):
        sampler_validation_config = {}

    services = RolloutServices(
        algorithm=algorithm,
        reward_scoring_mode=reward_scoring_mode,
        reward_service=reward_service,
        data_source=data_source,
        is_direct_sampling_mode=rollout_topology.training_actor_sampling_mode,
        max_samples_per_request=args.sampling.max_samples_per_request,
        reward_component_weights=reward_schema.component_weights(),
        prompt_batch_size=prompt_batch_size,
        evaluation_settings=args.rollout.evaluation,
        sampler_validation_config=sampler_validation_config,
        sampling_config=sampling_config,
        sampling_requirements=sampling_requirements,
        debug_mode=str(args.debug.debug_mode or "none"),
        debug_output_dir=getattr(args.debug, "debug_output_dir", None),
    )
    dataset_step_info = compute_dataset_step_info(
        data_source=services.data_source,
        prompts_per_rollout=services.prompt_batch_size,
    )
    return services, dataset_step_info


__all__ = [
    "DEFAULT_EVAL_FUNCTION_PATH",
    "DEFAULT_REWARD_HOOK_PATH",
    "DEFAULT_ROLLOUT_FUNCTION_PATH",
    "create_rollout_services",
]
