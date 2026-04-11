"""Driver-side rollout service construction."""

from __future__ import annotations

from diffusionrl.algorithms.construction import create_algorithm_from_init_payload
from diffusionrl.config.launch_resolution import LaunchConfig
from diffusionrl.config.resolution import collect_sampling_requirements, derive_rollout_topology
from diffusionrl.reward.schema import RewardSchema
from diffusionrl.rollout.service_interface import RolloutServices
from diffusionrl.utils import load_function

DEFAULT_ROLLOUT_FUNCTION_PATH = "diffusionrl.rollout.default_rollout.generate_rollout"
DEFAULT_EVAL_FUNCTION_PATH = "diffusionrl.rollout.default_rollout.evaluate_rollout"
DEFAULT_REWARD_HOOK_PATH = "diffusionrl.rollout.default_rollout.score_rewards_hook"


def create_rollout_services(
    args,
    *,
    launch_config: LaunchConfig,
) -> RolloutServices:
    """Create rollout services and return dataset-step info for the driver."""
    if not isinstance(launch_config, LaunchConfig):
        raise ValueError(
            "create_rollout_services requires LaunchConfig to be built by the driver."
        )

    algorithm = create_algorithm_from_init_payload(launch_config.algorithm_init_payload)
    sampling_requirements = collect_sampling_requirements(algorithm=algorithm)
    sampling_config = dict(launch_config.training_sampling_config)
    rollout_topology = derive_rollout_topology(args)
    reward_schema = RewardSchema.from_args(args)

    try:
        data_source_cls = load_function(args.data_source_dotpath)
        data_source = data_source_cls(args)
    except Exception as exc:
        raise RuntimeError(f"Failed to load data source: {exc}") from exc

    prompt_batch_size = int(
        getattr(algorithm, "prompts_per_rollout", args.algorithm.prompts_per_rollout)
    )
    sampler_validation_config = algorithm.get_sampler_validation_config(
        allow_replay=bool(launch_config.rollout_mode_info.replay_enabled)
    )
    if not isinstance(sampler_validation_config, dict):
        sampler_validation_config = {}

    services = RolloutServices(
        algorithm=algorithm,
        data_source=data_source,
        is_direct_sampling_mode=rollout_topology.training_actor_sampling_mode,
        max_samples_per_request=args.sampling.max_samples_per_request,
        reward_component_weights=reward_schema.component_weights(),
        prompt_batch_size=prompt_batch_size,
        evaluation_settings=args.evaluation,
        sampler_validation_config=sampler_validation_config,
        sampling_config=sampling_config,
        sampling_requirements=sampling_requirements,
        debug_mode=str(args.debug.debug_mode or "none"),
        debug_output_dir=getattr(args.debug, "debug_output_dir", None),
    )

    return services


__all__ = [
    "DEFAULT_EVAL_FUNCTION_PATH",
    "DEFAULT_REWARD_HOOK_PATH",
    "DEFAULT_ROLLOUT_FUNCTION_PATH",
    "create_rollout_services",
]
