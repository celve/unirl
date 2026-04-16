"""Driver-side rollout service construction."""

from __future__ import annotations

from diffusionrl.algorithms.construction import create_algorithm_from_init_payload
from diffusionrl.config import LaunchConfig
from diffusionrl.rollout.service_interface import RolloutServices
from diffusionrl.utils import load_function

DEFAULT_ROLLOUT_FUNCTION_PATH = "diffusionrl.rollout.default_rollout.generate_rollout"
DEFAULT_EVAL_FUNCTION_PATH = "diffusionrl.rollout.default_rollout.evaluate_rollout"
DEFAULT_REWARD_HOOK_PATH = "diffusionrl.rollout.default_rollout.score_rewards_hook"


def create_rollout_services(
    *,
    launch_config: LaunchConfig,
) -> RolloutServices:
    """Create rollout services and return dataset-step info for the driver."""
    if not isinstance(launch_config, LaunchConfig):
        raise ValueError(
            "create_rollout_services requires LaunchConfig to be built by the driver."
        )

    spec = launch_config.rollout_services
    algorithm = create_algorithm_from_init_payload(launch_config.algorithm_init_payload)
    sampling_requirements = algorithm.get_sampling_requirements()
    sampling_config = dict(launch_config.training_sampling_config)
    rollout_info = launch_config.rollout_info

    try:
        data_source_cls = load_function(spec.data_source_dotpath)
        data_source = data_source_cls(spec.data_source_args)
    except Exception as exc:
        raise RuntimeError(f"Failed to load data source: {exc}") from exc

    prompt_batch_size = int(
        getattr(algorithm, "prompts_per_rollout", spec.prompts_per_rollout)
    )
    sampler_validation_config = algorithm.get_sampler_validation_config(
        allow_replay=spec.replay_enabled,
    )
    if not isinstance(sampler_validation_config, dict):
        sampler_validation_config = {}

    services = RolloutServices(
        algorithm=algorithm,
        data_source=data_source,
        is_direct_sampling_mode=rollout_info.training_actor_sampling_mode,
        max_samples_per_request=spec.max_samples_per_request,
        reward_component_weights=spec.reward_spec.component_weights(),
        prompt_batch_size=prompt_batch_size,
        evaluation_settings=spec.evaluation_settings,
        sampler_validation_config=sampler_validation_config,
        sampling_config=sampling_config,
        sampling_requirements=sampling_requirements,
        debug_mode=spec.debug_mode,
        debug_output_dir=spec.debug_output_dir,
    )

    return services


__all__ = [
    "DEFAULT_EVAL_FUNCTION_PATH",
    "DEFAULT_REWARD_HOOK_PATH",
    "DEFAULT_ROLLOUT_FUNCTION_PATH",
    "create_rollout_services",
]
