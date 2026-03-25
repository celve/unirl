"""Driver-side rollout runtime owner."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional, Tuple

from diffusionrl.algorithms.construction import instantiate_algorithm_from_config
from diffusionrl.config.launch_resolution import ResolvedLaunchConfig
from diffusionrl.config.resolution import collect_sampling_requirements, derive_rollout_topology
from diffusionrl.reward.factory import create_driver_reward_executor
from diffusionrl.reward.schema import RewardSchema
from diffusionrl.rollout.service_interface import RolloutServices, compute_dataset_step_info
from diffusionrl.utils import load_function

logger = logging.getLogger(__name__)

DEFAULT_ROLLOUT_FUNCTION_PATH = "diffusionrl.rollout.default_rollout.generate_rollout"
DEFAULT_EVAL_FUNCTION_PATH = "diffusionrl.rollout.default_rollout.evaluate_rollout"
DEFAULT_REWARD_HOOK_PATH = "diffusionrl.rollout.default_rollout.score_rewards_hook"


@dataclass
class DriverRolloutRuntime:
    """Driver-side owner for rollout services, hooks, and attached sampling resources."""

    args: Any
    services: RolloutServices
    reward_service: Any
    reward_schema: RewardSchema
    rollout_function: Any
    rollout_function_path: str
    eval_function: Any
    eval_function_path: str
    reward_hook: Any
    reward_hook_path: str
    sampling_group: Any = None

    def attach_sampling_group(self, actor_group: Any) -> None:
        self.sampling_group = actor_group
        self.services.attach_sampling_group(actor_group)

    def get_sampling_group(self) -> Any:
        if self.sampling_group is None:
            raise RuntimeError("No sampling group attached for driver rollout runtime.")
        return self.sampling_group

    def uses_default_rollout_function(self) -> bool:
        return str(self.rollout_function_path).strip() == DEFAULT_ROLLOUT_FUNCTION_PATH

    def get_dataset_step_info(self) -> Dict[str, Any]:
        return compute_dataset_step_info(
            data_source=self.services.data_source,
            prompts_per_rollout=self.services.prompt_batch_size,
        )

    def dispose(self) -> None:
        if self.reward_service is not None:
            self.reward_service.dispose()
            self.reward_service = None
        self.sampling_group = None
        self.services.attach_sampling_group(None)


def create_driver_rollout_runtime(
    args,
    *,
    reward_pg_result: Optional[Any] = None,
    launch_config: ResolvedLaunchConfig,
) -> Tuple[DriverRolloutRuntime, Dict[str, Any]]:
    """Create the driver-side rollout runtime and return dataset-step info."""
    if not isinstance(launch_config, ResolvedLaunchConfig):
        raise ValueError(
            "create_driver_rollout_runtime requires ResolvedLaunchConfig to be built by the driver."
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
        raise RuntimeError("Driver rollout runtime failed to initialize driver-side reward service.")

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
    rollout_function_path = str(args.rollout_function_path)
    eval_function_path = str(args.eval_function_path)
    reward_hook_path = str(args.reward_hook_path)
    rollout_function = load_function(rollout_function_path)
    eval_function = load_function(eval_function_path)
    reward_hook = load_function(reward_hook_path)
    runtime = DriverRolloutRuntime(
        args=args,
        services=services,
        reward_service=reward_service,
        reward_schema=reward_schema,
        rollout_function=rollout_function,
        rollout_function_path=rollout_function_path,
        eval_function=eval_function,
        eval_function_path=eval_function_path,
        reward_hook=reward_hook,
        reward_hook_path=reward_hook_path,
    )
    return runtime, runtime.get_dataset_step_info()


__all__ = [
    "DEFAULT_EVAL_FUNCTION_PATH",
    "DEFAULT_REWARD_HOOK_PATH",
    "DEFAULT_ROLLOUT_FUNCTION_PATH",
    "DriverRolloutRuntime",
    "create_driver_rollout_runtime",
]
