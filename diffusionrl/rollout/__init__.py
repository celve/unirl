"""Rollout extension surface and default rollout implementations."""

from diffusionrl.rollout.base_types import (
    RewardHookResult,
    RolloutContext,
    RolloutFunctionResult,
)
from diffusionrl.rollout.default_rollout import (
    evaluate_rollout,
    generate_rollout,
    score_rewards_hook,
)
from diffusionrl.rollout.driver_runtime import (
    DEFAULT_EVAL_FUNCTION_PATH,
    DEFAULT_REWARD_HOOK_PATH,
    DEFAULT_ROLLOUT_FUNCTION_PATH,
    DriverRolloutRuntime,
    create_driver_rollout_runtime,
)
from diffusionrl.rollout.primitives import (
    build_rollout_request,
    estimate_request_batches,
    execute_request_batches,
    launch_request_batches_async,
    plan_request_batches,
    resolve_request_batches_async,
    validate_sampler_outputs_against_contract,
)
from diffusionrl.rollout.service_interface import RolloutServices

__all__ = [
    "DEFAULT_EVAL_FUNCTION_PATH",
    "DEFAULT_REWARD_HOOK_PATH",
    "DEFAULT_ROLLOUT_FUNCTION_PATH",
    "DriverRolloutRuntime",
    "RewardHookResult",
    "RolloutContext",
    "RolloutFunctionResult",
    "RolloutServices",
    "build_rollout_request",
    "create_driver_rollout_runtime",
    "evaluate_rollout",
    "estimate_request_batches",
    "execute_request_batches",
    "generate_rollout",
    "launch_request_batches_async",
    "plan_request_batches",
    "resolve_request_batches_async",
    "score_rewards_hook",
    "validate_sampler_outputs_against_contract",
]
