"""Configuration validation for typed config objects and runtime payloads."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from enum import Enum
from typing import Any, Dict, Optional

from diffusionrl.algorithms.base import BaseAlgorithmConfig
from diffusionrl.config.assembly import LaunchConfig
from diffusionrl.config.resolution import DIRECT_ROLLOUT_MODE, rollout_mode_is_colocated
from diffusionrl.config.spec import RolloutInfo, TrainingPlan
from diffusionrl.config.training_sections import (
    LrSchedulerConfig,
    OptimizerConfig,
    TrainingExecutionConfig,
)
from diffusionrl.construction import ComponentInitPayload
from diffusionrl.ray.actor_config import (
    RolloutActorConfig,
    TrainingActorConfig,
)
from diffusionrl.reward.config import RewardSpec
from diffusionrl.training.types import (
    BaseTrainBackendConfig,
    TrainTopology,
    supported_train_backends,
)
from diffusionrl.types.engine import EngineConfig, uses_dedicated_rollout_engine
from diffusionrl.types.sampling import SamplingParams
from diffusionrl.utils.misc import load_function

logger = logging.getLogger(__name__)


# ============================================================================
# Common validation primitives
# ============================================================================

ENV_REPO_ROOT = "DIFFUSIONRL_REPO_ROOT"


class PrecisionName(str, Enum):
    BF16 = "bf16"
    BFLOAT16 = "bfloat16"
    FP16 = "fp16"
    FLOAT16 = "float16"
    HALF = "half"
    FP32 = "fp32"
    FLOAT32 = "float32"
    FLOAT = "float"


# Aliases accepted for optional dtype fields (e.g. rollout transport cast-off).
_PRECISION_DISABLE_ALIASES = frozenset(
    {"", "none", "off", "disable", "disabled"}
)


def repo_root(*, env_repo_root: str) -> str:
    """Resolve repository root from environment override or package-relative path."""
    env_root = os.getenv(env_repo_root)
    if env_root:
        return os.path.abspath(os.path.expanduser(env_root))
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def is_probably_local_weight_sync_dir(path: str, *, root: str) -> bool:
    """Best-effort guard for local-only paths in multi-node checkpoint sync."""
    if not path:
        return True
    real = os.path.realpath(path)
    for prefix in ("/tmp", "/var/tmp", "/dev/shm"):
        if real == prefix or real.startswith(prefix + os.sep):
            return True
    if real == root or real.startswith(root + os.sep):
        return True
    return False


def validate_dotpath(path: str, *, label: str) -> None:
    """Fail fast when a configured dotpath is not importable."""
    try:
        load_function(path)
    except Exception as exc:
        raise ValueError(
            f"Invalid {label} path: {path!r}. Import failed: {exc}. "
            f"Check that the module is installed and the dotpath is correct "
            f"(e.g. 'diffusionrl.algorithms.grpo.GRPOAlgorithm')."
        ) from exc


def validate_precision_type(
    value: Any,
    *,
    field_name: str,
    allow_disable_aliases: bool = False,
) -> None:
    """Validate precision aliases used by config-facing precision fields."""
    key = str(value or "").strip().lower()
    if allow_disable_aliases and key in _PRECISION_DISABLE_ALIASES:
        return
    try:
        PrecisionName(key)
    except ValueError as exc:
        if allow_disable_aliases:
            raise ValueError(
                f"{field_name} must be one of "
                "fp32/fp16/bf16 or none/off (disable), "
                f"got: {value!r}"
            ) from exc
        raise ValueError(
            f"{field_name} must be one of bf16/fp16/fp32, got: {value!r}"
        ) from exc


def validate_colocate_fractions(
    *,
    colocate_training_gpu_fraction: float,
    colocate_rollout_gpu_fraction: float,
) -> None:
    """Validate colocate GPU fraction bounds."""
    if colocate_training_gpu_fraction <= 0 or colocate_rollout_gpu_fraction <= 0:
        raise ValueError(
            "colocate_training_gpu_fraction and colocate_rollout_gpu_fraction must be > 0"
        )
    if colocate_training_gpu_fraction + colocate_rollout_gpu_fraction > 1.0:
        raise ValueError(
            "colocate_training_gpu_fraction + colocate_rollout_gpu_fraction must be <= 1.0"
        )


def validate_reward_config(reward_config: RewardSpec) -> None:
    """Validate reward configuration consistency from the typed reward spec."""
    execution_plan = reward_config.to_execution_plan()
    if execution_plan.uses_http_backend:
        logger.info("Reward mode: sampling-actor HTTP (external service)")
    else:
        logger.info(
            "Reward mode: sampling-actor-local worker (local_reward_device=%s)",
            execution_plan.local_device,
        )


# ============================================================================
# Typed spec / declaration validation
# ============================================================================


def validate_engine_algorithm_contract(
    *,
    algorithm_type: str,
    rollout_info: RolloutInfo,
    effective_engine_capabilities: Optional[Dict[str, bool]],
    sampling_requirements: SamplingRequirements,
) -> None:
    """Validate engine/algorithm compatibility using pre-resolved capabilities."""
    if rollout_info.training_actor_sampling_mode:
        return

    rollout_engine = rollout_info.rollout_engine
    if not rollout_engine:
        raise ValueError(
            "Dedicated rollout validation requires rollout.rollout_engine to be set explicitly. "
            "Run validate_args() before resolving dedicated rollout engine capabilities."
        )

    if effective_engine_capabilities is None:
        raise ValueError(
            "Dedicated rollout validation requires resolved engine capabilities. "
            "Run derive_config() before validate_args()."
        )

    required_dict = sampling_requirements.to_dict()
    missing = [
        key
        for key, needed in required_dict.items()
        if bool(needed) and not bool(effective_engine_capabilities.get(key, False))
    ]
    if missing:
        raise ValueError(
            f"Engine capability mismatch for algorithm_type={algorithm_type}: "
            f"rollout.rollout_engine={rollout_engine} lacks {missing}. "
            f"engine_capabilities={effective_engine_capabilities}, required={required_dict}. "
            "Use a compatible dedicated rollout engine, or fall back to direct "
            "training-actor sampling for trajectory/log-prob-heavy algorithms."
        )


def validate_training_actor_sampling_mode(
    *,
    rollout_info: RolloutInfo,
    backend_capabilities: Mapping[str, Any],
    backend_name: str,
) -> None:
    """Validate direct-sampling topology compatibility."""
    if not rollout_info.training_actor_sampling_mode:
        return

    if rollout_info.mode != DIRECT_ROLLOUT_MODE or uses_dedicated_rollout_engine(rollout_info.rollout_engine):
        raise ValueError(
            "Dedicated rollout engines cannot use direct_sampling mode. "
            f"Got rollout.mode={rollout_info.mode!r}, "
            f"rollout.rollout_engine={rollout_info.rollout_engine!r}."
        )

    if not bool(backend_capabilities.get("supports_training_actor_sampling", False)):
        raise ValueError(
            "rollout.mode=%r resolves to direct training-actor sampling, "
            "but train_backend=%r does not declare supports_training_actor_sampling=true."
            % (rollout_info.mode, backend_name)
        )


# ============================================================================
# Resolved rollout / launch validation
# ============================================================================


def validate_direct_sampling_batch_geometry(
    *,
    rollout_info: RolloutInfo,
    max_samples_per_request: Optional[int],
) -> None:
    """Validate prompt-batch splitting for training-actor direct sampling."""
    if max_samples_per_request is None:
        return

    if not rollout_info.training_actor_sampling_mode:
        raise ValueError(
            "sampling.max_samples_per_request is only valid when "
            "sampling runs directly on training actors."
        )

    max_samples_per_request = int(max_samples_per_request)
    if max_samples_per_request < 1:
        raise ValueError("sampling.max_samples_per_request must be >= 1.")


def validate_weight_sync(
    *,
    rollout_info: RolloutInfo,
    sync_protocol: str,
    sync_dir: str,
    rollout_num_nodes: int,
    training_num_nodes: int,
) -> None:
    """Validate explicit weight-sync protocol against rollout topology."""
    rollout_engine = rollout_info.rollout_engine
    resolved_mode = str(sync_protocol or "").strip().lower()
    if rollout_info.training_actor_sampling_mode:
        if resolved_mode != "disabled":
            raise ValueError(
                "direct training-actor sampling requires sync.protocol='disabled'. "
                f"Got sync.protocol={resolved_mode!r}."
            )
        return
    if resolved_mode == "disabled":
        raise ValueError(
            "sync.protocol='disabled' is only valid when sampling runs directly on training actors. "
            f"Got rollout.rollout_engine={rollout_engine!r}."
        )
    is_multi_node = (
        int(rollout_num_nodes) > 1
        or int(training_num_nodes) > 1
    )
    if resolved_mode in {"tensor_payload", "nccl_broadcast"} and rollout_engine != "sglang":
        raise ValueError(
            "sync.protocol in {tensor_payload,nccl_broadcast} currently requires "
            "rollout.rollout_engine='sglang'. "
            f"Got rollout.rollout_engine={rollout_engine!r}."
        )

    if (
        resolved_mode == "checkpoint_path"
        and is_multi_node
        and is_probably_local_weight_sync_dir(
            sync_dir,
            root=repo_root(env_repo_root=ENV_REPO_ROOT),
        )
    ):
        raise ValueError(
            "sync.protocol=checkpoint_path in multi-node mode requires a shared filesystem path. "
            f"Got local-only sync.dir={sync_dir}. "
            "Use a shared mount (e.g. /mnt/shared/... or NFS path)."
        )


def validate_rollout_layout(
    *,
    rollout_info: RolloutInfo,
    rollout_num_nodes: int,
    rollout_num_gpus_per_node: int,
    training_num_nodes: int,
    training_num_gpus_per_node: int,
    rollout_num_gpus_per_actor: int,
    allow_noset_multi_gpu_inference: bool,
) -> None:
    """Validate rollout actor GPU layout and colocate constraints."""
    if rollout_info.training_actor_sampling_mode:
        return

    if rollout_info.mode == DIRECT_ROLLOUT_MODE:
        raise ValueError(
            "Dedicated rollout actor layout validation only applies to dedicated rollout engines. "
            f"Got rollout.mode={rollout_info.mode!r}."
        )
    rollout_gpus = int(rollout_num_gpus_per_actor or 0)
    rollout_total_gpus = int(rollout_num_nodes) * int(rollout_num_gpus_per_node)
    if rollout_mode_is_colocated(rollout_info.mode):
        training_total_gpus = int(training_num_nodes) * int(training_num_gpus_per_node)
        rollout_gpu_pool_size = max(rollout_total_gpus, training_total_gpus)
    else:
        rollout_gpu_pool_size = rollout_total_gpus
    if rollout_gpu_pool_size < 1:
        raise ValueError(
            "Dedicated rollout actors require a positive rollout GPU pool from placement config. "
            f"Got rollout_num_nodes={rollout_num_nodes}, "
            f"rollout_num_gpus_per_node={rollout_num_gpus_per_node}, "
            f"training_num_nodes={training_num_nodes}, "
            f"training_num_gpus_per_node={training_num_gpus_per_node}."
        )
    if rollout_gpu_pool_size < rollout_gpus:
        raise ValueError(
            "Dedicated rollout placement does not have enough GPUs for one rollout actor. "
            f"Available rollout GPU pool={rollout_gpu_pool_size}, "
            f"rollout.num_gpus_per_actor={rollout_gpus}."
        )
    if rollout_gpus > 1 and rollout_gpu_pool_size % rollout_gpus != 0:
        raise ValueError(
            "Dedicated rollout GPU pool must be divisible by rollout.num_gpus_per_actor "
            "for multi-GPU rollout actors. "
            f"Available rollout GPU pool={rollout_gpu_pool_size}, "
            f"rollout.num_gpus_per_actor={rollout_gpus}."
        )
    is_sglang_engine = rollout_info.is_sglang_engine
    if rollout_gpus > 1 and rollout_mode_is_colocated(rollout_info.mode) and not is_sglang_engine:
        raise ValueError(
            "colocate with multi-GPU rollout actors is only supported "
            "for rollout.rollout_engine='sglang'."
        )
    if rollout_gpus > 1:
        if not allow_noset_multi_gpu_inference:
            if rollout_mode_is_colocated(rollout_info.mode) and is_sglang_engine:
                raise ValueError(
                    "sglang colocate with multi-GPU rollout requires NOSET actor layout. "
                    "Set --ray.allow-noset-multi-gpu-inference=true."
                )
            raise ValueError(
                "multi-GPU rollout actor layout requires "
                "--ray.allow-noset-multi-gpu-inference=true. "
                "Default layout keeps integer single-GPU actors."
            )
        logger.warning(
            "allow_noset_multi_gpu_inference=true enabled. "
            "This is an experimental actor layout and is not part of the default path."
        )


# ============================================================================
# Resolved backend validation
# ============================================================================


def validate_train_backend_config(
    *,
    train_backend_config: BaseTrainBackendConfig,
) -> None:
    """Validate cross-domain backend constraints after canonicalization.

    New-style configs (``FSDPBackendConfig``, ``VeOmniBackendConfig``) carry
    no registry metadata; the identifier check runs upstream in
    ``resolve_train_backend_identifier``. Kept as a placeholder for future
    cross-domain invariants.
    """
    del train_backend_config


# ============================================================================
# Resolved training geometry validation
# ============================================================================


def validate_training_batch_geometry(
    *,
    prompts_per_rollout: int,
    samples_per_prompt: int,
    global_batch_size: int,
    num_updates_per_batch: int,
    micro_batch_size: Optional[int],
    topology: Optional[TrainTopology] = None,
) -> None:
    """Validate batch-geometry invariants using resolved training geometry."""
    prompts_per_rollout = int(prompts_per_rollout)
    samples_per_prompt = int(samples_per_prompt)
    if samples_per_prompt < 1:
        raise ValueError(f"algorithm.samples_per_prompt must be >= 1. Got {samples_per_prompt}.")

    num_updates_per_batch = int(num_updates_per_batch)
    global_batch_size = int(global_batch_size)
    if topology is None:
        raise ValueError(
            "validate_training_batch_geometry requires a resolved TrainTopology."
        )
    dp_size = int(topology.dp_size)
    dp_replicate_size = int(topology.dp_replicate_size)
    raw_micro_batch_size = micro_batch_size

    def _format_geometry(
        *,
        local_batch_size: Optional[int],
        update_batch_size: Optional[int],
        micro_batch_size: Optional[int],
    ) -> str:
        local_text = str(local_batch_size) if local_batch_size is not None else "<not divisible by dp_size>"
        update_text = (
            str(update_batch_size)
            if update_batch_size is not None
            else "<not divisible by num_updates_per_batch>"
        )
        if raw_micro_batch_size is None:
            micro_text = "auto"
            if micro_batch_size is not None:
                micro_text = f"auto (= {micro_batch_size})"
        else:
            micro_text = str(micro_batch_size if micro_batch_size is not None else raw_micro_batch_size)
        return "\n".join(
            [
                "Resolved training batch geometry:",
                f"  global_batch_size = prompts_per_rollout({prompts_per_rollout}) * "
                f"samples_per_prompt({samples_per_prompt}) = {global_batch_size}",
                f"  local_batch_size = global_batch_size / dp_size({dp_size}) = {local_text}",
                f"  local_mini_batch_size = local_batch_size / "
                f"num_updates_per_batch({num_updates_per_batch}) = {update_text}",
                f"  micro_batch_size = {micro_text}",
                f"  dp_replicate_size = {dp_replicate_size}",
            ]
        )

    def _raise_geometry_error(
        *,
        reason: str,
        fix_hint: str,
        local_batch_size: Optional[int],
        update_batch_size: Optional[int],
        micro_batch_size: Optional[int],
    ) -> None:
        raise ValueError(
            "\n".join(
                [
                    f"Invalid training batch geometry: {reason}",
                    _format_geometry(
                        local_batch_size=local_batch_size,
                        update_batch_size=update_batch_size,
                        micro_batch_size=micro_batch_size,
                    ),
                    f"Fix: {fix_hint}",
                ]
            )
        )

    if global_batch_size % dp_size != 0:
        _raise_geometry_error(
            reason="global rollout batch cannot be split evenly across training DP ranks "
            "(global_batch_size % dp_size != 0).",
            fix_hint="Adjust algorithm.prompts_per_rollout, algorithm.samples_per_prompt, "
            "or the training backend dp_size.",
            local_batch_size=None,
            update_batch_size=None,
            micro_batch_size=None,
        )
    local_batch_size = int(global_batch_size // dp_size)

    if global_batch_size % dp_replicate_size != 0:
        _raise_geometry_error(
            reason="global rollout batch must also be divisible by dp_replicate_size "
            "(global_batch_size % dp_replicate_size != 0).",
            fix_hint="Adjust algorithm.prompts_per_rollout, algorithm.samples_per_prompt, "
            "or the backend replicate topology.",
            local_batch_size=local_batch_size,
            update_batch_size=None,
            micro_batch_size=None,
        )

    if local_batch_size % num_updates_per_batch != 0:
        _raise_geometry_error(
            reason="local batch cannot be split evenly into optimizer updates "
            "(local_batch_size % num_updates_per_batch != 0).",
            fix_hint="Choose a training.num_updates_per_batch that evenly divides "
            "the resolved local_batch_size.",
            local_batch_size=local_batch_size,
            update_batch_size=None,
            micro_batch_size=None,
        )
    update_batch_size = int(local_batch_size // num_updates_per_batch)

    if raw_micro_batch_size is None:
        micro_batch_size = int(update_batch_size)
    else:
        micro_batch_size = int(raw_micro_batch_size)
        if micro_batch_size < 1:
            _raise_geometry_error(
                reason="training.micro_batch_size must be >= 1.",
                fix_hint="Set training.micro_batch_size to a positive integer, "
                "or omit it to use the resolved local_mini_batch_size.",
                local_batch_size=local_batch_size,
                update_batch_size=update_batch_size,
                micro_batch_size=micro_batch_size,
            )

    if update_batch_size % micro_batch_size != 0:
        _raise_geometry_error(
            reason="local mini-batch cannot be split evenly into micro-batches "
            "(local_mini_batch_size % micro_batch_size != 0).",
            fix_hint="Choose a training.micro_batch_size that evenly divides "
            "the resolved local_mini_batch_size.",
            local_batch_size=local_batch_size,
            update_batch_size=update_batch_size,
            micro_batch_size=micro_batch_size,
        )


# ============================================================================
# Runtime init payload validation
# ============================================================================


def validate_rollout_engine_config(config: EngineConfig) -> None:
    """Minimal pre-dispatch validation for dedicated rollout engine config."""
    if not isinstance(config, EngineConfig):
        raise ValueError(
            f"rollout_engine_config must be an EngineConfig, got: {type(config).__name__}"
        )
    if not isinstance(config.engine_kwargs, dict):
        raise ValueError(
            "rollout_engine_config.engine_kwargs must be a dict, "
            f"got: {type(config.engine_kwargs).__name__}"
        )
    if not str(config.sampler_dotpath or "").strip():
        raise ValueError("rollout_engine_config.sampler_dotpath is required.")


def validate_rollout_actor_init_config(config: RolloutActorConfig) -> None:
    """Minimal pre-dispatch validation for rollout actor init config."""
    if not isinstance(config, RolloutActorConfig):
        raise ValueError(
            "rollout_actor_init_config must be a RolloutActorConfig, "
            f"got: {type(config).__name__}"
        )

    engine_init_payload = config.engine_init_payload
    if not isinstance(engine_init_payload, ComponentInitPayload):
        raise ValueError(
            "rollout_actor_init_config.engine_init_payload must be a ComponentInitPayload, "
            f"got: {type(engine_init_payload).__name__}"
        )
    reward_config = config.reward_config
    if not isinstance(reward_config, RewardSpec):
        raise ValueError(
            "rollout_actor_init_config.reward_config must be a RewardSpec, "
            f"got: {type(reward_config).__name__}"
        )

    engine_config = engine_init_payload.component_config
    if not isinstance(engine_config, EngineConfig):
        raise ValueError(
            "rollout_actor_init_config.engine_init_payload.component_config must be "
            "an EngineConfig, "
            f"got: {type(engine_config).__name__}"
        )

    validate_rollout_engine_config(engine_config)

    for required_key in (
        "num_inference_steps",
        "eta",
        "sde_type",
        "shift",
        "guidance_scale",
        "height",
        "width",
        "num_frames",
    ):
        if getattr(engine_config, required_key) is None:
            raise ValueError(
                f"rollout_actor_init_config.engine_init_payload.component_config.{required_key} is required."
            )

def validate_training_actor_init_config(config: TrainingActorConfig) -> None:
    """Minimal pre-dispatch validation for training actor config."""
    if not isinstance(config, TrainingActorConfig):
        raise ValueError(
            "training_actor_init_config must be a TrainingActorConfig, "
            f"got: {type(config).__name__}"
        )

    if not isinstance(config.optimizer_config, OptimizerConfig):
        raise ValueError(
            "training_actor_init_config.optimizer_config must be an OptimizerConfig, "
            f"got: {type(config.optimizer_config).__name__}"
        )
    if not isinstance(config.scheduler_config, LrSchedulerConfig):
        raise ValueError(
            "training_actor_init_config.scheduler_config must be an LrSchedulerConfig, "
            f"got: {type(config.scheduler_config).__name__}"
        )
    if not isinstance(config.training_config, TrainingExecutionConfig):
        raise ValueError(
            "training_actor_init_config.training_config must be a TrainingExecutionConfig, "
            f"got: {type(config.training_config).__name__}"
        )
    if config.sampling_config is not None and not isinstance(
        config.sampling_config, SamplingParams
    ):
        raise ValueError(
            "training_actor_init_config.sampling_config must be a SamplingParams or None, "
            f"got: {type(config.sampling_config).__name__}"
        )
    if not isinstance(config.reward_config, RewardSpec):
        raise ValueError(
            "training_actor_init_config.reward_config must be a RewardSpec, "
            f"got: {type(config.reward_config).__name__}"
        )
    if not isinstance(config.topology_config, TrainTopology):
        raise ValueError(
            "training_actor_init_config.topology_config must be a TrainTopology, "
            f"got: {type(config.topology_config).__name__}"
        )
    if not isinstance(config.training_plan_config, TrainingPlan):
        raise ValueError(
            "training_actor_init_config.training_plan_config must be a TrainingPlan, "
            f"got: {type(config.training_plan_config).__name__}"
        )

    from diffusionrl.construction import ComponentInitPayload
    from diffusionrl.models.config import ModelBundleConfig

    algorithm_init_payload = config.algorithm_init_payload
    if not isinstance(algorithm_init_payload, ComponentInitPayload):
        raise ValueError(
            "algorithm_init_payload must be a ComponentInitPayload, "
            f"got: {type(algorithm_init_payload).__name__}"
        )

    algorithm_config = algorithm_init_payload.component_config
    if not isinstance(algorithm_config, BaseAlgorithmConfig):
        raise ValueError(
            "algorithm_init_payload.component_config must be a BaseAlgorithmConfig instance, "
            f"got: {type(algorithm_config).__name__}"
        )

    model_init_payload = config.model_init_payload
    if not isinstance(model_init_payload, ComponentInitPayload):
        raise ValueError(
            "model_init_payload must be a ComponentInitPayload, "
            f"got: {type(model_init_payload).__name__}"
        )
    model_config = model_init_payload.component_config
    if not isinstance(model_config, ModelBundleConfig):
        raise ValueError(
            "model_init_payload.component_config must be a ModelBundleConfig, "
            f"got: {type(model_config).__name__}"
        )

    train_backend_init_payload = config.train_backend_init_payload
    if not isinstance(train_backend_init_payload, ComponentInitPayload):
        raise ValueError(
            "train_backend_init_payload must be a ComponentInitPayload, "
            f"got: {type(train_backend_init_payload).__name__}"
        )
    train_backend_config = train_backend_init_payload.component_config
    if not isinstance(train_backend_config, BaseTrainBackendConfig):
        raise ValueError(
            "train_backend_init_payload.component_config must be a TrainBackendConfig, "
            f"got: {type(train_backend_config).__name__}"
        )

    topology_config = config.topology_config
    for required_key in ("actor_count", "world_size", "dp_size"):
        value = getattr(topology_config, required_key)
        if value is None:
            raise ValueError(f"topology_config.{required_key} is required.")
        if int(value) < 1:
            raise ValueError(
                f"topology_config.{required_key} must be >= 1, got: {value!r}"
            )

    training_plan_config = config.training_plan_config
    if not isinstance(training_plan_config, TrainingPlan):
        raise ValueError(
            "training_actor_init_config.training_plan_config must be a TrainingPlan, "
            f"got: {type(training_plan_config).__name__}"
        )
    if training_plan_config.local_batch_size != (
        training_plan_config.local_mini_batch_size
        * training_plan_config.num_updates_per_batch
    ):
        raise ValueError(
            "training_plan_config.local_batch_size must equal "
            "local_mini_batch_size * num_updates_per_batch. "
            f"Got local_batch_size={training_plan_config.local_batch_size}, "
            f"local_mini_batch_size={training_plan_config.local_mini_batch_size}, "
            f"num_updates_per_batch={training_plan_config.num_updates_per_batch}."
        )
    if (
        training_plan_config.local_mini_batch_size
        % training_plan_config.micro_batch_size
        != 0
    ):
        raise ValueError(
            "training_plan_config.micro_batch_size must evenly divide "
            "local_mini_batch_size. "
            f"Got micro_batch_size={training_plan_config.micro_batch_size}, "
            f"local_mini_batch_size={training_plan_config.local_mini_batch_size}."
        )


def validate_launch_config_for_train(*, launch_config: LaunchConfig) -> None:
    """Verify a resolved LaunchConfig is compatible with ``diffusionrl.train``.

    Takes resolved framework objects rather than raw args; by design this
    validator runs after cmdline/schema.py has fully populated the launch
    config from CLI args.
    """
    rollout_info = launch_config.rollout_info
    if (
        not rollout_info.training_actor_sampling_mode
        and launch_config.rollout is None
    ):
        raise RuntimeError(
            "train.py requires a dedicated rollout launch when not using "
            "training_actor_sampling_mode "
            "(launch_config.rollout must be set)."
        )
    sync_protocol = str(rollout_info.sync_protocol or "").strip()
    if sync_protocol == "checkpoint_path":
        raise NotImplementedError(
            "train.py does not yet support sync_mode='checkpoint_path' "
            "(would call training_runtime.export_weights_to_path which the new TrainActor lacks)."
        )


# ============================================================================
# __all__
# ============================================================================

__all__ = [
    "ENV_REPO_ROOT",
    "is_probably_local_weight_sync_dir",
    "repo_root",
    "validate_colocate_fractions",
    "validate_direct_sampling_batch_geometry",
    "validate_dotpath",
    "validate_launch_config_for_train",
    "validate_precision_type",
    "validate_engine_algorithm_contract",
    "validate_reward_config",
    "validate_rollout_actor_init_config",
    "validate_rollout_engine_config",
    "validate_rollout_layout",
    "validate_train_backend_config",
    "validate_training_actor_init_config",
    "validate_training_actor_sampling_mode",
    "validate_training_batch_geometry",
    "validate_weight_sync",
]
