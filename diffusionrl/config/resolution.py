"""Config resolution helpers.

This module owns config semantics only:

- derive canonical config-facing values from ``TrainingArguments``
- share one cached ``ConfigBundle`` across validation and launch assembly
- avoid validate-time rewrites of the user-facing config object

Launch payload assembly, placement planning, and actor init config construction
live in ``launch_resolution.py``.

Training geometry is rollout-driven only:

- ``algorithm.prompts_per_rollout`` and ``algorithm.samples_per_prompt`` define
  the global rollout batch.
- local training batch size is derived from the resolved training topology.
- ``training.num_updates_per_local_batch`` shapes how one resolved local batch
  is split into optimizer updates.
- ``training.local_update_batch_size`` is always derived as
  ``local_batch_size / num_updates_per_local_batch``.
- ``training.local_micro_batch_size`` only controls micro-step slicing inside
  one local update batch and must evenly divide the resolved
  ``training.local_update_batch_size``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional, Tuple

from diffusionrl.algorithms.construction import (
    resolve_sampling_spec,
)
from diffusionrl.models import list_model_types, resolve_model_bundle_path
from diffusionrl.samplers.engine import get_engine_class_path
from diffusionrl.sde.rules import normalize_sde_type
from diffusionrl.training.backends import (
    TrainBackendConfig,
    TrainBackendCapabilities,
    TrainTopology,
    resolve_train_backend_capabilities_from_config,
    resolve_train_backend_config_from_args,
)
from diffusionrl.types.engine import normalize_engine_type
from diffusionrl.types.sampling import SamplingSpec, SamplingRequirements
from diffusionrl.utils.misc import load_function

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "diffusionrl.models.hunyuan.HunyuanModelBundle"
DEFAULT_SAMPLER_PATH = "diffusionrl.samplers.fsdp.hunyuan_sampler.FSDPHunyuanSampler"
DIRECT_ROLLOUT_MODE = "direct_sampling"
SEPARATE_ROLLOUT_MODE = "separate"
COLOCATE_ROLLOUT_MODE = "colocate"

ROLLOUT_MODES = {DIRECT_ROLLOUT_MODE, SEPARATE_ROLLOUT_MODE, COLOCATE_ROLLOUT_MODE}
ROLLOUT_ENGINE_TYPES = {
    "fsdp",
    "sglang",
}

_RESOLVED_CONFIG_CACHE_ATTR = "_diffusionrl_resolved_config"


def normalize_rollout_mode(value: Any) -> str:
    return str(value or "").strip().lower()


def normalize_rollout_service_engine(value: Any) -> Optional[str]:
    normalized = normalize_engine_type(value)
    if normalized and normalized not in ROLLOUT_ENGINE_TYPES:
        raise ValueError(
            "rollout.topology.service_engine must be one of "
            f"{sorted(ROLLOUT_ENGINE_TYPES)}, got: {value!r}"
        )
    return normalized or None


def rollout_mode_uses_service(mode: str) -> bool:
    return mode in {
        SEPARATE_ROLLOUT_MODE,
        COLOCATE_ROLLOUT_MODE,
    }


def rollout_mode_is_colocated(mode: str) -> bool:
    return mode == COLOCATE_ROLLOUT_MODE


def collect_sampling_requirements(*, algorithm: Any) -> SamplingRequirements:
    """Resolve final sampling requirements from algorithm.get_sampling_requirements()."""
    requirements = algorithm.get_sampling_requirements()
    raw_extras = getattr(requirements, "extras", None)
    extras: Dict[str, Any] = dict(raw_extras) if isinstance(raw_extras, Mapping) else {}

    return SamplingRequirements(
        requires_trajectory=bool(getattr(requirements, "requires_trajectory", True)),
        requires_log_prob=bool(getattr(requirements, "requires_log_prob", True)),
        requires_embeddings=bool(getattr(requirements, "requires_embeddings", True)),
        extras=extras,
    )


def load_engine_capabilities(engine_type: str) -> Dict[str, bool]:
    """Resolve engine capabilities from engine class declaration."""
    engine_path = get_engine_class_path(engine_type)
    engine_cls = load_function(engine_path)
    declared = getattr(engine_cls, "declared_capabilities", None)
    if not callable(declared):
        raise ValueError(
            f"Engine class {engine_path} must define classmethod declared_capabilities()."
        )
    return dict(declared())


@dataclass(frozen=True)
class ModelSpec:
    """Resolved model/sampler selection without mutating args."""

    model_path: str
    model_cls: Any
    model_type: str
    sampler_path: str
    model_default_engine_type: Optional[str]


@dataclass(frozen=True)
class RolloutTopology:
    """Resolved rollout topology without mutating args."""

    mode: str
    service_engine: Optional[str]

    @property
    def training_actor_sampling_mode(self) -> bool:
        return self.mode == DIRECT_ROLLOUT_MODE

    @property
    def is_sglang_engine(self) -> bool:
        return self.service_engine == "sglang"


@dataclass(frozen=True)
class RolloutModeInfo:
    """Resolved rollout mode state shared by validation and entrypoints."""

    rollout_topology: RolloutTopology
    training_actor_sampling_mode: bool
    is_sglang_engine: bool
    logprob_source: str
    replay_guard: bool
    replay_enabled: bool
    sync_protocol: str
    algorithm_type: str
    max_samples_per_request: Optional[int]
    effective_engine_capabilities: Optional[Dict[str, bool]] = None



@dataclass(frozen=True)
class TrainingPlan:
    """Authoritative training batch/update plan derived from explicit config."""

    global_batch_size: int
    local_batch_size: int
    local_update_batch_size: int
    local_micro_batch_size: int
    num_updates_per_local_batch: int
    update_slices: Tuple[Tuple[int, int], ...]
    mini_batch_slices_per_update: Tuple[Tuple[Tuple[int, int], ...], ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "global_batch_size": self.global_batch_size,
            "local_batch_size": self.local_batch_size,
            "local_update_batch_size": self.local_update_batch_size,
            "local_micro_batch_size": self.local_micro_batch_size,
            "num_updates_per_local_batch": self.num_updates_per_local_batch,
            "update_slices": [
                [start, end]
                for start, end in self.update_slices
            ],
            "mini_batch_slices_per_update": [
                [[start, end] for start, end in per_update]
                for per_update in self.mini_batch_slices_per_update
            ],
        }


@dataclass(frozen=True)
class ConfigBundle:
    """Canonical resolved config state shared by validation and launch assembly."""

    model_spec: ModelSpec
    rollout_mode_info: RolloutModeInfo
    sampling_spec: SamplingSpec
    train_backend_config: TrainBackendConfig
    train_backend_capabilities: TrainBackendCapabilities
    training_topology: TrainTopology
    training_plan: Optional[TrainingPlan] = None


def derive_sampling_spec(
    args: Any,
    *,
    model_spec: Optional[ModelSpec] = None,
) -> SamplingSpec:
    """Resolve the canonical sampling spec from SamplingConfig once."""
    resolved_model = model_spec if model_spec is not None else derive_model_spec(args)
    return resolve_sampling_spec(
        sampling=args.sampling,
        sampler_path=resolved_model.sampler_path,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        seed=args.seed,
    )


def normalize_train_backend_name(args: Any) -> str:
    return str(args.training.train_backend).strip().lower()


def normalize_lora_target_modules(raw: Any) -> Optional[list[str]]:
    if raw is None:
        return None
    if isinstance(raw, list):
        resolved = [str(item).strip() for item in raw if str(item).strip()]
        return resolved or None
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return None
        resolved = [item.strip() for item in stripped.split(",") if item.strip()]
        return resolved or None
    raise ValueError(
        "training.lora_target_modules must be a comma-separated string, list, or null. "
        f"Got: {type(raw).__name__}"
    )

def normalize_logprob_source(args: Any) -> str:
    return str(args.sampling.logprob_source).strip().lower()


def derive_model_spec(
    args: Any,
) -> ModelSpec:
    raw_model_path = str(args.model.model_path or "").strip()
    if not raw_model_path or raw_model_path == DEFAULT_MODEL_PATH:
        resolved_model_path = resolve_model_bundle_path(args.model.model_type)
        if not resolved_model_path:
            raise ValueError(
                f"Unknown model_type={args.model.model_type!r}. "
                f"Discovered model types: {list_model_types()}. "
                "Provide --model.model-path explicitly for custom models."
            )
    else:
        resolved_model_path = raw_model_path

    model_cls = load_function(resolved_model_path)

    resolved_model_type = str(args.model.model_type or "").strip().lower()
    declared_model_type_fn = getattr(model_cls, "declared_model_type", None)
    if callable(declared_model_type_fn):
        declared_model_type = declared_model_type_fn()
        if isinstance(declared_model_type, str) and declared_model_type.strip():
            declared_model_type = declared_model_type.strip().lower()
            if (
                resolved_model_type
                and raw_model_path
                and raw_model_path != DEFAULT_MODEL_PATH
                and resolved_model_type != declared_model_type
            ):
                raise ValueError(
                    "Configured model_type does not match the declared model type from model_path. "
                    f"Got model_type={resolved_model_type!r}, "
                    f"declared_model_type={declared_model_type!r}, "
                    f"model_path={resolved_model_path!r}."
                )
            resolved_model_type = declared_model_type

    model_default_sampler_path = None
    sampler_path_fn = getattr(model_cls, "default_sampler_path", None)
    if callable(sampler_path_fn):
        model_default_sampler_path = sampler_path_fn()

    model_default_engine_type = None
    engine_type_fn = getattr(model_cls, "default_sampler_engine", None)
    if callable(engine_type_fn):
        model_default_engine_type = engine_type_fn()

    raw_sampler_path = str(args.sampling.sampler_path or "").strip()
    resolved_sampler_path = raw_sampler_path
    if not resolved_sampler_path and model_default_sampler_path:
        resolved_sampler_path = str(model_default_sampler_path).strip()
    if not resolved_sampler_path:
        resolved_sampler_path = DEFAULT_SAMPLER_PATH

    return ModelSpec(
        model_path=resolved_model_path,
        model_cls=model_cls,
        model_type=resolved_model_type,
        sampler_path=resolved_sampler_path,
        model_default_engine_type=(
            str(model_default_engine_type).strip().lower()
            if isinstance(model_default_engine_type, str) and model_default_engine_type.strip()
            else None
        ),
    )


def _resolve_rollout_mode_value(value: Any) -> str:
    normalized = normalize_rollout_mode(value)
    if not normalized:
        raise ValueError(
            "rollout.topology.mode must be set explicitly. "
            "Implicit rollout topology derivation has been removed."
        )
    if normalized not in ROLLOUT_MODES:
        raise ValueError(
            "rollout.topology.mode must be one of "
            f"{sorted(ROLLOUT_MODES)}, got: {value!r}"
        )
    return normalized


def derive_rollout_topology(args: Any) -> RolloutTopology:
    rollout_topology_config = args.rollout.topology
    rollout_mode = _resolve_rollout_mode_value(rollout_topology_config.mode)
    rollout_service_engine = normalize_rollout_service_engine(
        rollout_topology_config.service_engine
    )
    if rollout_mode == DIRECT_ROLLOUT_MODE and rollout_service_engine is not None:
        raise ValueError(
            "direct_rollout is the only public direct-sampling selector. "
            "Leave rollout.topology.service_engine unset in direct_rollout mode."
        )
    return RolloutTopology(
        mode=rollout_mode,
        service_engine=rollout_service_engine,
    )


def derive_rollout_mode_info(args: Any) -> RolloutModeInfo:
    """Derive shared rollout mode state without mutating args."""
    rollout_topology = derive_rollout_topology(args)
    training_actor_sampling_mode = bool(rollout_topology.training_actor_sampling_mode)
    algorithm_type = str(args.algorithm.algorithm_type or "grpo").strip().lower()
    return RolloutModeInfo(
        rollout_topology=rollout_topology,
        training_actor_sampling_mode=training_actor_sampling_mode,
        is_sglang_engine=bool(rollout_topology.is_sglang_engine),
        logprob_source=normalize_logprob_source(args),
        replay_guard=(not training_actor_sampling_mode) and algorithm_type == "grpo",
        replay_enabled=bool(args.sampling.replay_log_probs),
        sync_protocol=str(args.sync.protocol or "").strip().lower(),
        algorithm_type=algorithm_type,
        max_samples_per_request=args.sampling.max_samples_per_request,
    )


def resolve_effective_engine_capabilities(
    *,
    rollout_mode_info: RolloutModeInfo,
) -> Optional[Dict[str, bool]]:
    """Resolve adjusted engine capabilities accounting for replay and logprob modes.

    Returns None when no dedicated rollout engine is used (direct sampling mode).
    The returned capabilities reflect effective availability after accounting for
    training-side replay and native logprob paths.
    """
    if rollout_mode_info.training_actor_sampling_mode:
        return None
    service_engine = rollout_mode_info.rollout_topology.service_engine
    if not service_engine:
        return None

    engine_caps = load_engine_capabilities(service_engine)

    # Replay mode: training-side replay provides log_prob and embeddings,
    # so the engine does not need to supply them natively.
    allow_replay = (
        rollout_mode_info.replay_enabled
        and rollout_mode_info.algorithm_type == "grpo"
    )
    if allow_replay:
        engine_caps = dict(engine_caps, requires_log_prob=True, requires_embeddings=True)
        logger.warning(
            "replay_log_probs=true enabled: allowing %s+GRPO with "
            "training-side old-log-prob replay (experimental path).",
            service_engine,
        )

    # Native logprob with replay guard: engine provides log_prob natively.
    if (
        rollout_mode_info.is_sglang_engine
        and rollout_mode_info.replay_guard
        and rollout_mode_info.logprob_source == "native"
    ):
        engine_caps = dict(engine_caps, requires_log_prob=True, requires_embeddings=True)

    return engine_caps


def derive_sampling_host_engine_type(
    args: Any,
    *,
    rollout_mode_info: Optional[RolloutModeInfo] = None,
) -> str:
    """Resolve the internal sampler host type from strict rollout topology."""
    resolved_mode_info = (
        rollout_mode_info if rollout_mode_info is not None else derive_rollout_mode_info(args)
    )
    if resolved_mode_info.training_actor_sampling_mode:
        return "fsdp"
    service_engine = resolved_mode_info.rollout_topology.service_engine
    if not service_engine:
        raise ValueError(
            "Dedicated rollout sampling requires rollout.topology.service_engine to be set."
        )
    return service_engine


def require_rollout_service_num_gpus(args: Any) -> int:
    """Require explicit dedicated rollout GPU ownership when a service is used."""
    rollout_topology_config = args.rollout.topology
    rollout_mode = _resolve_rollout_mode_value(rollout_topology_config.mode)
    if not rollout_mode_uses_service(rollout_mode):
        return 0

    raw_num_gpus = rollout_topology_config.service_num_gpus
    if raw_num_gpus is None:
        raise ValueError(
            "Dedicated rollout services require rollout.topology.service_num_gpus to be set explicitly. "
            "Do not infer actor GPU ownership from tp/sp parallel hints."
        )
    try:
        resolved = int(raw_num_gpus)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "rollout.topology.service_num_gpus must be an integer >= 1, "
            f"got: {raw_num_gpus!r}"
        ) from exc
    if resolved < 1:
        raise ValueError(
            "rollout.topology.service_num_gpus must be >= 1 for dedicated rollout services, "
            f"got: {resolved}"
        )
    return resolved


def derive_rollout_actor_gpu_count(
    args: Any,
    *,
    topology: Optional[RolloutTopology] = None,
) -> int:
    """Derive GPUs per rollout actor from explicit rollout topology."""
    resolved_topology = topology if topology is not None else derive_rollout_topology(args)
    if not rollout_mode_uses_service(resolved_topology.mode):
        return 0
    if not resolved_topology.service_engine:
        raise ValueError(
            "rollout.topology.mode requires a dedicated rollout service, but "
            "rollout.topology.service_engine is unset."
        )
    return require_rollout_service_num_gpus(args)


def derive_rollout_gpu_pool_size(
    args: Any,
    *,
    topology: Optional[RolloutTopology] = None,
) -> int:
    """Resolve placement GPU capacity available to rollout actors."""
    rollout_total_gpus = int(args.ray.rollout_num_nodes) * int(args.ray.rollout_num_gpus_per_node)
    resolved_topology = topology if topology is not None else derive_rollout_topology(args)
    if not rollout_mode_is_colocated(resolved_topology.mode):
        return rollout_total_gpus
    training_total_gpus = int(args.ray.training_num_nodes) * int(args.ray.training_num_gpus_per_node)
    return max(rollout_total_gpus, training_total_gpus)


def _require_positive_int(*, name: str, value: Any) -> int:
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{name} must be >= 1. Got {resolved}.")
    return resolved


def _build_relative_slices(
    *,
    total_size: int,
    chunk_size: int,
) -> Tuple[Tuple[int, int], ...]:
    resolved_total = _require_positive_int(name="total_size", value=total_size)
    resolved_chunk = _require_positive_int(name="chunk_size", value=chunk_size)
    slices = []
    start = 0
    while start < resolved_total:
        end = min(start + resolved_chunk, resolved_total)
        slices.append((int(start), int(end)))
        start = end
    return tuple(slices)


def _derive_local_batch_from_rollout_geometry(
    args: Any,
    *,
    training_topology: Optional[TrainTopology] = None,
) -> int:
    prompts_per_rollout = args.algorithm.prompts_per_rollout
    if prompts_per_rollout is None:
        raise ValueError(
            "algorithm.prompts_per_rollout is required. "
            "Rollout geometry is defined only by "
            "algorithm.prompts_per_rollout * algorithm.samples_per_prompt."
        )

    total_samples = (
        _require_positive_int(
            name="algorithm.prompts_per_rollout",
            value=prompts_per_rollout,
        )
        * _require_positive_int(
            name="algorithm.samples_per_prompt",
            value=args.algorithm.samples_per_prompt,
        )
    )
    resolved_training_topology = (
        training_topology if training_topology is not None else derive_training_topology(args)
    )
    dp_size = resolved_training_topology.dp_size
    if total_samples % dp_size != 0:
        raise ValueError(
            "Nominal rollout batch size must be divisible by the effective training dp_size. "
            f"Got total_samples={total_samples}, dp_size={dp_size}. "
            "Adjust algorithm.prompts_per_rollout, algorithm.samples_per_prompt, "
            "or the training backend topology."
        )
    return int(total_samples // dp_size)

def derive_training_topology(
    args: Any,
    *,
    backend_name: Optional[str] = None,
    backend_kwargs: Optional[Mapping[str, Any]] = None,
) -> TrainTopology:
    actor_count = max(
        1,
        int(args.ray.training_num_nodes) * int(args.ray.training_num_gpus_per_node),
    )
    if backend_name is None or backend_kwargs is None:
        resolved_backend_config = resolve_train_backend_config_from_args(args)
        backend = resolved_backend_config.name
        resolved_backend_kwargs = dict(resolved_backend_config.kwargs)
    else:
        backend = str(backend_name).strip().lower()
        resolved_backend_kwargs = dict(backend_kwargs)

    if backend == "veomni":
        dp_size = int(resolved_backend_kwargs.get("dp_size") or actor_count)
        dp_replicate_size = int(resolved_backend_kwargs.get("dp_replicate_size") or 1)
        dp_shard_size = resolved_backend_kwargs.get("dp_shard_size")
        if dp_shard_size is None:
            dp_shard_size = max(1, dp_size // max(1, dp_replicate_size))
        return TrainTopology(
            actor_count=actor_count,
            world_size=actor_count,
            dp_size=dp_size,
            dp_replicate_size=dp_replicate_size,
            dp_shard_size=int(dp_shard_size),
            tp_size=int(resolved_backend_kwargs.get("tp_size") or 1),
            pp_size=int(resolved_backend_kwargs.get("pp_size") or 1),
            sp_size=int(resolved_backend_kwargs.get("sp_size") or 1),
            ep_size=int(resolved_backend_kwargs.get("ep_size") or 1),
        )

    if backend == "megatron":
        hinted_dp_size = resolved_backend_kwargs.get("dp_size")
        tp_size = int(resolved_backend_kwargs.get("tp_size") or 1)
        pp_size = int(resolved_backend_kwargs.get("pp_size") or 1)
        sp_size = int(resolved_backend_kwargs.get("sp_size") or 1)
        ep_size = int(resolved_backend_kwargs.get("ep_size") or 1)
        denom = max(1, tp_size * pp_size * sp_size)
        dp_size = (
            int(hinted_dp_size)
            if hinted_dp_size is not None
            else max(1, actor_count // denom)
        )
        return TrainTopology(
            actor_count=actor_count,
            world_size=actor_count,
            dp_size=dp_size,
            dp_replicate_size=dp_size,
            dp_shard_size=1,
            tp_size=tp_size,
            pp_size=pp_size,
            sp_size=sp_size,
            ep_size=ep_size,
        )

    return TrainTopology(
        actor_count=actor_count,
        world_size=actor_count,
        dp_size=actor_count,
        dp_replicate_size=1,
        dp_shard_size=actor_count,
    )

def derive_num_updates_per_local_batch(args: Any) -> int:
    raw = args.training.num_updates_per_local_batch
    if raw is None:
        return 1
    return _require_positive_int(
        name="training.num_updates_per_local_batch",
        value=raw,
    )


def derive_global_rollout_batch_size(args: Any) -> int:
    return (
        require_prompts_per_rollout(args)
        * _require_positive_int(
            name="algorithm.samples_per_prompt",
            value=args.algorithm.samples_per_prompt,
        )
    )


def require_prompts_per_rollout(args: Any) -> int:
    prompts_per_rollout = args.algorithm.prompts_per_rollout
    if prompts_per_rollout is None:
        raise ValueError(
            "algorithm.prompts_per_rollout is required. "
            "Rollout geometry is defined only by "
            "algorithm.prompts_per_rollout * algorithm.samples_per_prompt."
        )
    return _require_positive_int(
        name="algorithm.prompts_per_rollout",
        value=prompts_per_rollout,
    )


def derive_training_plan(
    args: Any,
    *,
    training_topology: Optional[TrainTopology] = None,
) -> TrainingPlan:
    """Derive the resolved local/update/micro execution plan from validated args."""
    # Expects args that already passed validate_args(); this helper only derives
    # the resolved execution plan and does not repeat user-facing geometry
    # validation.
    global_batch_size = derive_global_rollout_batch_size(args)
    local_batch_size = _derive_local_batch_from_rollout_geometry(
        args,
        training_topology=training_topology,
    )
    num_updates_per_local_batch = derive_num_updates_per_local_batch(args)
    update_batch_size = local_batch_size // num_updates_per_local_batch
    raw_micro_batch_size = args.training.local_micro_batch_size
    micro_batch_size = (
        update_batch_size
        if raw_micro_batch_size is None
        else int(raw_micro_batch_size)
    )
    update_slices = tuple(
        (update_index * update_batch_size, (update_index + 1) * update_batch_size)
        for update_index in range(num_updates_per_local_batch)
    )
    mini_batch_slices_per_update = tuple(
        _build_relative_slices(
            total_size=update_batch_size,
            chunk_size=micro_batch_size,
        )
        for _ in range(num_updates_per_local_batch)
    )

    return TrainingPlan(
        global_batch_size=global_batch_size,
        local_batch_size=local_batch_size,
        local_update_batch_size=update_batch_size,
        local_micro_batch_size=micro_batch_size,
        num_updates_per_local_batch=num_updates_per_local_batch,
        update_slices=update_slices,
        mini_batch_slices_per_update=mini_batch_slices_per_update,
    )


def resolve_config(
    args: Any,
    *,
    include_training_plan: bool = False,
) -> ConfigBundle:
    """Resolve and cache canonical config-derived state for one args object."""
    cached = getattr(args, _RESOLVED_CONFIG_CACHE_ATTR, None)
    if isinstance(cached, ConfigBundle):
        if include_training_plan and cached.training_plan is None:
            cached = replace(
                cached,
                training_plan=derive_training_plan(
                    args,
                    training_topology=cached.training_topology,
                ),
            )
            setattr(args, _RESOLVED_CONFIG_CACHE_ATTR, cached)
        return cached

    model_spec = derive_model_spec(args)
    rollout_mode_info = derive_rollout_mode_info(args)
    effective_caps = resolve_effective_engine_capabilities(
        rollout_mode_info=rollout_mode_info,
    )
    if effective_caps is not None:
        rollout_mode_info = replace(
            rollout_mode_info,
            effective_engine_capabilities=effective_caps,
        )
    sampling_spec = derive_sampling_spec(args, model_spec=model_spec)
    train_backend_config = resolve_train_backend_config_from_args(args)
    train_backend_capabilities = resolve_train_backend_capabilities_from_config(
        train_backend_config
    )
    training_topology = derive_training_topology(
        args,
        backend_name=train_backend_config.name,
        backend_kwargs=train_backend_config.kwargs,
    )
    training_plan = None
    if include_training_plan:
        training_plan = derive_training_plan(
            args,
            training_topology=training_topology,
        )
    resolved = ConfigBundle(
        model_spec=model_spec,
        rollout_mode_info=rollout_mode_info,
        sampling_spec=sampling_spec,
        train_backend_config=train_backend_config,
        train_backend_capabilities=train_backend_capabilities,
        training_topology=training_topology,
        training_plan=training_plan,
    )
    setattr(args, _RESOLVED_CONFIG_CACHE_ATTR, resolved)
    return resolved


__all__ = [
    "ConfigBundle",
    "ModelSpec",
    "RolloutModeInfo",
    "RolloutTopology",
    "TrainingPlan",
    "TrainTopology",
    "COLOCATE_ROLLOUT_MODE",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_SAMPLER_PATH",
    "DIRECT_ROLLOUT_MODE",
    "ROLLOUT_ENGINE_TYPES",
    "ROLLOUT_MODES",
    "SEPARATE_ROLLOUT_MODE",
    "normalize_rollout_mode",
    "normalize_rollout_service_engine",
    "collect_sampling_requirements",
    "derive_global_rollout_batch_size",
    "derive_model_spec",
    "derive_num_updates_per_local_batch",
    "derive_rollout_actor_gpu_count",
    "derive_rollout_gpu_pool_size",
    "derive_sampling_host_engine_type",
    "derive_rollout_mode_info",
    "derive_rollout_topology",
    "derive_sampling_spec",
    "derive_training_plan",
    "derive_training_topology",
    "load_engine_capabilities",
    "normalize_logprob_source",
    "normalize_lora_target_modules",
    "normalize_train_backend_name",
    "require_prompts_per_rollout",
    "require_rollout_service_num_gpus",
    "resolve_effective_engine_capabilities",
    "resolve_config",
    "rollout_mode_is_colocated",
    "rollout_mode_uses_service",
    "rollout_mode_label",
]
