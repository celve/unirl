"""Config resolution helpers.

These helpers derive runtime-facing values from ``TrainingArguments`` without
mutating the original config object. Validation and builders should use this
module instead of relying on validate-time argument rewrites.

Most helpers are pure data resolution. A small number of contract helpers
currently instantiate transient algorithm objects so validation can read
algorithm-declared requirements. Those helpers should remain side-effect-light:
they must not initialize device/runtime state, actors, or placement resources.

Training geometry is resolved in one of two explicit modes:

- Mode A, rollout-driven geometry:
  no local training geometry owner is set. ``algorithm.prompts_per_rollout``
  and ``algorithm.samples_per_prompt`` define the global rollout batch, and the
  local training batch is derived from the resolved training topology.
- Mode B, local-training-driven geometry:
  ``training.local_update_batch_size`` or
  ``training.num_updates_per_local_batch`` is set. Local training geometry then
  owns the batch plan, and global rollout batch size / prompts_per_rollout are
  derived from that plan plus ``algorithm.samples_per_prompt``.
  ``training.local_micro_batch_size`` alone only controls micro-step slicing; it
  does not take ownership of the rollout batch size.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from diffusionrl.algorithms.registry import DEFAULT_ALGORITHM_PATHS
from diffusionrl.config.rollout_topology import (
    DIRECT_ROLLOUT_MODE,
    ROLLOUT_ENGINE_TYPES,
    ROLLOUT_MODES,
    resolve_rollout_service_num_gpus,
    rollout_mode_is_colocated,
    rollout_mode_uses_service,
)
from diffusionrl.models import list_model_types, resolve_model_bundle_path
from diffusionrl.samplers.engine import get_engine_class_path
from diffusionrl.types.sampling import SamplingRequirements
from diffusionrl.utils.misc import load_function

DEFAULT_MODEL_PATH = "diffusionrl.models.hunyuan.HunyuanModelBundle"
DEFAULT_SAMPLER_PATH = "diffusionrl.samplers.fsdp.hunyuan_sampler.FSDPHunyuanSampler"


def _instantiate_algorithm_for_contracts(args: Any) -> Any:
    """Instantiate a transient algorithm only for config-time contract checks."""
    from diffusionrl.config.build_domain_args import build_algorithm_config

    algorithm_config = build_algorithm_config(args)
    algorithm_path = algorithm_config.get("algorithm_path")
    if not isinstance(algorithm_path, str) or not algorithm_path.strip():
        raise ValueError("build_algorithm_config() returned an empty algorithm_path.")
    try:
        algorithm_cls = load_function(algorithm_path.strip())
    except Exception as exc:
        raise ValueError(
            "Cannot resolve algorithm class from args.algorithm.algorithm_path="
            f"{algorithm_path!r}."
        ) from exc
    if not hasattr(algorithm_cls, "from_config"):
        raise ValueError(
            f"Algorithm class {algorithm_cls.__name__} must define classmethod from_config(config)."
        )
    return algorithm_cls.from_config(algorithm_config)


def resolve_sampling_requirements(
    args: Any,
    *,
    algorithm: Optional[Any] = None,
) -> SamplingRequirements:
    """Resolve final sampling contract from algorithm.get_sampling_requirements()."""
    resolved_algorithm = algorithm if algorithm is not None else _instantiate_algorithm_for_contracts(args)
    requirements = resolved_algorithm.get_sampling_requirements()
    raw_extras = getattr(requirements, "extras", None)
    extras: Dict[str, Any] = dict(raw_extras) if isinstance(raw_extras, Mapping) else {}

    return SamplingRequirements(
        requires_trajectory=bool(getattr(requirements, "requires_trajectory", True)),
        requires_log_prob=bool(getattr(requirements, "requires_log_prob", True)),
        requires_embeddings=bool(getattr(requirements, "requires_embeddings", True)),
        extras=extras,
    )


def resolve_engine_capabilities(engine_type: str) -> Dict[str, bool]:
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
class ResolvedModelRuntime:
    """Resolved model/sampler identity without mutating args."""

    model_path: str
    model_cls: Any
    model_type: str
    sampler_path: str
    model_default_engine_type: Optional[str]


@dataclass(frozen=True)
class ResolvedRolloutTopology:
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
class ResolvedTrainTopology:
    """Authoritative training topology derived from explicit config only.

    actor_count is the size of the launched training actor group.
    world_size is the distributed training rank count.
    dp_size is the data-parallel consumer count used for training batch geometry.

    These values often coincide in the current FSDP mainline, but they should
    not be treated as interchangeable concepts.
    """

    actor_count: int
    world_size: int
    dp_size: int
    dp_replicate_size: int = 1
    dp_shard_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    sp_size: int = 1
    ep_size: int = 1
    data_partition_axis: str = "dp"
    partition_mode: str = "data_parallel"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "actor_count": int(self.actor_count),
            "world_size": int(self.world_size),
            "dp_size": int(self.dp_size),
            "dp_replicate_size": int(self.dp_replicate_size),
            "dp_shard_size": int(self.dp_shard_size),
            "tp_size": int(self.tp_size),
            "pp_size": int(self.pp_size),
            "sp_size": int(self.sp_size),
            "ep_size": int(self.ep_size),
            "data_partition_axis": str(self.data_partition_axis),
            "partition_mode": str(self.partition_mode),
        }


@dataclass(frozen=True)
class ResolvedTrainingPlan:
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
            "global_batch_size": int(self.global_batch_size),
            "local_batch_size": int(self.local_batch_size),
            "local_update_batch_size": int(self.local_update_batch_size),
            "local_micro_batch_size": int(self.local_micro_batch_size),
            "num_updates_per_local_batch": int(self.num_updates_per_local_batch),
            "update_slices": [
                [int(start), int(end)]
                for start, end in self.update_slices
            ],
            "mini_batch_slices_per_update": [
                [[int(start), int(end)] for start, end in per_update]
                for per_update in self.mini_batch_slices_per_update
            ],
        }


def parse_json_object(raw: Any, *, field_name: str) -> Dict[str, Any]:
    """Parse a mapping-like config payload without mutating the source field."""

    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except Exception as exc:
            raise ValueError(f"Invalid {field_name}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(
                f"{field_name} must decode to a JSON object, got: {type(parsed).__name__}"
            )
        return dict(parsed)
    raise ValueError(
        f"{field_name} must be a JSON object (YAML mapping) or JSON object string, "
        f"got: {type(raw).__name__}"
    )


def resolve_algorithm_kwargs(args: Any) -> Dict[str, Any]:
    return parse_json_object(
        getattr(args.algorithm, "algorithm_kwargs", {}),
        field_name="algorithm.algorithm_kwargs",
    )


def resolve_train_backend_kwargs(args: Any) -> Dict[str, Any]:
    return parse_json_object(
        getattr(args.training, "train_backend_kwargs", {}),
        field_name="training.train_backend_kwargs",
    )


def resolve_algorithm_path(args: Any) -> str:
    raw_algorithm_path = getattr(args.algorithm, "algorithm_path", None)
    if isinstance(raw_algorithm_path, str) and raw_algorithm_path.strip():
        return raw_algorithm_path.strip()

    algorithm_type = str(getattr(args.algorithm, "algorithm_type", "") or "").strip().lower()
    resolved = DEFAULT_ALGORITHM_PATHS.get(algorithm_type)
    if not resolved:
        raise ValueError(
            f"Cannot resolve algorithm.algorithm_path for algorithm_type={algorithm_type!r}. "
            "Provide --algorithm.algorithm-path explicitly or register this algorithm_type."
        )
    return resolved


def resolve_train_backend_name(args: Any) -> str:
    return str(getattr(args.training, "train_backend", "fsdp") or "fsdp").strip().lower()


def resolve_lora_target_modules(raw: Any) -> Optional[list[str]]:
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


def resolve_debug_mode(args: Any) -> str:
    value = getattr(args.debug, "debug_mode", "none")
    if value is None:
        return "none"
    if not isinstance(value, str):
        raise ValueError(f"debug.debug_mode must be a string, got: {type(value).__name__}")
    return value


def resolve_logprob_source(args: Any) -> str:
    value = getattr(args.sampling, "logprob_source", "replay")
    if value is None:
        return "replay"
    if not isinstance(value, str):
        raise ValueError(f"sampling.logprob_source must be a string, got: {type(value).__name__}")
    return value


def resolve_model_runtime(
    args: Any,
    *,
    explicit_sampler_path: bool,
) -> ResolvedModelRuntime:
    raw_model_path = str(getattr(args.model, "model_path", "") or "").strip()
    if not raw_model_path or raw_model_path == DEFAULT_MODEL_PATH:
        resolved_model_path = resolve_model_bundle_path(getattr(args.model, "model_type", None))
        if not resolved_model_path:
            raise ValueError(
                f"Unknown model_type={getattr(args.model, 'model_type', None)!r}. "
                f"Discovered model types: {list_model_types()}. "
                "Provide --model.model-path explicitly for custom models."
            )
    else:
        resolved_model_path = raw_model_path

    model_cls = load_function(resolved_model_path)

    resolved_model_type = str(getattr(args.model, "model_type", "") or "").strip().lower()
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

    raw_sampler_path = str(getattr(args.sampling, "sampler_path", "") or "").strip()
    if raw_sampler_path and not (raw_sampler_path == DEFAULT_SAMPLER_PATH and not explicit_sampler_path):
        resolved_sampler_path = raw_sampler_path
    elif model_default_sampler_path:
        resolved_sampler_path = str(model_default_sampler_path).strip()
    else:
        resolved_sampler_path = raw_sampler_path or DEFAULT_SAMPLER_PATH

    return ResolvedModelRuntime(
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
    if value is None:
        raise ValueError(
            "rollout.mode must be set explicitly. "
            "Implicit rollout topology derivation has been removed."
        )
    if not isinstance(value, str):
        raise ValueError(
            "rollout.mode must be one of "
            f"{sorted(ROLLOUT_MODES)}, got non-string value: {value!r}"
        )
    if not value:
        raise ValueError(
            "rollout.mode must be set explicitly. "
            "Implicit rollout topology derivation has been removed."
        )
    if value not in ROLLOUT_MODES:
        raise ValueError(
            "rollout.mode must be one of "
            f"{sorted(ROLLOUT_MODES)}, got: {value!r}"
        )
    return value


def _resolve_rollout_service_engine_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            "rollout.service_engine must be one of "
            f"{sorted(ROLLOUT_ENGINE_TYPES)}, got non-string value: {value!r}"
        )
    if not value:
        return None
    if value not in ROLLOUT_ENGINE_TYPES:
        raise ValueError(
            "rollout.service_engine must be one of "
            f"{sorted(ROLLOUT_ENGINE_TYPES)}, got: {value!r}"
        )
    return value


def resolve_rollout_topology(args: Any) -> ResolvedRolloutTopology:
    rollout_mode = _resolve_rollout_mode_value(getattr(args.rollout, "mode", None))
    rollout_service_engine = _resolve_rollout_service_engine_value(
        getattr(args.rollout, "service_engine", None)
    )
    return ResolvedRolloutTopology(
        mode=rollout_mode,
        service_engine=rollout_service_engine,
    )


def resolve_rollout_gpus_per_actor(args: Any) -> int:
    """Resolve GPUs per rollout actor based on explicit rollout topology."""
    topology = resolve_rollout_topology(args)
    if not rollout_mode_uses_service(topology.mode):
        return 0
    if not topology.service_engine:
        raise ValueError(
            "rollout.mode requires a dedicated rollout service, but rollout.service_engine is unset."
        )
    if topology.service_engine != "sglang":
        raise ValueError(
            f"Unsupported dedicated rollout engine: {topology.service_engine!r}. "
            "Expected: sglang."
        )
    return resolve_rollout_service_num_gpus(args)


def resolve_rollout_gpu_pool_size(args: Any) -> int:
    """Resolve placement GPU capacity available to rollout actors."""
    rollout_total_gpus = int(args.ray.rollout_num_nodes) * int(args.ray.rollout_num_gpus_per_node)
    topology = resolve_rollout_topology(args)
    if not rollout_mode_is_colocated(topology.mode):
        return rollout_total_gpus
    training_total_gpus = int(args.ray.training_num_nodes) * int(args.ray.training_num_gpus_per_node)
    return max(rollout_total_gpus, training_total_gpus)


def _maybe_positive_int(value: Any) -> Optional[int]:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    if resolved < 1:
        return None
    return resolved


def _uses_explicit_training_geometry(args: Any) -> bool:
    """Whether local training geometry owns batch resolution."""
    training_cfg = getattr(args, "training", None)
    if training_cfg is None:
        return False
    return any(
        getattr(training_cfg, field_name, None) is not None
        for field_name in (
            "local_update_batch_size",
            "num_updates_per_local_batch",
        )
    )


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


def _resolve_local_batch_from_rollout_geometry(args: Any) -> int:
    prompts_per_rollout = getattr(args.algorithm, "prompts_per_rollout", None)
    if prompts_per_rollout is None:
        raise ValueError(
            "algorithm.prompts_per_rollout is required when training geometry is not "
            "owned by training.local_update_batch_size."
        )

    total_samples = (
        _require_positive_int(
            name="algorithm.prompts_per_rollout",
            value=prompts_per_rollout,
        )
        * _require_positive_int(
            name="algorithm.samples_per_prompt",
            value=getattr(args.algorithm, "samples_per_prompt", 1),
        )
    )
    dp_size = resolve_training_topology(args).dp_size
    if total_samples % dp_size != 0:
        raise ValueError(
            "Nominal rollout batch size must be divisible by the effective training dp_size. "
            f"Got total_samples={total_samples}, dp_size={dp_size}. "
            "Adjust algorithm.prompts_per_rollout, algorithm.samples_per_prompt, "
            "or the training backend topology."
        )
    return int(total_samples // dp_size)


def resolve_training_dp_size(args: Any) -> int:
    return resolve_training_topology(args).dp_size


def resolve_training_topology(args: Any) -> ResolvedTrainTopology:
    actor_count = max(
        1,
        int(args.ray.training_num_nodes) * int(args.ray.training_num_gpus_per_node),
    )
    backend = resolve_train_backend_name(args)
    backend_kwargs = resolve_train_backend_kwargs(args)

    if backend == "veomni":
        dp_size = _maybe_positive_int(backend_kwargs.get("dp_size")) or actor_count
        dp_replicate_size = _maybe_positive_int(backend_kwargs.get("dp_replicate_size")) or 1
        dp_shard_size = _maybe_positive_int(backend_kwargs.get("dp_shard_size"))
        if dp_shard_size is None:
            dp_shard_size = max(1, dp_size // max(1, dp_replicate_size))
        return ResolvedTrainTopology(
            actor_count=actor_count,
            world_size=actor_count,
            dp_size=dp_size,
            dp_replicate_size=dp_replicate_size,
            dp_shard_size=dp_shard_size,
            tp_size=_maybe_positive_int(backend_kwargs.get("tp_size")) or 1,
            pp_size=_maybe_positive_int(backend_kwargs.get("pp_size")) or 1,
            sp_size=_maybe_positive_int(backend_kwargs.get("sp_size")) or 1,
            ep_size=_maybe_positive_int(backend_kwargs.get("ep_size")) or 1,
        )

    if backend == "megatron":
        hinted_dp_size = _maybe_positive_int(backend_kwargs.get("dp_size"))
        tp_size = _maybe_positive_int(backend_kwargs.get("tp_size")) or 1
        pp_size = _maybe_positive_int(backend_kwargs.get("pp_size")) or 1
        sp_size = _maybe_positive_int(backend_kwargs.get("sp_size")) or 1
        ep_size = _maybe_positive_int(backend_kwargs.get("ep_size")) or 1
        denom = max(1, tp_size * pp_size * sp_size)
        dp_size = hinted_dp_size if hinted_dp_size is not None else max(1, actor_count // denom)
        return ResolvedTrainTopology(
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

    return ResolvedTrainTopology(
        actor_count=actor_count,
        world_size=actor_count,
        dp_size=actor_count,
        dp_replicate_size=1,
        dp_shard_size=actor_count,
    )


def resolve_nominal_local_training_batch_size(args: Any) -> int:
    if (
        getattr(args.training, "local_update_batch_size", None) is not None
        or getattr(args.training, "num_updates_per_local_batch", None) is not None
    ):
        update_batch_size = resolve_local_update_batch_size(args)
        num_updates_per_local_batch = resolve_num_updates_per_local_batch(args)
        return int(update_batch_size) * int(num_updates_per_local_batch)
    return _resolve_local_batch_from_rollout_geometry(args)


def resolve_local_update_batch_size(args: Any) -> int:
    raw = getattr(args.training, "local_update_batch_size", None)
    if raw is not None:
        return _require_positive_int(
            name="training.local_update_batch_size",
            value=raw,
        )
    return _resolve_local_batch_from_rollout_geometry(args)


def resolve_local_micro_batch_size(args: Any) -> int:
    raw = getattr(args.training, "local_micro_batch_size", None)
    if raw is None:
        return int(resolve_local_update_batch_size(args))
    return _require_positive_int(
        name="training.local_micro_batch_size",
        value=raw,
    )


def resolve_num_updates_per_local_batch(args: Any) -> int:
    raw = getattr(args.training, "num_updates_per_local_batch", None)
    if raw is None:
        return 1
    if getattr(args.training, "local_update_batch_size", None) is None:
        raise ValueError(
            "training.num_updates_per_local_batch requires "
            "training.local_update_batch_size."
        )
    return _require_positive_int(
        name="training.num_updates_per_local_batch",
        value=raw,
    )


def resolve_global_rollout_batch_size(args: Any) -> int:
    if _uses_explicit_training_geometry(args):
        local_batch_size = resolve_nominal_local_training_batch_size(args)
        dp_size = resolve_training_topology(args).dp_size
        return int(local_batch_size) * int(dp_size)
    return (
        _require_positive_int(
            name="algorithm.prompts_per_rollout",
            value=getattr(args.algorithm, "prompts_per_rollout", None),
        )
        * _require_positive_int(
            name="algorithm.samples_per_prompt",
            value=getattr(args.algorithm, "samples_per_prompt", 1),
        )
    )


def resolve_prompts_per_rollout(args: Any) -> int:
    samples_per_prompt = _require_positive_int(
        name="algorithm.samples_per_prompt",
        value=getattr(args.algorithm, "samples_per_prompt", 1),
    )
    resolved_global_batch_size = resolve_global_rollout_batch_size(args)
    if resolved_global_batch_size % samples_per_prompt != 0:
        raise ValueError(
            "Resolved rollout sample budget must be divisible by algorithm.samples_per_prompt. "
            f"Got global_rollout_sample_budget={resolved_global_batch_size}, "
            f"samples_per_prompt={samples_per_prompt}."
        )
    derived_prompts = int(resolved_global_batch_size) // int(samples_per_prompt)
    explicit_prompts = getattr(args.algorithm, "prompts_per_rollout", None)
    if explicit_prompts is None:
        return int(derived_prompts)
    resolved_explicit = _require_positive_int(
        name="algorithm.prompts_per_rollout",
        value=explicit_prompts,
    )
    if _uses_explicit_training_geometry(args) and resolved_explicit != derived_prompts:
        raise ValueError(
            "algorithm.prompts_per_rollout conflicts with the explicit training geometry. "
            f"Got explicit prompts_per_rollout={resolved_explicit}, "
            f"derived prompts_per_rollout={derived_prompts}."
        )
    return int(resolved_explicit)


def resolve_training_plan(args: Any) -> ResolvedTrainingPlan:
    global_batch_size = resolve_global_rollout_batch_size(args)
    local_batch_size = resolve_nominal_local_training_batch_size(args)
    update_batch_size = resolve_local_update_batch_size(args)
    micro_batch_size = resolve_local_micro_batch_size(args)
    num_updates_per_local_batch = resolve_num_updates_per_local_batch(args)
    update_slices = tuple(
        (update_index * int(update_batch_size), (update_index + 1) * int(update_batch_size))
        for update_index in range(int(num_updates_per_local_batch))
    )
    mini_batch_slices_per_update = tuple(
        _build_relative_slices(
            total_size=int(update_batch_size),
            chunk_size=int(micro_batch_size),
        )
        for _ in range(int(num_updates_per_local_batch))
    )

    return ResolvedTrainingPlan(
        global_batch_size=int(global_batch_size),
        local_batch_size=int(local_batch_size),
        local_update_batch_size=int(update_batch_size),
        local_micro_batch_size=int(micro_batch_size),
        num_updates_per_local_batch=int(num_updates_per_local_batch),
        update_slices=update_slices,
        mini_batch_slices_per_update=mini_batch_slices_per_update,
    )


__all__ = [
    "ResolvedModelRuntime",
    "ResolvedRolloutTopology",
    "ResolvedTrainingPlan",
    "ResolvedTrainTopology",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_SAMPLER_PATH",
    "parse_json_object",
    "resolve_algorithm_kwargs",
    "resolve_algorithm_path",
    "resolve_debug_mode",
    "resolve_engine_capabilities",
    "resolve_logprob_source",
    "resolve_lora_target_modules",
    "resolve_model_runtime",
    "resolve_nominal_local_training_batch_size",
    "resolve_num_updates_per_local_batch",
    "resolve_local_micro_batch_size",
    "resolve_local_update_batch_size",
    "resolve_prompts_per_rollout",
    "resolve_global_rollout_batch_size",
    "resolve_rollout_gpu_pool_size",
    "resolve_rollout_gpus_per_actor",
    "resolve_rollout_topology",
    "resolve_sampling_requirements",
    "resolve_train_backend_kwargs",
    "resolve_train_backend_name",
    "resolve_training_dp_size",
    "resolve_training_plan",
    "resolve_training_topology",
]
