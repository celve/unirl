"""Cmdline-driven derivation from TrainingArguments to resolved framework objects."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

from diffusionrl.cmdline.actors import (
    build_rollout_actor_init_config_from_args,
    build_training_actor_init_config_from_args,
    build_training_sampling_config_from_args,
)
from diffusionrl.cmdline.registry import derive_component_cmdline_config_parser
from diffusionrl.config.assembly import (
    DerivedConfig,
    LaunchConfig,
    PlacementSpec,
    RolloutLaunch,
    RolloutServicesSpec,
    TrainingLaunch,
    WeightSyncSpec,
)
from diffusionrl.config.resolution import (
    COLOCATE_ROLLOUT_MODE,
    DEFAULT_MODEL_PATH,
    DEFAULT_SAMPLER_PATH,
    DIRECT_ROLLOUT_MODE,
    ROLLOUT_MODES,
    SEPARATE_ROLLOUT_MODE,
    load_engine_capabilities,
    rollout_mode_is_colocated,
)
from diffusionrl.config.spec import ModelSpec, RolloutInfo, SamplingSpec, TrainingPlan
from diffusionrl.construction import ComponentInitPayload
from diffusionrl.models import derive_model_bundle_path, list_model_types
from diffusionrl.registry import derive_registry_or_dotpath
from diffusionrl.training.backends import (
    BaseTrainBackendConfig,
    MegatronTrainBackendConfig,
    TrainTopology,
    VeOmniTrainBackendConfig,
    derive_train_backend_capabilities,
    derive_train_backend_launch_spec,
    supported_train_backends,
)
from diffusionrl.training.backends.registry import TRAIN_BACKEND_COMPONENT_FAMILY
from diffusionrl.types.engine import ROLLOUT_ENGINE_TYPES, normalize_engine_type
from diffusionrl.utils.misc import load_function

logger = logging.getLogger(__name__)


def derive_algorithm_identifier(args: Any) -> str:
    algorithm_dotpath = args.algorithm.algorithm_dotpath
    algorithm_type = args.algorithm.algorithm_type
    if isinstance(algorithm_dotpath, str) and algorithm_dotpath.strip():
        identifier = algorithm_dotpath.strip()
    else:
        identifier = str(algorithm_type)
    return identifier


def derive_train_backend_dotpath(args: Any) -> Optional[str]:
    backend_dotpath = str(args.training.train_backend_dotpath or "").strip()
    return backend_dotpath or None


def derive_train_backend_identifier(args: Any) -> str:
    backend_dotpath = derive_train_backend_dotpath(args)
    backend_name = str(args.training.train_backend or "fsdp").strip().lower()
    if backend_dotpath is None and backend_name not in supported_train_backends():
        raise ValueError(
            f"Unsupported train_backend={backend_name!r}. "
            f"Expected one of {list(supported_train_backends())} or provide training.train_backend_dotpath."
        )
    return backend_dotpath or backend_name


def derive_train_backend_config(args: Any) -> BaseTrainBackendConfig:
    backend_identifier = derive_train_backend_identifier(args)
    backend_cls = derive_registry_or_dotpath(
        component_family=TRAIN_BACKEND_COMPONENT_FAMILY,
        identifier=backend_identifier,
    )
    parser_fn = derive_component_cmdline_config_parser(backend_cls)
    config = parser_fn(args)
    if not isinstance(config, BaseTrainBackendConfig):
        raise TypeError(
            f"Train backend config parser for {backend_identifier!r} must return a "
            f"BaseTrainBackendConfig subclass, got {type(config).__name__}."
        )
    return config


def derive_algorithm_dotpath(args: Any) -> str:
    from diffusionrl.algorithms.registry import derive_algorithm_dotpath

    return derive_algorithm_dotpath(derive_algorithm_identifier(args))


def derive_model_spec(args: Any) -> ModelSpec:
    raw_model_dotpath = args.model.model_dotpath
    if not raw_model_dotpath or raw_model_dotpath == DEFAULT_MODEL_PATH:
        model_dotpath = derive_model_bundle_path(args.model.model_type)
        if not model_dotpath:
            raise ValueError(
                f"Unknown model_type={args.model.model_type!r}. "
                f"Discovered model types: {list_model_types()}. "
                "Provide --model.model-dotpath explicitly for custom models."
            )
    else:
        model_dotpath = raw_model_dotpath

    model_cls = load_function(model_dotpath)
    model_type = args.model.model_type
    declared_model_type_fn = getattr(model_cls, "declared_model_type", None)
    if callable(declared_model_type_fn):
        declared_model_type = declared_model_type_fn()
        if (
            model_type
            and model_dotpath
            and model_dotpath != DEFAULT_MODEL_PATH
            and model_type != declared_model_type
        ):
            raise ValueError(
                "Configured model_type does not match the declared model type from model_dotpath. "
                f"Got model_type={model_type!r}, "
                f"declared_model_type={declared_model_type!r}, "
                f"model_dotpath={model_dotpath!r}."
            )
        model_type = declared_model_type

    default_sampler_dotpath = None
    sampler_dotpath_fn = getattr(model_cls, "default_sampler_dotpath", None)
    if callable(sampler_dotpath_fn):
        default_sampler_dotpath = sampler_dotpath_fn()

    sampler_dotpath = args.sampling.sampler_dotpath
    if not sampler_dotpath and default_sampler_dotpath:
        sampler_dotpath = default_sampler_dotpath
    if not sampler_dotpath:
        sampler_dotpath = DEFAULT_SAMPLER_PATH

    return ModelSpec(
        model_dotpath=model_dotpath,
        model_cls=model_cls,
        model_type=model_type,
        sampler_dotpath=sampler_dotpath,
    )


def derive_sampling_spec(
    args: Any,
    *,
    model_spec: Optional[ModelSpec] = None,
) -> SamplingSpec:
    model_spec = model_spec if model_spec is not None else derive_model_spec(args)
    sampling = args.sampling
    return SamplingSpec(
        sampler_dotpath=model_spec.sampler_dotpath,
        num_inference_steps=sampling.num_inference_steps,
        guidance_scale=sampling.guidance_scale,
        height=sampling.height,
        width=sampling.width,
        num_frames=sampling.num_frames,
        seed=args.seed,
        replay_sampler_dotpath=sampling.replay_sampler_dotpath,
        sampling_adapter=sampling.sampling_adapter,
        init_same_noise=sampling.init_same_noise,
        sampler_kwargs=sampling.sampler_kwargs,
        eta=sampling.eta,
        sde_type=sampling.sde_type,
        shift=sampling.shift,
    )


def derive_rollout_info(args: Any) -> RolloutInfo:
    rollout_config = args.rollout
    rollout_mode = rollout_config.mode
    if rollout_mode not in ROLLOUT_MODES:
        raise ValueError(
            "derive_rollout_info() requires a validated rollout.mode. "
            f"Got: {rollout_mode!r}. Run validate_args() first."
        )
    normalized_rollout_engine = normalize_engine_type(rollout_config.rollout_engine)
    training_actor_sampling_mode = rollout_mode == DIRECT_ROLLOUT_MODE
    if training_actor_sampling_mode:
        if normalized_rollout_engine:
            raise ValueError(
                "direct_sampling is the only public direct-sampling selector. "
                "Leave rollout.rollout_engine unset in direct_sampling mode."
            )
        rollout_engine = None
    else:
        if normalized_rollout_engine not in ROLLOUT_ENGINE_TYPES:
            raise ValueError(
                "rollout.rollout_engine must be one of "
                f"{sorted(ROLLOUT_ENGINE_TYPES)}, got: {rollout_config.rollout_engine!r}"
            )
        rollout_engine = normalized_rollout_engine
    algorithm_type = args.algorithm.algorithm_type
    logprob_source = args.sampling.logprob_source
    replay_enabled = logprob_source == "replay"
    is_sglang_engine = rollout_engine == "sglang"

    engine_caps = _derive_engine_capabilities(
        rollout_engine=rollout_engine,
        training_actor_sampling_mode=training_actor_sampling_mode,
        algorithm_type=algorithm_type,
        logprob_source=logprob_source,
        replay_enabled=replay_enabled,
        is_sglang_engine=is_sglang_engine,
    )

    return RolloutInfo(
        mode=rollout_mode,
        rollout_engine=rollout_engine,
        training_actor_sampling_mode=training_actor_sampling_mode,
        is_sglang_engine=is_sglang_engine,
        logprob_source=logprob_source,
        replay_enabled=replay_enabled,
        sync_protocol=args.sync.protocol,
        algorithm_type=algorithm_type,
        max_samples_per_request=args.sampling.max_samples_per_request,
        effective_engine_capabilities=engine_caps,
    )


def derive_training_topology(
    args: Any,
    *,
    train_backend_config: Optional[BaseTrainBackendConfig] = None,
) -> TrainTopology:
    actor_count = max(
        1,
        args.ray.training_num_nodes * args.ray.training_num_gpus_per_node,
    )
    config = (
        train_backend_config
        if isinstance(train_backend_config, BaseTrainBackendConfig)
        else derive_train_backend_config(args)
    )

    if isinstance(config, VeOmniTrainBackendConfig):
        dp_size = int(config.dp_size or actor_count)
        dp_replicate_size = int(config.dp_replicate_size or 1)
        dp_shard_size = config.dp_shard_size
        if dp_shard_size is None:
            dp_shard_size = max(1, dp_size // max(1, dp_replicate_size))
        return TrainTopology(
            actor_count=actor_count,
            world_size=actor_count,
            dp_size=dp_size,
            dp_replicate_size=dp_replicate_size,
            dp_shard_size=dp_shard_size,
            tp_size=config.tp_size or 1,
            pp_size=config.pp_size or 1,
            sp_size=config.sp_size or 1,
            ep_size=config.ep_size or 1,
        )

    if isinstance(config, MegatronTrainBackendConfig):
        hinted_dp_size = config.dp_size
        tp_size = config.tp_size or 1
        pp_size = config.pp_size or 1
        sp_size = config.sp_size or 1
        ep_size = config.ep_size or 1
        denom = max(1, tp_size * pp_size * sp_size)
        dp_size = (
            hinted_dp_size if hinted_dp_size is not None else max(1, actor_count // denom)
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


def derive_training_plan(
    args: Any,
    *,
    training_topology: TrainTopology,
) -> TrainingPlan:
    global_batch_size = args.algorithm.global_batch_size
    dp_size = training_topology.dp_size
    if global_batch_size % dp_size != 0:
        raise ValueError(
            "Global_batch_size must be divisible by the effective training dp_size. "
            f"Got global_batch_size={global_batch_size}, dp_size={dp_size}. "
            "Adjust algorithm.prompts_per_rollout, algorithm.samples_per_prompt, "
            "or the training backend topology."
        )
    local_batch_size = global_batch_size // dp_size
    num_updates_per_batch = args.training.num_updates_per_batch
    mini_batch_size = local_batch_size // num_updates_per_batch
    raw_micro_batch_size = args.training.micro_batch_size
    micro_batch_size = raw_micro_batch_size or mini_batch_size
    assert micro_batch_size >= 1, (
        f"micro_batch_size must be >= 1, Got {micro_batch_size}"
    )
    assert mini_batch_size >= micro_batch_size, (
        f"mini_batch_size must be >= micro_batch_size, Got {mini_batch_size}"
    )

    update_slices = tuple(
        (update_index * mini_batch_size, (update_index + 1) * mini_batch_size)
        for update_index in range(num_updates_per_batch)
    )
    mini_batch_slices_per_update = tuple(
        _build_relative_slices(total_size=mini_batch_size, chunk_size=micro_batch_size)
        for _ in range(num_updates_per_batch)
    )

    return TrainingPlan(
        global_batch_size=global_batch_size,
        local_batch_size=local_batch_size,
        local_mini_batch_size=mini_batch_size,
        micro_batch_size=micro_batch_size,
        num_updates_per_batch=num_updates_per_batch,
        update_slices=update_slices,
        mini_batch_slices_per_update=mini_batch_slices_per_update,
    )


def derive_config(
    args: Any,
    *,
    existing: Optional[DerivedConfig] = None,
    include_train_backend_capabilities: bool = True,
) -> DerivedConfig:
    if isinstance(existing, DerivedConfig):
        algorithm_dotpath = existing.algorithm_dotpath
        model_spec = existing.model_spec
        sampling_spec = existing.sampling_spec
        rollout_info = existing.rollout_info
        train_backend_config = existing.train_backend_config
        training_topology = existing.training_topology
        train_backend_capabilities = existing.train_backend_capabilities
    else:
        algorithm_dotpath = derive_algorithm_dotpath(args)
        model_spec = derive_model_spec(args)
        sampling_spec = derive_sampling_spec(args, model_spec=model_spec)
        rollout_info = derive_rollout_info(args)
        train_backend_config = derive_train_backend_config(args)
        training_topology = derive_training_topology(
            args,
            train_backend_config=train_backend_config,
        )
        train_backend_capabilities = None

    if include_train_backend_capabilities and train_backend_capabilities is None:
        train_backend_capabilities = derive_train_backend_capabilities(
            train_backend_config
        )

    return DerivedConfig(
        algorithm_dotpath=algorithm_dotpath,
        model_spec=model_spec,
        sampling_spec=sampling_spec,
        rollout_info=rollout_info,
        train_backend_config=train_backend_config,
        training_topology=training_topology,
        train_backend_capabilities=train_backend_capabilities,
        training_plan=None,
    )


def attach_training_plan(
    args: Any,
    *,
    derived_config: DerivedConfig,
    training_plan: Optional[TrainingPlan] = None,
) -> DerivedConfig:
    resolved_training_plan = training_plan
    if resolved_training_plan is None:
        resolved_training_plan = derive_training_plan(
            args,
            training_topology=derived_config.training_topology,
        )
    return replace(derived_config, training_plan=resolved_training_plan)


def _derive_engine_capabilities(
    *,
    rollout_engine: Optional[str],
    training_actor_sampling_mode: bool,
    algorithm_type: str,
    logprob_source: str,
    replay_enabled: bool,
    is_sglang_engine: bool,
) -> Optional[Dict[str, bool]]:
    if training_actor_sampling_mode or not rollout_engine:
        return None

    engine_caps = load_engine_capabilities(rollout_engine)

    allow_replay = replay_enabled and algorithm_type == "grpo"
    if allow_replay:
        engine_caps = dict(engine_caps, requires_log_prob=True, requires_embeddings=True)
        logger.warning(
            "logprob_source='replay' enabled: allowing %s+GRPO with "
            "training-side old-log-prob replay (experimental path).",
            rollout_engine,
        )

    if is_sglang_engine and algorithm_type == "grpo" and logprob_source == "native":
        engine_caps = dict(engine_caps, requires_log_prob=True, requires_embeddings=True)

    return engine_caps


def _build_relative_slices(
    *,
    total_size: int,
    chunk_size: int,
) -> Tuple[Tuple[int, int], ...]:
    slices = []
    start = 0
    while start < total_size:
        end = min(start + chunk_size, total_size)
        slices.append((start, end))
        start = end
    return tuple(slices)


def _build_placement_spec(
    args: Any,
    *,
    rollout_info: RolloutInfo,
) -> PlacementSpec:
    debug_mode = str(args.debug.mode or "none").strip().lower()
    rollout_num_nodes = int(args.ray.rollout_num_nodes)
    rollout_num_gpus_per_node = int(args.ray.rollout_num_gpus_per_node)
    training_num_nodes = int(args.ray.training_num_nodes)
    training_num_gpus_per_node = int(args.ray.training_num_gpus_per_node)

    if debug_mode == "train_only":
        rollout_num_nodes = 0
        rollout_num_gpus_per_node = 0
    elif rollout_info.training_actor_sampling_mode:
        rollout_num_nodes = 0
        rollout_num_gpus_per_node = 0

    return PlacementSpec(
        rollout_num_nodes=rollout_num_nodes,
        rollout_num_gpus_per_node=rollout_num_gpus_per_node,
        training_num_nodes=training_num_nodes,
        training_num_gpus_per_node=training_num_gpus_per_node,
        colocate_rollout=rollout_mode_is_colocated(rollout_info.mode),
        strategy=str(args.ray.placement_strategy),
    )


def build_launch_config(
    args: Any,
    *,
    derived_config: DerivedConfig,
) -> LaunchConfig:
    from diffusionrl.cmdline.algorithms import build_algorithm_init_payload_from_args
    from diffusionrl.cmdline.models import build_model_bundle_init_payload_from_args
    from diffusionrl.cmdline.rollout_engine import (
        build_rollout_engine_init_payload_from_args,
    )
    from diffusionrl.cmdline.train_backend import (
        build_train_backend_init_payload_from_args,
    )
    from diffusionrl.reward.config import RewardSpec

    derived = derived_config
    model_spec = derived.model_spec
    rollout_info = derived.rollout_info
    sampling_spec = derived.sampling_spec
    train_backend_config = derived.train_backend_config
    training_topology = derived.training_topology
    train_backend_capabilities = derived.require_train_backend_capabilities()
    training_plan = derived.require_training_plan()
    train_backend_launch_spec = derive_train_backend_launch_spec(
        train_backend_config,
        args=args,
        topology=training_topology,
    )

    algorithm_init_payload = build_algorithm_init_payload_from_args(
        args,
        sampling_spec=sampling_spec,
    )
    placement = _build_placement_spec(args, rollout_info=rollout_info)
    reward_config = RewardSpec.from_args(args)
    model_init_payload = build_model_bundle_init_payload_from_args(
        args, model_spec=model_spec
    )
    training_sampling_config = build_training_sampling_config_from_args(
        args=args,
        sampling_spec=sampling_spec,
        sampler_engine_type=rollout_info.sampling_engine,
    )
    train_backend_init_payload = build_train_backend_init_payload_from_args(
        args,
        train_backend_config=train_backend_config,
    )
    training_actor_init_config = build_training_actor_init_config_from_args(
        args,
        derived_config=derived,
        training_plan=training_plan,
        algorithm_init_payload=algorithm_init_payload,
        model_init_payload=model_init_payload,
        reward_config=reward_config,
        sampling_config=training_sampling_config,
        train_backend_init_payload=train_backend_init_payload,
    )
    training = TrainingLaunch(
        topology=training_topology,
        backend_name=train_backend_config.name,
        backend_capabilities=train_backend_capabilities,
        launch_spec=train_backend_launch_spec,
        colocate=placement.colocate_rollout,
        colocate_gpu_fraction=float(args.ray.colocate_training_gpu_fraction),
        actor_init_config=training_actor_init_config,
    )

    rollout: Optional[RolloutLaunch] = None
    if placement.rollout_num_nodes > 0 and placement.rollout_num_gpus_per_node > 0:
        rollout_engine = rollout_info.rollout_engine
        if not rollout_engine:
            raise ValueError(
                "Derived rollout launch requires rollout.rollout_engine to be set."
            )
        rollout_engine_init_payload = build_rollout_engine_init_payload_from_args(
            args,
            model_init_payload=model_init_payload,
            sampling_spec=sampling_spec,
            rollout_info=rollout_info,
        )
        rollout_actor_init_config = build_rollout_actor_init_config_from_args(
            args,
            derived_config=derived,
            model_init_payload=model_init_payload,
            reward_config=reward_config,
            engine_init_payload=rollout_engine_init_payload,
        )
        rollout = RolloutLaunch(
            rollout_engine=rollout_engine,
            actor_gpu_requirement=int(args.rollout.num_gpus_per_actor or 0),
            colocate=placement.colocate_rollout,
            colocate_gpu_fraction=float(args.ray.colocate_rollout_gpu_fraction),
            allow_noset_multi_gpu_inference=bool(
                args.ray.allow_noset_multi_gpu_inference
            ),
            engine_init_payload=rollout_engine_init_payload,
            actor_init_config=rollout_actor_init_config,
        )

    raw_target_modules = args.sync.target_modules
    if isinstance(raw_target_modules, (list, tuple)) and raw_target_modules:
        resolved_target_modules = [str(name) for name in raw_target_modules]
    else:
        resolved_target_modules = ["transformer"]

    weight_sync = WeightSyncSpec(
        protocol=str(args.sync.protocol or "").strip().lower(),
        target_modules=resolved_target_modules,
        bucket_size_mb=int(args.sync.bucket_size),
        flush_cache=bool(args.sync.flush_cache),
        sync_dir=str(args.sync.dir),
    )

    rollout_services = RolloutServicesSpec(
        data_source_dotpath=str(args.data_source_dotpath),
        data_source_args=args,
        reward_spec=reward_config,
        prompts_per_rollout=int(args.algorithm.prompts_per_rollout),
        replay_enabled=bool(args.sampling.replay_enabled),
        max_samples_per_request=(
            None
            if args.sampling.max_samples_per_request is None
            else int(args.sampling.max_samples_per_request)
        ),
        evaluation_settings=args.evaluation,
        debug_mode=str(args.debug.mode or "none"),
        debug_output_dir=getattr(args.debug, "output_dir", None),
    )

    return LaunchConfig(
        algorithm_init_payload=algorithm_init_payload,
        training_sampling_config=training_sampling_config,
        rollout_info=rollout_info,
        placement=placement,
        rollout=rollout,
        training=training,
        weight_sync=weight_sync,
        rollout_services=rollout_services,
    )


__all__ = [
    "attach_training_plan",
    "build_launch_config",
    "derive_algorithm_dotpath",
    "derive_algorithm_identifier",
    "derive_config",
    "derive_model_spec",
    "derive_rollout_info",
    "derive_sampling_spec",
    "derive_train_backend_config",
    "derive_train_backend_dotpath",
    "derive_train_backend_identifier",
    "derive_training_plan",
    "derive_training_topology",
]
