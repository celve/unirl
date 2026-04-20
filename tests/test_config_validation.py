from __future__ import annotations

from types import SimpleNamespace

import pytest

from diffusionrl.algorithms.base import BaseAlgorithmConfig
from diffusionrl.config.spec import RolloutInfo
from diffusionrl.config.spec import TrainingPlan
from diffusionrl.config.validation import (
    validate_colocate_fractions,
    validate_direct_sampling_batch_geometry,
    validate_dotpath,
    validate_engine_algorithm_contract,
    validate_precision_type,
    validate_reward_config,
    validate_rollout_actor_init_config,
    validate_rollout_engine_config,
    validate_rollout_layout,
    validate_train_backend_config,
    validate_training_actor_init_config,
    validate_training_actor_sampling_mode,
    validate_training_batch_geometry,
    validate_weight_sync,
)
from diffusionrl.config.training_sections import (
    LrSchedulerConfig as TrainingLrSchedulerConfig,
    OptimizerConfig as TrainingOptimizerConfig,
    TrainingExecutionConfig,
)
from diffusionrl.construction import ComponentInitPayload
from diffusionrl.models.config import ModelBundleConfig
from diffusionrl.ray.actor_config import (
    RolloutActorConfig,
    TrainingActorConfig,
)
from diffusionrl.reward.config import RewardSpec
from diffusionrl.training.backends import FSDPBackendConfig
from diffusionrl.training.types import BaseTrainBackendConfig, TrainTopology
from diffusionrl.types.engine import EngineConfig
from diffusionrl.types.sampling import SamplingRequirements


def _make_reward_spec() -> RewardSpec:
    return RewardSpec(
        reward_dotpath=None,
        reward_model_ckpt_path=None,
        reward_batch_size=1,
        local_reward_device="cpu",
        reward_backend="local",
        reward_service_urls=None,
        reward_components=["hpsv2"],
        reward_weights=None,
        reward_aggregation_method="mean",
    )


def _make_training_plan() -> TrainingPlan:
    return TrainingPlan(
        global_batch_size=4,
        local_batch_size=4,
        local_mini_batch_size=2,
        micro_batch_size=1,
        num_updates_per_batch=2,
        update_slices=((0, 2), (2, 4)),
        mini_batch_slices_per_update=(((0, 1), (1, 2)), ((2, 3), (3, 4))),
    )


def test_validate_colocate_fractions_rejects_sum_above_one() -> None:
    args = SimpleNamespace(
        ray=SimpleNamespace(
            colocate_training_gpu_fraction=0.7,
            colocate_rollout_gpu_fraction=0.4,
        )
    )

    with pytest.raises(ValueError, match="must be <= 1.0"):
        validate_colocate_fractions(
            colocate_training_gpu_fraction=args.ray.colocate_training_gpu_fraction,
            colocate_rollout_gpu_fraction=args.ray.colocate_rollout_gpu_fraction,
        )


def test_validate_dotpath_rejects_invalid_path() -> None:
    with pytest.raises(ValueError, match="Invalid algorithm path"):
        validate_dotpath("bad.module.symbol", label="algorithm")


def test_validate_precision_type_rejects_unknown_precision() -> None:
    with pytest.raises(ValueError, match="precision.model_precision must be one of"):
        validate_precision_type("weird16", field_name="precision.model_precision")


def test_validate_engine_algorithm_contract_rejects_missing_capabilities() -> None:
    rollout_info = RolloutInfo(
        mode="separate",
        rollout_engine="sglang",
        training_actor_sampling_mode=False,
        is_sglang_engine=True,
        logprob_source="engine",
        replay_enabled=False,
        sync_protocol="tensor_payload",
        algorithm_type="grpo",
        max_samples_per_request=None,
        effective_engine_capabilities={},
    )

    with pytest.raises(ValueError, match="Engine capability mismatch"):
        validate_engine_algorithm_contract(
            algorithm_type="grpo",
            rollout_info=rollout_info,
            effective_engine_capabilities={},
            sampling_requirements=SamplingRequirements(
                requires_trajectory=True,
                requires_log_prob=True,
                requires_embeddings=True,
            ),
        )


def test_validate_reward_config_rejects_invalid_backend() -> None:
    reward_config = RewardSpec(
        reward_dotpath=None,
        reward_model_ckpt_path=None,
        reward_batch_size=1,
        local_reward_device="cpu",
        reward_backend="weird",
        reward_service_urls=None,
        reward_components=["hpsv2"],
        reward_weights=None,
        reward_aggregation_method="mean",
    )

    with pytest.raises(ValueError, match="reward_backend must be one of local/http"):
        validate_reward_config(reward_config)


def test_validate_rollout_engine_config_requires_sampler_dotpath() -> None:
    config = EngineConfig(sampler_dotpath="")

    with pytest.raises(ValueError, match="sampler_dotpath is required"):
        validate_rollout_engine_config(config)


def test_validate_rollout_actor_init_config_rejects_non_reward_spec() -> None:
    config = RolloutActorConfig(
        engine_init_payload=ComponentInitPayload(
            component_dotpath="diffusionrl.samplers.sglang_engine.SGLangRolloutEngine",
            component_config=EngineConfig(
                sampler_dotpath="diffusionrl.samplers.default_sampler.DefaultSampler",
            ),
        ),
        reward_config="bad",
    )

    with pytest.raises(ValueError, match="reward_config must be a RewardSpec"):
        validate_rollout_actor_init_config(config)


def test_validate_rollout_layout_rejects_non_dedicated_rollout_mode() -> None:
    args = SimpleNamespace(
        rollout=SimpleNamespace(num_gpus_per_actor=1),
        ray=SimpleNamespace(
            rollout_num_nodes=1,
            rollout_num_gpus_per_node=1,
            training_num_nodes=1,
            training_num_gpus_per_node=1,
            allow_noset_multi_gpu_inference=False,
        ),
    )
    rollout_info = RolloutInfo(
        mode="direct_sampling",
        rollout_engine="sglang",
        training_actor_sampling_mode=False,
        is_sglang_engine=False,
        logprob_source="replay",
        replay_enabled=True,
        sync_protocol="tensor_payload",
        algorithm_type="grpo",
        max_samples_per_request=None,
    )

    with pytest.raises(ValueError, match="only applies to dedicated rollout engines"):
        validate_rollout_layout(
            rollout_info=rollout_info,
            rollout_num_nodes=args.ray.rollout_num_nodes,
            rollout_num_gpus_per_node=args.ray.rollout_num_gpus_per_node,
            training_num_nodes=args.ray.training_num_nodes,
            training_num_gpus_per_node=args.ray.training_num_gpus_per_node,
            rollout_num_gpus_per_actor=args.rollout.num_gpus_per_actor,
            allow_noset_multi_gpu_inference=args.ray.allow_noset_multi_gpu_inference,
        )


def test_validate_train_backend_config_accepts_new_style_configs() -> None:
    # New-style configs carry no ``name`` field; identifier validation
    # lives upstream in ``resolve_train_backend_identifier``.
    validate_train_backend_config(train_backend_config=FSDPBackendConfig())


def test_validate_training_actor_init_config_rejects_non_reward_spec() -> None:
    config = TrainingActorConfig(
        model_init_payload=ComponentInitPayload(
            component_dotpath="diffusionrl.models.sd3.SD3ModelBundle",
            component_config=ModelBundleConfig(
                pretrained_model_ckpt_path="/tmp/model",
            ),
        ),
        algorithm_init_payload=ComponentInitPayload(
            component_dotpath="diffusionrl.algorithms.grpo.GRPOAlgorithm",
            component_config=BaseAlgorithmConfig(),
        ),
        train_backend_init_payload=ComponentInitPayload(
            component_dotpath="diffusionrl.training.backends.fsdp.FSDPBackend",
            component_config=FSDPBackendConfig(),
        ),
        sampling_config={},
        reward_config="bad",
        optimizer_config=TrainingOptimizerConfig(type="adamw"),
        scheduler_config=TrainingLrSchedulerConfig(),
        training_config=TrainingExecutionConfig(
            max_grad_norm=1.0,
            replay_enabled=False,
        ),
        training_plan_config=_make_training_plan(),
        topology_config=TrainTopology(world_size=1, dp_size=1, actor_count=1),
    )

    with pytest.raises(ValueError, match="reward_config must be a RewardSpec"):
        validate_training_actor_init_config(config)


def test_validate_training_actor_sampling_mode_rejects_backend_without_support() -> None:
    rollout_info = RolloutInfo(
        mode="direct_sampling",
        rollout_engine=None,
        training_actor_sampling_mode=True,
        is_sglang_engine=False,
        logprob_source="replay",
        replay_enabled=True,
        sync_protocol="disabled",
        algorithm_type="grpo",
        max_samples_per_request=None,
    )

    with pytest.raises(ValueError, match="does not declare supports_training_actor_sampling=true"):
        validate_training_actor_sampling_mode(
            rollout_info=rollout_info,
            backend_capabilities={"supports_training_actor_sampling": False},
            backend_name="fsdp",
        )


def test_validate_training_batch_geometry_rejects_non_divisible_micro_batch() -> None:
    args = SimpleNamespace(
        algorithm=SimpleNamespace(
            prompts_per_rollout=4,
            samples_per_prompt=1,
            global_batch_size=4,
        ),
        training=SimpleNamespace(
            num_updates_per_batch=2,
            micro_batch_size=3,
        ),
    )
    topology = TrainTopology(world_size=1, dp_size=1, actor_count=1)

    with pytest.raises(ValueError, match="local mini-batch cannot be split evenly into micro-batches"):
        validate_training_batch_geometry(
            prompts_per_rollout=args.algorithm.prompts_per_rollout,
            samples_per_prompt=args.algorithm.samples_per_prompt,
            global_batch_size=args.algorithm.global_batch_size,
            num_updates_per_batch=args.training.num_updates_per_batch,
            micro_batch_size=args.training.micro_batch_size,
            topology=topology,
        )


def test_validate_weight_sync_rejects_non_disabled_protocol_in_direct_sampling() -> None:
    args = SimpleNamespace(
        sync=SimpleNamespace(protocol="tensor_payload", dir="/tmp/sync"),
        ray=SimpleNamespace(
            rollout_num_nodes=1,
            training_num_nodes=1,
        ),
    )
    rollout_info = RolloutInfo(
        mode="direct_sampling",
        rollout_engine=None,
        training_actor_sampling_mode=True,
        is_sglang_engine=False,
        logprob_source="replay",
        replay_enabled=True,
        sync_protocol="disabled",
        algorithm_type="grpo",
        max_samples_per_request=None,
    )

    with pytest.raises(ValueError, match="requires sync.protocol='disabled'"):
        validate_weight_sync(
            rollout_info=rollout_info,
            sync_protocol=args.sync.protocol,
            sync_dir=args.sync.dir,
            rollout_num_nodes=args.ray.rollout_num_nodes,
            training_num_nodes=args.ray.training_num_nodes,
        )


def test_validate_direct_sampling_batch_geometry_rejects_non_direct_mode() -> None:
    args = SimpleNamespace(
        sampling=SimpleNamespace(max_samples_per_request=2),
    )
    rollout_info = RolloutInfo(
        mode="separate",
        rollout_engine="sglang",
        training_actor_sampling_mode=False,
        is_sglang_engine=True,
        logprob_source="engine",
        replay_enabled=False,
        sync_protocol="tensor_payload",
        algorithm_type="grpo",
        max_samples_per_request=2,
    )

    with pytest.raises(ValueError, match="only valid when sampling runs directly on training actors"):
        validate_direct_sampling_batch_geometry(
            rollout_info=rollout_info,
            max_samples_per_request=args.sampling.max_samples_per_request,
        )
