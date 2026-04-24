from __future__ import annotations

from types import SimpleNamespace

import pytest

from diffusionrl.cmdline.schema import get_default_args
from diffusionrl.cmdline.validation import (
    validate_algorithm_kwargs_payload,
    validate_dynamic_dotpaths,
    validate_grouped_configs,
    validate_nft_sampling_contract,
    validate_offload_and_colocate_config,
    validate_rollout_mode,
    validate_rollout_mode_constraints,
    validate_rollout_topology_contract,
)
from diffusionrl.config.spec import RolloutInfo


def test_validate_grouped_configs_rejects_invalid_choice() -> None:
    args = get_default_args()
    args.reward.reward_backend = "bad-backend"

    with pytest.raises(ValueError, match="reward.reward_backend must be one of"):
        validate_grouped_configs(args)


def test_sampling_config_validate_rejects_unknown_sde_type() -> None:
    args = get_default_args()
    args.sampling.sde_type = "not-a-real-sde"

    with pytest.raises(ValueError, match="Unknown sampling.sde_type"):
        args.sampling.validate()


def test_training_config_validate_rejects_non_list_lora_modules() -> None:
    args = get_default_args()
    args.training.lora_target_modules = "proj"

    with pytest.raises(ValueError, match="must be a list of strings or null"):
        args.training.validate()


def test_validate_dynamic_dotpaths_rejects_invalid_data_source_path() -> None:
    args = SimpleNamespace(
        data_source_dotpath="bad.module.fn",
        training=SimpleNamespace(),
        sampling=SimpleNamespace(replay_sampler_dotpath=None),
        rollout=SimpleNamespace(plugin_dotpaths=[]),
    )

    with pytest.raises(ValueError, match="Invalid data_source path"):
        validate_dynamic_dotpaths(
            args,
            resolved_model=SimpleNamespace(
                sampler_dotpath=("diffusionrl.samplers.fsdp.hunyuan_sampler.FSDPHunyuanSampler")
            ),
            algorithm_dotpath="diffusionrl.algorithms.grpo.GRPOAlgorithm",
        )


def test_validate_nft_sampling_contract_requires_sampling_adapter() -> None:
    args = SimpleNamespace(
        algorithm=SimpleNamespace(
            algorithm_type="nft",
            algorithm_kwargs={"old_adapter_name": "old"},
        ),
        sampling=SimpleNamespace(
            sampling_adapter=None,
            sde_type="dpm2",
            eta=0.0,
        ),
    )

    with pytest.raises(ValueError, match="requires --sampling.sampling-adapter"):
        validate_nft_sampling_contract(args)


def test_validate_rollout_mode_constraints_rejects_model_without_sglang_support() -> None:
    class _ModelWithoutSGLang:
        pass

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
    )

    with pytest.raises(ValueError, match="supports_sglang_prompt_mode"):
        validate_rollout_mode_constraints(
            rollout_info=rollout_info,
            model_cls=_ModelWithoutSGLang,
        )


def test_validate_rollout_topology_contract_rejects_direct_sampling_with_service_fields() -> None:
    args = get_default_args()
    args.rollout.mode = "direct_sampling"
    args.rollout.rollout_engine = None
    args.rollout.num_gpus_per_actor = 1
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

    with pytest.raises(ValueError, match="must be unset"):
        validate_rollout_topology_contract(args, rollout_info=rollout_info)


def test_validate_algorithm_kwargs_payload_rejects_reserved_key() -> None:
    args = SimpleNamespace(
        algorithm=SimpleNamespace(
            algorithm_kwargs={"samples_per_prompt": 8},
        )
    )

    with pytest.raises(ValueError, match="framework-owned keys"):
        validate_algorithm_kwargs_payload(args)


def test_validate_rollout_mode_wraps_underlying_validation_error() -> None:
    args = get_default_args()
    args.rollout.mode = "direct_sampling"
    args.rollout.rollout_engine = None
    args.rollout.num_gpus_per_actor = None
    args.rollout.tp_size = None
    args.rollout.sp_size = None
    args.rollout.sglang_local_mode = False
    args.rollout.sglang_verify_weight_checksum = False
    args.rollout.sglang_disable_autocast = False
    args.rollout.sglang_kwargs = {}
    args.ray.offload_train = True
    args.ray.offload_rollout = False
    args.ray.colocate_training_gpu_fraction = 0.5
    args.ray.colocate_rollout_gpu_fraction = 0.5
    args.sampling.max_samples_per_request = None
    args.sync.protocol = "disabled"
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

    with pytest.raises(ValueError, match="Invalid rollout mode while validating offload/colocate settings"):
        validate_rollout_mode(
            args,
            rollout_info=rollout_info,
            backend_capabilities={"supports_training_actor_sampling": True},
            backend_name="fsdp",
        )


def test_validate_offload_and_colocate_config_rejects_offload_in_direct_sampling() -> None:
    args = SimpleNamespace(
        ray=SimpleNamespace(
            offload_train=True,
            offload_rollout=False,
            colocate_training_gpu_fraction=0.5,
            colocate_rollout_gpu_fraction=0.5,
        )
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

    with pytest.raises(ValueError, match="cannot be combined with ray.offload_train / ray.offload_rollout"):
        validate_offload_and_colocate_config(args, rollout_info=rollout_info)
