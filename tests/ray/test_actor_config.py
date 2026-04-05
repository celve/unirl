from types import SimpleNamespace

from diffusionrl.cmdline.algorithms import build_algorithm_init_payload_from_args
from diffusionrl.config.build_domain_args import (
    build_rollout_actor_init_config_from_args,
    build_training_actor_init_config_from_args,
)
from diffusionrl.config.validation import (
    validate_rollout_actor_init_config,
    validate_training_actor_init_config,
)
from diffusionrl.construction import ComponentInitPayload
from diffusionrl.models.config import ModelBundleConfig
from diffusionrl.ray.actor_config import RolloutActorConfig, TrainingActorConfig
from diffusionrl.training.backends import BaseTrainBackendConfig, FSDPTrainBackendConfig
from diffusionrl.types.engine import EngineConfig
from diffusionrl.types.sampling import SamplingSpec
from diffusionrl.types.sde import SDEConfig


def _make_algorithm_args():
    return SimpleNamespace(
        algorithm=SimpleNamespace(
            algorithm_type="grpo",
            algorithm_dotpath=None,
            algorithm_kwargs={"clip_range": 1e-3},
            samples_per_prompt=4,
            prompts_per_rollout=1,
            component_mix_stage="reward",
            adv_normalization_scope="group",
            adv_norm_eps=1e-8,
            clip_max=5.0,
            use_global_std=False,
            trim_outliers_ratio=0.0,
            eval_ema_decay=0.9,
            eval_ema_update_interval=1,
            shuffle_samples=True,
            shuffle_seed=7,
            training_share_rollout_indices=True,
            rollout_scheduler={"timestep_strategy": "all", "timestep_fraction": 0.75},
            training_scheduler={"timestep_strategy": "all", "timestep_fraction": 0.5},
        ),
        precision=SimpleNamespace(training_autocast_precision="bf16"),
        debug=SimpleNamespace(debug_output_dir=None),
    )


def _make_sampling_spec() -> SamplingSpec:
    return SamplingSpec(
        sampler_dotpath="diffusionrl.samplers.default_sampler.DefaultSampler",
        num_inference_steps=28,
        guidance_scale=3.5,
        height=64,
        width=64,
        num_frames=1,
        seed=123,
        replay_sampler_dotpath=None,
        sampling_adapter=None,
        init_same_noise=False,
        sampler_kwargs={},
        sde_config=SDEConfig(eta=0.5, sde_type="vp", shift=1.0),
    )


class _TopologyStub:
    def as_dict(self):
        return {
            "actor_count": 2,
            "world_size": 2,
            "dp_size": 2,
        }


class _PlanStub:
    def as_dict(self):
        return {
            "global_batch_size": 8,
            "local_batch_size": 4,
            "local_mini_batch_size": 2,
            "micro_batch_size": 1,
            "num_updates_per_batch": 2,
            "update_slices": [[0, 2], [2, 4]],
            "mini_batch_slices_per_update": [[[0, 1], [1, 2]], [[0, 1], [1, 2]]],
        }


def test_build_training_actor_init_config_from_args_returns_typed_config():
    args = _make_algorithm_args()
    sampling_spec = _make_sampling_spec()
    algorithm_init_payload = build_algorithm_init_payload_from_args(
        args,
        sampling_spec=sampling_spec,
    )

    args.training = SimpleNamespace(
        learning_rate=1e-4,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay=0.01,
        lr_scheduler_type="constant",
        warmup_steps=10,
        max_grad_norm=1.0,
    )
    args.rollout = SimpleNamespace(num_rollout=100)

    config = build_training_actor_init_config_from_args(
        args,
        replay_enabled=False,
        topology=_TopologyStub(),
        training_plan=_PlanStub(),
        algorithm_init_payload=algorithm_init_payload,
        model_init_payload=ComponentInitPayload(
            component_dotpath="diffusionrl.models.sd3.SD3ModelBundle",
            component_config=ModelBundleConfig(
                pretrained_model_ckpt_path="p",
            ),
        ),
        reward_config={"reward_components": []},
        sampling_config={"guidance_scale": 3.5, "sampler_dotpath": "s"},
        train_backend_init_payload=ComponentInitPayload(
            component_dotpath="diffusionrl.training.backends.fsdp.FSDPTrainBackend",
            component_config=FSDPTrainBackendConfig(),
        ),
    )

    assert isinstance(config, TrainingActorConfig)
    assert config.algorithm_init_payload is algorithm_init_payload
    assert isinstance(config.model_init_payload, ComponentInitPayload)
    assert isinstance(config.model_init_payload.component_config, ModelBundleConfig)
    assert isinstance(config.train_backend_init_payload, ComponentInitPayload)
    assert isinstance(
        config.train_backend_init_payload.component_config, BaseTrainBackendConfig
    )
    assert config.training_config["guidance_scale"] == 3.5

    validate_training_actor_init_config(config)


def test_build_rollout_actor_init_config_from_args_returns_typed_config():
    args = _make_algorithm_args()
    args.model = SimpleNamespace()
    args.training = SimpleNamespace()
    args.precision = SimpleNamespace(
        rollout_autocast_precision="bf16",
    )
    args.rollout = SimpleNamespace(
        num_gpus_per_actor=1,
        tp_size=None,
        sp_size=None,
        transport_dtype=None,
        transport_drop_decoded_videos=None,
        sglang_local_mode=None,
        sglang_verify_weight_checksum=None,
        sglang_disable_autocast=None,
        sglang_kwargs={},
        rollout_batch_size=4,
    )
    args.logging = SimpleNamespace(
        transport_log_payload_bytes=None,
    )
    args.sync = SimpleNamespace(
        dir="/tmp/sync",
        target_modules=None,
    )
    args.sampling = SimpleNamespace(
        fps=24,
    )
    args.ray = SimpleNamespace(
        offload_rollout=False,
    )

    config = build_rollout_actor_init_config_from_args(
        args,
        config_bundle=SimpleNamespace(
            sampling_spec=_make_sampling_spec(),
            model_spec=SimpleNamespace(
                model_dotpath="diffusionrl.models.sd3.SD3ModelBundle"
            ),
            rollout_mode_info=SimpleNamespace(
                logprob_source="engine",
                rollout_topology=SimpleNamespace(rollout_engine="sglang"),
            ),
        ),
        model_init_payload=ComponentInitPayload(
            component_dotpath="diffusionrl.models.sd3.SD3ModelBundle",
            component_config=ModelBundleConfig(
                pretrained_model_ckpt_path="p",
                use_lora=False,
            ),
        ),
        reward_config={"reward_components": []},
    )

    assert isinstance(config, RolloutActorConfig)
    assert isinstance(config.engine_init_payload, ComponentInitPayload)
    assert isinstance(config.engine_init_payload.component_config, EngineConfig)
    assert config.engine_init_payload.component_config.sampler_dotpath.endswith(
        "DefaultSampler"
    )
    assert config.reward_config["reward_components"] == []
    assert config.rollout_batch_size == 4

    validate_rollout_actor_init_config(config)
