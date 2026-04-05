from types import SimpleNamespace

import pytest

from diffusionrl.algorithms.construction import create_algorithm_from_init_payload
from diffusionrl.algorithms.grpo import GRPOAlgorithm, GRPOAlgorithmConfig
from diffusionrl.algorithms.nft import NFTAlgorithm, NFTAlgorithmConfig
from diffusionrl.cmdline.algorithms import build_algorithm_init_payload_from_args
from diffusionrl.construction import ComponentInitPayload
from diffusionrl.types.sampling import SamplingSpec
from diffusionrl.types.sde import SDEConfig


def _make_args(*, algorithm_type: str, algorithm_kwargs: dict | None = None):
    return SimpleNamespace(
        algorithm=SimpleNamespace(
            algorithm_type=algorithm_type,
            algorithm_dotpath=None,
            algorithm_kwargs=dict(algorithm_kwargs or {}),
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
            shuffle_seed=None,
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


def test_build_algorithm_init_payload_uses_grpo_algorithm_config():
    args = _make_args(
        algorithm_type="grpo",
        algorithm_kwargs={"clip_range": 1e-3, "skip_last_timestep": True},
    )
    algorithm_init_payload = build_algorithm_init_payload_from_args(
        args, sampling_spec=_make_sampling_spec()
    )

    assert isinstance(algorithm_init_payload, ComponentInitPayload)
    assert algorithm_init_payload.component_dotpath.endswith(".GRPOAlgorithm")
    assert isinstance(algorithm_init_payload.component_config, GRPOAlgorithmConfig)
    assert algorithm_init_payload.component_config.clip_range == 1e-3
    assert algorithm_init_payload.component_config.num_inference_steps == 28


def test_build_algorithm_init_payload_uses_nft_algorithm_config():
    args = _make_args(
        algorithm_type="nft",
        algorithm_kwargs={"beta": 0.25, "train_timestep_mode": "fixed"},
    )
    algorithm_init_payload = build_algorithm_init_payload_from_args(
        args, sampling_spec=_make_sampling_spec()
    )

    assert isinstance(algorithm_init_payload.component_config, NFTAlgorithmConfig)
    assert algorithm_init_payload.component_dotpath.endswith(".NFTAlgorithm")
    assert algorithm_init_payload.component_config.beta == 0.25
    assert (
        algorithm_init_payload.component_config.training_scheduler_config[
            "timestep_fraction"
        ]
        == 0.5
    )


def test_create_algorithm_from_grpo_init_payload():
    args = _make_args(
        algorithm_type="grpo",
        algorithm_kwargs={"ratio_reg_coef": 0.2, "model_type": "flux"},
    )
    algorithm_init_payload = build_algorithm_init_payload_from_args(
        args, sampling_spec=_make_sampling_spec()
    )
    algorithm = create_algorithm_from_init_payload(algorithm_init_payload)

    assert isinstance(algorithm, GRPOAlgorithm)
    assert isinstance(algorithm.config, GRPOAlgorithmConfig)
    assert algorithm.config.ratio_reg_coef == 0.2
    assert algorithm.model_type == "flux"


def test_build_algorithm_init_payload_allows_grpo_kl_coef_in_algorithm_kwargs():
    args = _make_args(
        algorithm_type="grpo",
        algorithm_kwargs={"kl_coef": 0.2, "clip_range": 1e-3},
    )
    algorithm_init_payload = build_algorithm_init_payload_from_args(
        args, sampling_spec=_make_sampling_spec()
    )

    assert algorithm_init_payload.component_config.kl_coef == 0.2
    assert algorithm_init_payload.component_config.clip_range == 1e-3


def test_create_algorithm_from_nft_init_payload():
    args = _make_args(
        algorithm_type="nft",
        algorithm_kwargs={"beta": 0.3, "ema_decay": 0.02},
    )
    algorithm_init_payload = build_algorithm_init_payload_from_args(
        args, sampling_spec=_make_sampling_spec()
    )
    algorithm = create_algorithm_from_init_payload(algorithm_init_payload)

    assert isinstance(algorithm, NFTAlgorithm)
    assert isinstance(algorithm.config, NFTAlgorithmConfig)
    assert algorithm.beta == 0.3
    assert algorithm.ema_decay == 0.02


def test_build_algorithm_init_payload_rejects_unknown_grpo_algorithm_kwargs():
    args = _make_args(
        algorithm_type="grpo",
        algorithm_kwargs={"clip_range": 1e-3, "nonexistent_knob": 1},
    )

    with pytest.raises(
        ValueError,
        match=r"unsupported keys for GRPOAlgorithmConfig: \['nonexistent_knob'\]",
    ):
        build_algorithm_init_payload_from_args(
            args, sampling_spec=_make_sampling_spec()
        )


def test_build_algorithm_init_payload_rejects_base_config_keys_inside_algorithm_kwargs():
    args = _make_args(
        algorithm_type="nft",
        algorithm_kwargs={"beta": 0.2, "component_mix_stage": "advantage"},
    )

    with pytest.raises(
        ValueError,
        match=r"unsupported keys for NFTAlgorithmConfig: \['component_mix_stage'\]",
    ):
        build_algorithm_init_payload_from_args(
            args, sampling_spec=_make_sampling_spec()
        )
