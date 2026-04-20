from types import SimpleNamespace

from diffusionrl.cmdline.rollout_engine import (
    build_rollout_engine_init_payload_from_args,
)
from diffusionrl.construction import ComponentInitPayload
from diffusionrl.models.config import ModelBundleConfig
from diffusionrl.types.engine import EngineConfig
from diffusionrl.types.sampling import SamplingParams, SDEConfig


def _make_sampling_spec() -> SamplingParams:
    return SamplingParams(
        num_inference_steps=28,
        guidance_scale=3.5,
        height=64,
        width=64,
        num_frames=1,
        seed=123,
        init_same_noise=False,
        sampler_kwargs={},
        sde_config=SDEConfig(eta=0.5, sde_type="vp", shift=1.0),
    )


def test_build_rollout_engine_init_payload_from_args_returns_typed_payload():
    args = SimpleNamespace(
        precision=SimpleNamespace(rollout_autocast_precision="bf16"),
        rollout=SimpleNamespace(
            num_gpus_per_actor=1,
            tp_size=2,
            sp_size=None,
            sglang_local_mode=False,
            sglang_verify_weight_checksum=True,
            sglang_disable_autocast=False,
            sglang_kwargs={"host": "127.0.0.1", "scheduler_port": 31000},
        ),
        sync=SimpleNamespace(
            dir="/tmp/sync",
            target_modules=["transformer.blocks.0"],
        ),
        sampling=SimpleNamespace(fps=24),
        ray=SimpleNamespace(offload_rollout=True),
    )

    payload = build_rollout_engine_init_payload_from_args(
        args,
        model_init_payload=ComponentInitPayload(
            component_dotpath="diffusionrl.models.sd3.SD3ModelBundle",
            component_config=ModelBundleConfig(
                pretrained_model_ckpt_path="/tmp/model",
                use_lora=False,
            ),
        ),
        sampling_spec=_make_sampling_spec(),
        rollout_info=SimpleNamespace(
            rollout_engine="sglang",
            logprob_source="engine",
        ),
    )

    assert isinstance(payload, ComponentInitPayload)
    assert payload.component_dotpath.endswith("SGLangRolloutEngine")
    assert isinstance(payload.component_config, EngineConfig)
    assert payload.component_config.sampler_dotpath.endswith("DefaultSampler")
    assert payload.component_config.num_inference_steps == 28
    assert payload.component_config.engine_kwargs["num_gpus"] == 1
    assert payload.component_config.engine_kwargs["tp_size"] == 2
    assert payload.component_config.engine_kwargs["server_kwargs"] == {
        "host": "127.0.0.1",
        "scheduler_port": 31000,
    }
    assert payload.component_config.engine_kwargs["logprob_source"] == "engine"
    assert payload.component_config.engine_kwargs["require_memory_api"] is True
