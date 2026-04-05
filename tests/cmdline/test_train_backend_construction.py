from types import SimpleNamespace

from diffusionrl.cmdline.train_backend import build_train_backend_init_payload_from_args
from diffusionrl.construction import ComponentInitPayload
from diffusionrl.training.backends import (
    FSDPTrainBackendConfig,
    MegatronTrainBackendConfig,
    VeOmniTrainBackendConfig,
)
from diffusionrl.training.backends.construction import (
    create_train_backend_from_init_payload,
)


def test_build_train_backend_init_payload_from_args_returns_typed_payload():
    args = SimpleNamespace(
        training=SimpleNamespace(
            train_backend="fsdp",
            train_backend_dotpath=None,
            train_backend_kwargs={"use_fsdp": True, "mixed_precision": True},
            fsdp_cpu_offload=False,
            use_gradient_checkpointing=False,
        ),
        precision=SimpleNamespace(
            fsdp_precision="bf16",
        ),
        ray=SimpleNamespace(
            training_num_nodes=1,
            training_num_gpus_per_node=2,
        ),
    )

    payload = build_train_backend_init_payload_from_args(args)

    assert isinstance(payload, ComponentInitPayload)
    assert payload.component_dotpath.endswith("FSDPTrainBackend")
    assert isinstance(payload.component_config, FSDPTrainBackendConfig)
    assert payload.component_config.name == "fsdp"
    assert payload.component_config.use_fsdp is True
    assert payload.component_config.mixed_precision is True
    assert payload.component_config.cpu_offload is False
    assert payload.component_config.param_dtype == "bf16"


def test_build_train_backend_init_payload_from_dotpath_uses_component_config_attr():
    args = SimpleNamespace(
        training=SimpleNamespace(
            train_backend="veomni",
            train_backend_dotpath="diffusionrl.training.backends.veomni_native.VeOmniNativeTrainBackend",
            train_backend_kwargs={"tp_size": 2, "parallelize_kwargs": {"foo": "bar"}},
            fsdp_cpu_offload=False,
            use_gradient_checkpointing=True,
        ),
        precision=SimpleNamespace(
            fsdp_precision="bf16",
        ),
        ray=SimpleNamespace(
            training_num_nodes=1,
            training_num_gpus_per_node=2,
        ),
    )

    payload = build_train_backend_init_payload_from_args(args)

    assert isinstance(payload, ComponentInitPayload)
    assert payload.component_dotpath.endswith("VeOmniNativeTrainBackend")
    assert isinstance(payload.component_config, VeOmniTrainBackendConfig)
    assert payload.component_config.tp_size == 2
    assert payload.component_config.parallelize_kwargs == {"foo": "bar"}


def test_create_train_backend_from_init_payload_constructs_backend():
    payload = ComponentInitPayload(
        component_dotpath="diffusionrl.training.backends.fsdp.FSDPTrainBackend",
        component_config=FSDPTrainBackendConfig(
            cpu_offload=False,
            param_dtype="bf16",
            use_fsdp=True,
            mixed_precision=True,
        ),
    )

    backend = create_train_backend_from_init_payload(payload)

    assert backend.name == "fsdp"
    assert backend._use_fsdp is True


def test_create_megatron_backend_from_init_payload_constructs_backend():
    payload = ComponentInitPayload(
        component_dotpath="diffusionrl.training.backends.megatron.MegatronTrainBackend",
        component_config=MegatronTrainBackendConfig(
            actor_class_path="pkg.module.MegatronActor",
            dp_size=2,
            tp_size=4,
            pp_size=2,
            runtime_env={"env_vars": {"A": "B"}},
            actor_kwargs={"x": 1},
        ),
    )

    backend = create_train_backend_from_init_payload(payload)

    assert backend.name == "megatron"
    assert backend.config.actor_class_path == "pkg.module.MegatronActor"
    assert backend.config.tp_size == 4
    assert backend._actor_class_path == "pkg.module.MegatronActor"
    assert backend._tp_size == 4
    assert backend._pp_size == 2
    assert backend._launch_runtime_env == {"env_vars": {"A": "B"}}
    assert backend._launch_actor_kwargs == {"x": 1}


def test_create_veomni_backend_from_init_payload_constructs_backend():
    payload = ComponentInitPayload(
        component_dotpath="diffusionrl.training.backends.veomni.VeOmniTrainBackend",
        component_config=VeOmniTrainBackendConfig(
            dp_size=2,
            tp_size=2,
            enable_gradient_checkpointing=True,
            parallelize_kwargs={"foo": "bar"},
        ),
    )

    backend = create_train_backend_from_init_payload(payload)

    assert backend.name == "veomni"
    assert backend.config.data_parallel_mode == "fsdp2"
    assert backend._dp_mode == "fsdp2"
    assert backend._dp_size_hint == 2
    assert backend._tp_size == 2
    assert backend._enable_gradient_checkpointing is True
    assert backend._parallelize_extra_kwargs == {"foo": "bar"}
