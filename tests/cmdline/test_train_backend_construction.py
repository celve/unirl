from types import SimpleNamespace

import pytest
import torch

from diffusionrl.cmdline.train_backend import build_train_backend_init_payload_from_args
from diffusionrl.construction import ComponentInitPayload
from diffusionrl.training.backends import (
    FSDPBackendConfig,
    VeOmniBackendConfig,
)


def _base_args(**training_overrides):
    training = dict(
        train_backend="fsdp",
        train_backend_kwargs={},
        fsdp_cpu_offload=False,
        use_gradient_checkpointing=False,
    )
    training.update(training_overrides)
    return SimpleNamespace(
        training=SimpleNamespace(**training),
        precision=SimpleNamespace(fsdp_precision="bf16"),
        ray=SimpleNamespace(
            training_num_nodes=1,
            training_num_gpus_per_node=2,
        ),
    )


def test_build_fsdp_train_backend_init_payload_returns_typed_payload():
    args = _base_args(train_backend_kwargs={"mixed_precision": True})

    payload = build_train_backend_init_payload_from_args(args)

    assert isinstance(payload, ComponentInitPayload)
    assert payload.component_dotpath.endswith("FSDPBackend")
    assert isinstance(payload.component_config, FSDPBackendConfig)
    assert payload.component_config.mixed_precision is True
    assert payload.component_config.cpu_offload is False
    assert payload.component_config.param_dtype == torch.bfloat16
    # Defaulted FSDP sharding knobs land on the new config.
    assert payload.component_config.fsdp_mode == "full"
    assert payload.component_config.reshard_after_forward is True


def test_build_train_backend_init_payload_honors_fsdp_mode_and_reshard():
    args = SimpleNamespace(
        training=SimpleNamespace(
            train_backend="fsdp",
            train_backend_kwargs={},
            fsdp_cpu_offload=False,
            use_gradient_checkpointing=False,
            fsdp_mode="hybrid",
            reshard_after_forward=False,
        ),
        precision=SimpleNamespace(
            fsdp_precision="bf16",
        ),
        ray=SimpleNamespace(
            training_num_nodes=2,
            training_num_gpus_per_node=8,
        ),
    )

    payload = build_train_backend_init_payload_from_args(args)

    assert isinstance(payload.component_config, FSDPBackendConfig)
    assert payload.component_config.fsdp_mode == "hybrid"
    assert payload.component_config.reshard_after_forward is False


def test_build_veomni_train_backend_init_payload_returns_typed_payload():
    args = _base_args(
        train_backend="veomni",
        train_backend_kwargs={
            "tp_size": 2,
            "parallelize_kwargs": {"foo": "bar"},
            "enable_gradient_checkpointing": True,
        },
    )

    payload = build_train_backend_init_payload_from_args(args)

    assert isinstance(payload, ComponentInitPayload)
    assert payload.component_dotpath.endswith("VeOmniBackend")
    assert isinstance(payload.component_config, VeOmniBackendConfig)
    assert payload.component_config.name == "veomni"
    assert payload.component_config.tp_size == 2
    assert payload.component_config.parallelize_kwargs == {"foo": "bar"}
    assert payload.component_config.enable_gradient_checkpointing is True


def test_backend_configs_declare_stable_name_classvar():
    # Regression: cmdline.resolution and cmdline.schema read ``.name`` off
    # the config with no fallback, so a missing ClassVar crashes every
    # launch on that backend.
    assert FSDPBackendConfig.name == "fsdp"
    assert VeOmniBackendConfig.name == "veomni"


def test_megatron_backend_is_rejected():
    args = _base_args(train_backend="megatron")

    with pytest.raises(ValueError, match="Megatron"):
        build_train_backend_init_payload_from_args(args)


def test_unknown_backend_raises():
    args = _base_args(train_backend="not_a_backend")

    with pytest.raises(ValueError, match="Unsupported train_backend"):
        build_train_backend_init_payload_from_args(args)


def test_unknown_train_backend_kwargs_rejected_for_fsdp():
    args = _base_args(train_backend_kwargs={"nonexistent_field": True})

    with pytest.raises(ValueError, match="unsupported keys"):
        build_train_backend_init_payload_from_args(args)


def test_unknown_train_backend_kwargs_rejected_for_veomni():
    args = _base_args(
        train_backend="veomni",
        train_backend_kwargs={"nonexistent_field": True},
    )

    with pytest.raises(ValueError, match="unsupported keys"):
        build_train_backend_init_payload_from_args(args)


# -----------------------------------------------------------------------------
# FSDPBackend static-helper unit tests (no distributed init required)
# -----------------------------------------------------------------------------


def test_fsdp_backend_filter_lora_state_keeps_only_lora_entries():
    from diffusionrl.training.backends.fsdp import FSDPBackend

    state = {
        "transformer.lora_A.default.weight": torch.zeros(2),
        "transformer.LoRA_B.default.weight": torch.zeros(2),
        "transformer.to_q.weight": torch.zeros(2),
        "bias": torch.zeros(1),
    }

    filtered = FSDPBackend._filter_lora_state(state)

    assert set(filtered.keys()) == {
        "transformer.lora_A.default.weight",
        "transformer.LoRA_B.default.weight",
    }


def test_fsdp_backend_extract_peft_lora_state_without_peft_returns_empty(monkeypatch):
    # Block `from peft.utils import ...` from resolving so the helper hits
    # its except-branch and returns {} gracefully.
    import sys

    from diffusionrl.training.backends.fsdp import FSDPBackend

    monkeypatch.setitem(sys.modules, "peft.utils", None)

    result = FSDPBackend._extract_peft_lora_state(torch.nn.Linear(2, 2))

    assert result == {}


def test_fsdp_backend_build_state_dict_options_retries_on_unknown_kwargs(monkeypatch):
    import torch.distributed.checkpoint.state_dict as sd_mod

    from diffusionrl.training.backends.fsdp import FSDPBackend

    observed: list[dict] = []

    class StubStateDictOptions:
        def __init__(self, **kwargs):
            observed.append(kwargs)
            if "broadcast_from_rank0" in kwargs:
                raise TypeError("unexpected keyword 'broadcast_from_rank0'")
            self.kwargs = kwargs

    monkeypatch.setattr(sd_mod, "StateDictOptions", StubStateDictOptions)

    result = FSDPBackend._build_state_dict_options(
        full_state_dict=True,
        broadcast_from_rank0=True,
        cpu_offload=False,
    )

    # First attempt passed every kwarg (and raised); second attempt dropped
    # broadcast_from_rank0 and succeeded.
    assert observed[0] == {
        "full_state_dict": True,
        "broadcast_from_rank0": True,
        "cpu_offload": False,
    }
    assert observed[1] == {"full_state_dict": True, "cpu_offload": False}
    assert isinstance(result, StubStateDictOptions)
    assert result.kwargs == {"full_state_dict": True, "cpu_offload": False}


def test_fsdp_backend_build_state_dict_options_falls_through_to_empty(monkeypatch):
    import torch.distributed.checkpoint.state_dict as sd_mod

    from diffusionrl.training.backends.fsdp import FSDPBackend

    class AlwaysRejects:
        def __init__(self, **kwargs):
            if kwargs:
                raise TypeError("rejects everything")
            self.kwargs = kwargs

    monkeypatch.setattr(sd_mod, "StateDictOptions", AlwaysRejects)

    result = FSDPBackend._build_state_dict_options(
        full_state_dict=True,
        cpu_offload=True,
    )

    # Falls through to the final empty-kwargs construction.
    assert isinstance(result, AlwaysRejects)
    assert result.kwargs == {}


def test_fsdp_backend_config_defaults_include_new_fields():
    cfg = FSDPBackendConfig()

    assert cfg.fsdp_mode == "full"
    assert cfg.reshard_after_forward is True
    assert cfg.param_dtype == torch.bfloat16
    assert cfg.mixed_precision is True
    assert cfg.cpu_offload is False
