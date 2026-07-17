from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from unirl.models.qwen3 import (
    Qwen3Bundle,
    Qwen3PipelineConfig,
    Qwen3TokenValueModel,
    Qwen3ValueBundle,
    Qwen3ValueConfig,
    Qwen3ValuePipeline,
    Qwen3ValueStage,
    configure_value_trainable_parameters,
    inspect_value_checkpoint,
)
from unirl.models.qwen3.conditions import Qwen3ARConditions
from unirl.models.qwen3.replay_layout import (
    build_packed_replay_layout,
    build_padded_replay_layout,
)
from unirl.types.conditions import TextTokenCondition
from unirl.types.segments import TextSegment


class _DenseMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(1, 1, bias=False)
        self.up_proj = nn.Linear(1, 1, bias=False)
        self.down_proj = nn.Linear(1, 1, bias=False)


class _DenseBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(1, 1, bias=False)
        self.mlp = _DenseMLP()
        self.input_layernorm = nn.LayerNorm(1)


class _FakeDenseDecoder(nn.Module):
    def __init__(self, config: SimpleNamespace | None = None) -> None:
        super().__init__()
        self.config = config or SimpleNamespace(hidden_size=1, _attn_implementation="eager")
        self.embed_tokens = nn.Embedding(128, 1)
        self.layers = nn.ModuleList([_DenseBlock()])
        self.norm = nn.LayerNorm(1)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        del attention_mask, position_ids, use_cache, return_dict
        # Token id as hidden state makes the selected causal position observable.
        return SimpleNamespace(last_hidden_state=input_ids.float().unsqueeze(-1))


class _FakeTokenizer:
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"


def _value_bundle() -> Qwen3ValueBundle:
    transformer = Qwen3TokenValueModel(model=_FakeDenseDecoder())
    with torch.no_grad():
        transformer.value_head.weight.fill_(1.0)
    names = configure_value_trainable_parameters(transformer)
    return Qwen3ValueBundle(
        transformer=transformer,
        tokenizer=_FakeTokenizer(),
        dtype=torch.float32,
        device=torch.device("cpu"),
        pretrained_path="fake",
        trainable_parameter_names=names,
    )


def _batch() -> tuple[Qwen3ARConditions, TextSegment]:
    conditions = Qwen3ARConditions(
        prompt=TextTokenCondition(
            input_ids=torch.tensor([[10, 11, 0], [20, 21, 22]]),
            attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
        )
    )
    segment = TextSegment.pack(
        tokens=[torch.tensor([12, 13]), torch.tensor([23])],
        log_probs=[torch.zeros(2), torch.zeros(1)],
    )
    return conditions, segment


def test_value_stage_uses_state_before_each_action_token() -> None:
    conditions, segment = _batch()
    stage = Qwen3ValueStage(model=_value_bundle(), autocast_precision="fp32")

    values = stage.predict_values(conditions, segment=segment)

    # a0 is valued at the last prompt state; a1 at the preceding action state.
    torch.testing.assert_close(values, torch.tensor([11.0, 12.0, 22.0]))
    assert values.dtype == torch.float32


def test_packed_and_padded_value_positions_are_identical() -> None:
    conditions, segment = _batch()
    stage = Qwen3ValueStage(model=_value_bundle(), autocast_precision="fp32")

    padded = stage.padding_predict_values(conditions, segment=segment)
    packed = stage.packed_predict_values(conditions, segment=segment)

    assert packed is not None
    torch.testing.assert_close(packed, padded)


def test_shared_replay_layout_has_actor_critic_prediction_indices() -> None:
    conditions, segment = _batch()
    assert conditions.prompt is not None
    padded = build_padded_replay_layout(
        prompt_ids=conditions.prompt.input_ids,
        prompt_mask=conditions.prompt.attention_mask,
        segment=segment,
        device=torch.device("cpu"),
        pad_id=0,
        caller="test",
    )
    packed = build_packed_replay_layout(
        prompt_ids=conditions.prompt.input_ids,
        prompt_mask=conditions.prompt.attention_mask,
        segment=segment,
        device=torch.device("cpu"),
        pad_id=0,
        caller="test",
    )

    assert packed is not None
    assert padded.prompt_len == 3
    assert padded.input_ids.tolist() == [[0, 10, 11, 12, 13], [20, 21, 22, 23, 0]]
    assert packed.input_ids.tolist() == [[10, 11, 12, 13, 20, 21, 22, 23]]
    assert packed.predict_index.tolist() == [1, 2, 6]


def test_freeze_policy_enables_only_value_head_and_dense_mlp() -> None:
    model = Qwen3TokenValueModel(model=_FakeDenseDecoder())

    names = configure_value_trainable_parameters(model)

    assert set(names) == {
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "value_head.weight",
    }
    assert not model.model.layers[0].self_attn.q_proj.weight.requires_grad
    assert not model.model.embed_tokens.weight.requires_grad
    assert not model.model.norm.weight.requires_grad


class _Experts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.ones(2, 1, 2))
        self.down_proj = nn.Parameter(torch.ones(2, 1, 1))


class _MoEMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(1, 2, bias=False)
        self.experts = _Experts()


class _FakeMoEDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=1)
        block = nn.Module()
        block.mlp = _MoEMLP()
        block.self_attn = nn.Linear(1, 1, bias=False)
        self.layers = nn.ModuleList([block])


def test_freeze_policy_enables_moe_experts_but_not_router() -> None:
    model = Qwen3TokenValueModel(model=_FakeMoEDecoder())

    names = configure_value_trainable_parameters(model)

    assert set(names) == {
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
        "value_head.weight",
    }
    assert not model.model.layers[0].mlp.gate.weight.requires_grad
    assert not model.model.layers[0].self_attn.weight.requires_grad


def test_freeze_policy_fails_if_architecture_has_no_supported_projection() -> None:
    decoder = nn.Module()
    decoder.config = SimpleNamespace(hidden_size=1)
    decoder.attention = nn.Linear(1, 1, bias=False)
    model = Qwen3TokenValueModel(model=decoder)

    with pytest.raises(RuntimeError, match="matched no MLP/expert projections"):
        configure_value_trainable_parameters(model)


def _save_value_artifact(path, *, include_head: bool, include_frozen: bool = True) -> None:
    state = {
        "model.layers.0.mlp.gate_proj.weight": torch.ones(1, 1),
        "model.layers.0.mlp.up_proj.weight": torch.ones(1, 1),
        "model.layers.0.mlp.down_proj.weight": torch.ones(1, 1),
    }
    if include_frozen:
        state.update(
            {
                "model.embed_tokens.weight": torch.ones(128, 1),
                "model.layers.0.self_attn.q_proj.weight": torch.ones(1, 1),
                "model.layers.0.input_layernorm.weight": torch.ones(1),
                "model.layers.0.input_layernorm.bias": torch.zeros(1),
                "model.norm.weight": torch.ones(1),
                "model.norm.bias": torch.zeros(1),
            }
        )
    if include_head:
        state["value_head.weight"] = torch.tensor([[3.0]])
    save_file(state, str(path / "model.safetensors"))


def _install_fake_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("transformers")

    class AutoConfig:
        @staticmethod
        def from_pretrained(path: str, **kwargs):
            del path, kwargs
            return SimpleNamespace(hidden_size=1, _attn_implementation="eager")

    class AutoModel:
        @staticmethod
        def from_pretrained(path: str, **kwargs):
            del path, kwargs
            return _FakeDenseDecoder()

        @staticmethod
        def from_config(config, **kwargs):
            if "attn_implementation" in kwargs:
                config._attn_implementation = kwargs["attn_implementation"]
            return _FakeDenseDecoder(config)

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path: str, **kwargs):
            del path, kwargs
            return _FakeTokenizer()

    module.AutoConfig = AutoConfig
    module.AutoModel = AutoModel
    module.AutoModelForCausalLM = AutoModel
    module.AutoTokenizer = AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", module)


def test_value_bundle_loads_head_and_has_no_lm_head(tmp_path, monkeypatch) -> None:
    _save_value_artifact(tmp_path, include_head=True)
    _install_fake_transformers(monkeypatch)

    bundle = Qwen3ValueBundle.from_config(
        Qwen3ValueConfig(
            pretrained_value_ckpt_path=str(tmp_path),
            model_precision="fp32",
            device="cpu",
        )
    )

    torch.testing.assert_close(bundle.transformer.value_head.weight, torch.tensor([[3.0]]))
    assert not hasattr(bundle.transformer, "lm_head")
    assert "value_head.weight" in bundle.trainable_parameter_names


def test_value_bundle_requires_head_unless_smoke_override(tmp_path, monkeypatch) -> None:
    _save_value_artifact(tmp_path, include_head=False)
    _install_fake_transformers(monkeypatch)

    with pytest.raises(ValueError, match="missing required 'value_head.weight'"):
        Qwen3ValueBundle.from_config(
            Qwen3ValueConfig(
                pretrained_value_ckpt_path=str(tmp_path),
                model_precision="fp32",
                device="cpu",
            )
        )

    bundle = Qwen3ValueBundle.from_config(
        Qwen3ValueConfig(
            pretrained_value_ckpt_path=str(tmp_path),
            model_precision="fp32",
            device="cpu",
            allow_random_value_init=True,
        )
    )
    assert bundle.transformer.value_head.weight.requires_grad


def test_actor_meta_bundle_propagates_attention_implementation(tmp_path, monkeypatch) -> None:
    _install_fake_transformers(monkeypatch)

    bundle = Qwen3Bundle.from_config(
        Qwen3PipelineConfig(
            pretrained_model_ckpt_path=str(tmp_path),
            model_precision="fp32",
            device="cpu",
            meta_init_transformer=True,
            attn_implementation="flex_attention",
        )
    )

    assert bundle.transformer.config._attn_implementation == "flex_attention"


def test_value_meta_bundle_propagates_attention_and_requires_complete_artifact(
    tmp_path, monkeypatch
) -> None:
    _save_value_artifact(tmp_path, include_head=True, include_frozen=True)
    _install_fake_transformers(monkeypatch)

    bundle = Qwen3ValueBundle.from_config(
        Qwen3ValueConfig(
            pretrained_value_ckpt_path=str(tmp_path),
            model_precision="fp32",
            device="cpu",
            meta_init_transformer=True,
            attn_implementation="flex_attention",
        )
    )

    assert bundle.transformer.config._attn_implementation == "flex_attention"


def test_value_meta_bundle_rejects_missing_frozen_parameter(tmp_path, monkeypatch) -> None:
    _save_value_artifact(tmp_path, include_head=True, include_frozen=False)
    _install_fake_transformers(monkeypatch)

    with pytest.raises(ValueError, match=r"missing model weight\(s\)"):
        Qwen3ValueBundle.from_config(
            Qwen3ValueConfig(
                pretrained_value_ckpt_path=str(tmp_path),
                model_precision="fp32",
                device="cpu",
                meta_init_transformer=True,
                attn_implementation="flex_attention",
            )
        )


def test_value_eager_bundle_rejects_partially_random_decoder(tmp_path, monkeypatch) -> None:
    _save_value_artifact(tmp_path, include_head=True, include_frozen=False)
    _install_fake_transformers(monkeypatch)

    with pytest.raises(ValueError, match=r"missing model weight\(s\)"):
        Qwen3ValueBundle.from_config(
            Qwen3ValueConfig(
                pretrained_value_ckpt_path=str(tmp_path),
                model_precision="fp32",
                device="cpu",
            )
        )


def test_checkpoint_manifest_and_config_validation(tmp_path) -> None:
    _save_value_artifact(tmp_path, include_head=True)
    manifest = inspect_value_checkpoint(str(tmp_path))
    assert "value_head.weight" in manifest.keys
    assert any(key.startswith("model.") for key in manifest.keys)

    with pytest.raises(ValueError, match="cannot be used with meta_init"):
        Qwen3ValueConfig(
            pretrained_value_ckpt_path=str(tmp_path),
            meta_init_transformer=True,
            allow_random_value_init=True,
        )


def test_value_pipeline_is_training_only() -> None:
    pipeline = Qwen3ValuePipeline.from_bundle(_value_bundle(), autocast_precision="fp32")
    assert isinstance(pipeline.value, Qwen3ValueStage)
    with pytest.raises(RuntimeError, match="training-only"):
        pipeline.generate(None)  # type: ignore[arg-type]
