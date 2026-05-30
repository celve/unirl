"""Unit tests for :class:`SGLangLLMEngineConfig` dataclass validation."""

from __future__ import annotations

import pytest

from diffusionrl.rollout.engine.sglang_llm.config import SGLangLLMEngineConfig


def test_minimal_valid_config() -> None:
    cfg = SGLangLLMEngineConfig(pretrained_model_ckpt_path="/tmp/model")
    assert cfg.pretrained_model_ckpt_path == "/tmp/model"
    assert cfg.tp_size is None
    assert cfg.concurrency == 8
    assert cfg.max_new_tokens == 512
    assert cfg.temperature == pytest.approx(0.7)
    assert cfg.top_p == pytest.approx(0.9)
    assert cfg.engine_kwargs == {}


def test_engine_kwargs_none_becomes_empty_dict() -> None:
    cfg = SGLangLLMEngineConfig(pretrained_model_ckpt_path="/tmp/model", engine_kwargs=None)
    assert cfg.engine_kwargs == {}


def test_enable_lora_engine_kwarg_accepted() -> None:
    """Native LoRA-pool mode is opt-in again; enable_lora must not be rejected."""
    cfg = SGLangLLMEngineConfig(
        pretrained_model_ckpt_path="/tmp/model",
        engine_kwargs={"enable_lora": True, "max_lora_rank": 16, "lora_target_modules": ["q_proj"]},
    )
    assert cfg.engine_kwargs["enable_lora"] is True


def test_missing_model_path_rejected() -> None:
    with pytest.raises(ValueError, match="pretrained_model_ckpt_path must be set"):
        SGLangLLMEngineConfig(pretrained_model_ckpt_path="")


def test_tp_size_zero_rejected() -> None:
    with pytest.raises(ValueError, match="tp_size must be >= 1"):
        SGLangLLMEngineConfig(pretrained_model_ckpt_path="/tmp/m", tp_size=0)


def test_tp_size_one_accepted() -> None:
    cfg = SGLangLLMEngineConfig(pretrained_model_ckpt_path="/tmp/m", tp_size=1)
    assert cfg.tp_size == 1


def test_concurrency_zero_rejected() -> None:
    with pytest.raises(ValueError, match="concurrency must be >= 1"):
        SGLangLLMEngineConfig(pretrained_model_ckpt_path="/tmp/m", concurrency=0)


def test_max_new_tokens_zero_rejected() -> None:
    with pytest.raises(ValueError, match="max_new_tokens must be >= 1"):
        SGLangLLMEngineConfig(pretrained_model_ckpt_path="/tmp/m", max_new_tokens=0)


def test_temperature_zero_rejected() -> None:
    with pytest.raises(ValueError, match="temperature must be > 0"):
        SGLangLLMEngineConfig(pretrained_model_ckpt_path="/tmp/m", temperature=0.0)


def test_top_p_zero_rejected() -> None:
    with pytest.raises(ValueError, match=r"top_p must be in \(0, 1\]"):
        SGLangLLMEngineConfig(pretrained_model_ckpt_path="/tmp/m", top_p=0.0)


def test_top_p_above_one_rejected() -> None:
    with pytest.raises(ValueError, match=r"top_p must be in \(0, 1\]"):
        SGLangLLMEngineConfig(pretrained_model_ckpt_path="/tmp/m", top_p=1.5)


def test_top_p_exactly_one_accepted() -> None:
    cfg = SGLangLLMEngineConfig(pretrained_model_ckpt_path="/tmp/m", top_p=1.0)
    assert cfg.top_p == 1.0


def test_register_config_attaches_target() -> None:
    """The @register_config decorator should add a _target_ field on the class."""
    cfg = SGLangLLMEngineConfig(pretrained_model_ckpt_path="/tmp/m")
    target = getattr(cfg, "_target_", None)
    assert target == "diffusionrl.rollout.engine.sglang_llm.engine.SGLangLLMRolloutEngine"
