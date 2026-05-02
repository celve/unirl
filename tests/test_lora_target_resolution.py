"""Tests for ``validate_lora_target_modules`` + SGLang ``build_server_kwargs``.

Verifies the resolution priority (explicit cfg > class default > None+warn)
and that the materialised list is forwarded into SGLang ``ServerArgs``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pytest
from _helpers import unseal_for_testing
from hydra import compose, initialize_config_dir

import diffusionrl  # noqa: F401
from diffusionrl.algorithms import grpo as _grpo  # noqa: F401
from diffusionrl.algorithms import nft as _nft  # noqa: F401
from diffusionrl.config.validation import validate_lora_target_modules
from diffusionrl.models import flux as _flux  # noqa: F401
from diffusionrl.models import sd3 as _sd3  # noqa: F401
from diffusionrl.models.config import ModelBundleConfig
from diffusionrl.samplers.sglang import engine as _sglang_engine  # noqa: F401
from diffusionrl.samplers.sglang.config import SGLangEngineConfig

_CONF_DIR = str(Path(__file__).resolve().parent.parent / "conf")


@pytest.fixture
def hydra_context():
    with initialize_config_dir(config_dir=_CONF_DIR, version_base=None):
        yield


def _compose_sd3_lora():
    cfg = compose(
        config_name="train",
        overrides=["+experiment=flowgrpo_fast_sd3"],
    )
    unseal_for_testing(cfg)
    return cfg


def test_validator_materialises_sd3_default(hydra_context):
    cfg = _compose_sd3_lora()
    cfg.model.use_lora = True
    cfg.model.lora_target_modules = None
    validate_lora_target_modules(cfg)
    materialised = list(cfg.model.lora_target_modules)
    assert len(materialised) > 0
    # SD3 default targets joint-attention keys (Flow-GRPO canonical set)
    assert any("attn" in m for m in materialised)


def test_validator_respects_explicit_override(hydra_context):
    cfg = _compose_sd3_lora()
    cfg.model.use_lora = True
    explicit = ["custom_layer.q", "custom_layer.k"]
    cfg.model.lora_target_modules = list(explicit)
    validate_lora_target_modules(cfg)
    assert list(cfg.model.lora_target_modules) == explicit


def test_validator_skipped_when_lora_disabled(hydra_context):
    cfg = _compose_sd3_lora()
    cfg.model.use_lora = False
    cfg.model.lora_target_modules = None
    validate_lora_target_modules(cfg)
    assert cfg.model.lora_target_modules is None


def test_validator_warns_when_class_returns_none(hydra_context, caplog):
    """Stub a model class whose default_lora_target_modules returns None."""
    cfg = _compose_sd3_lora()
    cfg.model.use_lora = True
    cfg.model.lora_target_modules = None
    # Point _target_ at a stub class with a None classmethod.
    cfg.model._target_ = "tests.test_lora_target_resolution._StubBundleNoneDefault"
    with caplog.at_level("WARNING"):
        validate_lora_target_modules(cfg)
    assert cfg.model.lora_target_modules is None
    assert any("default_lora_target_modules() returned None" in r.message for r in caplog.records)


class _StubBundleNoneDefault:
    @classmethod
    def default_lora_target_modules(cls):
        return None


def _fake_server_args_cls():
    """Synthetic ``ServerArgs`` shape exposing the LoRA fields we care about."""

    @dataclass
    class _ServerArgs:
        model_path: Optional[str] = None
        num_gpus: int = 1
        tp_size: Optional[int] = None
        sp_degree: Optional[int] = None
        disable_autocast: bool = False
        lora_merge_mode: Optional[str] = None
        lora_target_modules: Optional[List[str]] = None
        host: Optional[str] = None
        port: Optional[int] = None
        scheduler_port: Optional[int] = None
        master_port: Optional[int] = None

    return _ServerArgs


def _make_engine_cfg() -> SGLangEngineConfig:
    return SGLangEngineConfig(
        local_mode=True,
        num_gpus=1,
        tp_size=1,
    )


def _make_model_cfg(*, use_lora: bool, target_modules: Optional[List[str]]) -> ModelBundleConfig:
    return ModelBundleConfig(
        pretrained_model_ckpt_path="/tmp/fake-ckpt",
        use_lora=use_lora,
        lora_rank=8,
        lora_alpha=8,
        lora_target_modules=target_modules,
    )


def test_build_server_kwargs_forwards_lora_targets():
    engine_cfg = _make_engine_cfg()
    model_cfg = _make_model_cfg(
        use_lora=True,
        target_modules=["attn.add_q_proj", "attn.add_k_proj"],
    )
    kwargs = engine_cfg.build_server_kwargs(_fake_server_args_cls(), model_config=model_cfg)
    assert kwargs.get("lora_target_modules") == ["attn.add_q_proj", "attn.add_k_proj"]
    assert kwargs.get("lora_merge_mode") == "online"


def test_build_server_kwargs_skips_targets_when_lora_disabled():
    engine_cfg = _make_engine_cfg()
    model_cfg = _make_model_cfg(use_lora=False, target_modules=["attn.add_q_proj"])
    kwargs = engine_cfg.build_server_kwargs(_fake_server_args_cls(), model_config=model_cfg)
    assert "lora_target_modules" not in kwargs


def test_build_server_kwargs_skips_targets_when_unresolved():
    engine_cfg = _make_engine_cfg()
    model_cfg = _make_model_cfg(use_lora=True, target_modules=None)
    kwargs = engine_cfg.build_server_kwargs(_fake_server_args_cls(), model_config=model_cfg)
    assert "lora_target_modules" not in kwargs
    # lora_merge_mode still gets the default when use_lora=True.
    assert kwargs.get("lora_merge_mode") == "online"
