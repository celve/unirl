"""Pure-function helper tests for the multi-track TrainActor.

Exercises module-level helpers from :mod:`diffusionrl.training.track_builder`
without Ray or GPU.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from diffusionrl.training.track_builder import (
    _build_algorithm_for_track,
    _detect_lora_on_model,
    _resolve_primary_track,
)

# ---------------------------------------------------------------------------
# _resolve_primary_track
# ---------------------------------------------------------------------------


def test_resolve_primary_track_defaults_to_first_key():
    cfg = OmegaConf.create({"training": {"tracks": {"image": {}, "refined": {}}}})
    assert _resolve_primary_track(cfg) == "image"


def test_resolve_primary_track_honors_explicit_selector():
    cfg = OmegaConf.create({"training": {"primary_track": "refined", "tracks": {"image": {}, "refined": {}}}})
    assert _resolve_primary_track(cfg) == "refined"


def test_resolve_primary_track_rejects_unknown_selector():
    cfg = OmegaConf.create({"training": {"primary_track": "phantom", "tracks": {"image": {}, "refined": {}}}})
    with pytest.raises(ValueError, match="primary_track.*not a key"):
        _resolve_primary_track(cfg)


# ---------------------------------------------------------------------------
# _build_algorithm_for_track
# ---------------------------------------------------------------------------


class _FakeDiffusionStage:
    def __init__(self, init_value: float = 0.5) -> None:
        self.param = nn.Parameter(torch.tensor(float(init_value)))

    def diffuse(self, *args, **kwargs):
        raise NotImplementedError

    def replay(self, *args, **kwargs):
        raise NotImplementedError


def test_build_algorithm_for_track_resolves_stage_via_stage_attr_default():
    pipeline = MagicMock()
    pipeline.image = _FakeDiffusionStage()

    alg_node = OmegaConf.create(
        {
            "_target_": "diffusionrl.algorithms.grpo.DiffusionGRPO",
            "clip_range": 0.2,
        }
    )
    alg = _build_algorithm_for_track(
        alg_node,
        track_name="image",
        pipeline=pipeline,
        stage=pipeline.image,
        ema=None,
    )
    assert alg.stage is pipeline.image


def test_build_algorithm_for_track_honors_stage_attr_override():
    pipeline = MagicMock()
    pipeline.diffusion = _FakeDiffusionStage()

    alg_node = OmegaConf.create(
        {
            "_target_": "diffusionrl.algorithms.grpo.DiffusionGRPO",
            "clip_range": 0.2,
            "stage_attr": "diffusion",
        }
    )
    alg = _build_algorithm_for_track(
        alg_node,
        track_name="image",
        pipeline=pipeline,
        stage=pipeline.diffusion,
        ema=None,
    )
    assert alg.stage is pipeline.diffusion


def test_build_algorithm_for_track_strips_control_keys_from_instantiate():
    pipeline = MagicMock()
    pipeline.image = _FakeDiffusionStage()

    alg_node = OmegaConf.create(
        {
            "_target_": "diffusionrl.algorithms.grpo.DiffusionGRPO",
            "clip_range": 0.2,
            "stage_attr": "image",
            "params_attr": "image_params",
            "conditions_cls": None,
        }
    )
    _build_algorithm_for_track(
        alg_node,
        track_name="image",
        pipeline=pipeline,
        stage=pipeline.image,
        ema=None,
    )


def test_build_algorithm_for_track_injects_sampling_params_for_diffusion_algorithm():
    class _Pipeline:
        pass

    pipeline = _Pipeline()
    pipeline.image = _FakeDiffusionStage()
    sampling_params = object()

    alg_node = OmegaConf.create(
        {
            "_target_": "diffusionrl.algorithms.grpo.DiffusionGRPO",
            "clip_range": 0.2,
        }
    )
    alg = _build_algorithm_for_track(
        alg_node,
        track_name="image",
        pipeline=pipeline,
        stage=pipeline.image,
        ema=None,
        sampling_params=sampling_params,
    )
    assert alg.params is sampling_params


def test_build_algorithm_for_track_nft_requires_ema():
    pipeline = MagicMock()
    pipeline.image = _FakeDiffusionStage()

    alg_node = OmegaConf.create(
        {
            "_target_": "diffusionrl.algorithms.nft.DiffusionNFT",
            "beta": 1.0,
        }
    )
    with pytest.raises(RuntimeError, match="requires ema_lora"):
        _build_algorithm_for_track(
            alg_node,
            track_name="image",
            pipeline=pipeline,
            stage=pipeline.image,
            ema=None,
        )


# ---------------------------------------------------------------------------
# _detect_lora_on_model
# ---------------------------------------------------------------------------


def test_detect_lora_on_model_returns_false_for_plain_module():
    model = nn.Linear(4, 4)
    assert _detect_lora_on_model(model) is False


def test_detect_lora_on_model_returns_true_when_peft_config_present():
    model = nn.Linear(4, 4)
    model.peft_config = {"default": object()}
    assert _detect_lora_on_model(model) is True


def test_detect_lora_on_model_walks_into_module_attribute():
    inner = nn.Linear(4, 4)
    inner.peft_config = {"default": object()}
    wrapper = nn.Linear(4, 4)
    wrapper.module = inner
    assert _detect_lora_on_model(wrapper) is True


def test_detect_lora_on_model_walks_into_base_model_attribute():
    inner = nn.Linear(4, 4)
    inner.peft_config = {"default": object()}
    wrapper = nn.Linear(4, 4)
    wrapper.base_model = inner
    assert _detect_lora_on_model(wrapper) is True
