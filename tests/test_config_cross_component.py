"""Negative + positive tests for the cross-component validators.

Each Cat A validator from diffusionrl/config/validation.py gets:
  1. A positive test — default composition passes.
  2. A negative test — mutate the relevant cross-component fields to
     violate the invariant, assert the validator raises.

Exercises the validators in isolation; the ``_run_cross_component_
validators`` helper in ``train.py`` composes them in order and is
exercised indirectly by the default-composition test in
``test_cfg_validation.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _helpers import unseal_for_testing
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import diffusionrl  # noqa: F401
from diffusionrl.algorithms import grpo as _grpo  # noqa: F401
from diffusionrl.algorithms import nft as _nft  # noqa: F401
from diffusionrl.config import debug_config as _debug_config  # noqa: F401
from diffusionrl.config import evaluation_config as _evaluation_config  # noqa: F401
from diffusionrl.config import logging_config as _logging_config  # noqa: F401
from diffusionrl.config import resume_config as _resume_config  # noqa: F401
from diffusionrl.config import run_config as _run_config  # noqa: F401
from diffusionrl.config.validation import (
    is_direct_sampling,
    validate_dynamic_dotpaths,
    validate_offload_contract,
    validate_rollout_layout,
    validate_training_actor_sampling_mode,
    validate_training_batch_geometry,
    validate_weight_sync_contract,
)
from diffusionrl.distributed import weight_sync as _weight_sync  # noqa: F401
from diffusionrl.models import flux as _flux  # noqa: F401
from diffusionrl.ray import placement as _placement  # noqa: F401
from diffusionrl.ray.train_actor import TrainingExecutionConfig  # noqa: F401
from diffusionrl.reward import config as _reward_config  # noqa: F401
from diffusionrl.samplers.fsdp import engine as _fsdp_engine  # noqa: F401
from diffusionrl.samplers.sglang import engine as _sglang_engine  # noqa: F401
from diffusionrl.training.backends import fsdp as _fsdp  # noqa: F401
from diffusionrl.types import sampling as _sampling  # noqa: F401

_CONF_DIR = str(Path(__file__).resolve().parent.parent / "conf")


@pytest.fixture
def cfg():
    with initialize_config_dir(config_dir=_CONF_DIR, version_base=None):
        composed = compose(config_name="train")
        unseal_for_testing(composed)
        yield composed


_FSDP_ENGINE_TARGET = "diffusionrl.samplers.fsdp.engine.FSDPSamplingEngine"
_SGLANG_ENGINE_TARGET = "diffusionrl.samplers.sglang.engine.SGLangRolloutEngine"
_TENSOR_SYNC_TARGET = "diffusionrl.distributed.weight_sync.tensor.UpdateWeightFromTensor"
_NCCL_SYNC_TARGET = "diffusionrl.distributed.weight_sync.nccl.UpdateWeightFromDistributed"


class TestIsDirectSampling:
    def test_default_is_direct(self, cfg):
        # Default composition wires rollout/engine: fsdp → direct sampling.
        assert is_direct_sampling(cfg) is True

    def test_fsdp_engine_target_flips_to_direct(self, cfg):
        cfg.rollout.engine._target_ = _FSDP_ENGINE_TARGET
        assert is_direct_sampling(cfg) is True


class TestValidateDynamicDotpaths:
    def test_default_passes(self, cfg):
        validate_dynamic_dotpaths(cfg)

    def test_rejects_empty(self, cfg):
        cfg.run.data_source_dotpath = ""
        with pytest.raises(ValueError, match="data_source_dotpath must be a non-empty dotpath"):
            validate_dynamic_dotpaths(cfg)

    def test_rejects_unimportable(self, cfg):
        cfg.run.data_source_dotpath = "diffusionrl.nonexistent.Module"
        with pytest.raises(ValueError, match="failed to import"):
            validate_dynamic_dotpaths(cfg)


class TestValidateTrainingActorSamplingMode:
    def test_default_passes(self, cfg):
        validate_training_actor_sampling_mode(cfg)

    def test_rejects_direct_sampling_with_unsupported_backend(self, cfg, monkeypatch):
        from diffusionrl.training import types as training_types

        cfg.rollout.engine._target_ = _FSDP_ENGINE_TARGET
        # Pin a synthetic backend that declares no support.
        from diffusionrl.training.backends.base import TrainBackendCapabilities

        fake = TrainBackendCapabilities(name="fake", supports_training_actor_sampling=False)
        monkeypatch.setattr(training_types, "resolve_train_backend_capabilities", lambda _name: fake)
        with pytest.raises(ValueError, match="supports_training_actor_sampling=True"):
            validate_training_actor_sampling_mode(cfg)


class TestValidateTrainingBatchGeometry:
    def test_default_passes(self, cfg):
        validate_training_batch_geometry(cfg)

    def test_rejects_indivisible_dp_size(self, cfg):
        cfg.training.plan.global_batch_size = 3
        cfg.training.topology.dp_size = 2
        with pytest.raises(ValueError, match="must be divisible by .* dp_size"):
            validate_training_batch_geometry(cfg)

    def test_rejects_indivisible_dp_replicate_size(self, cfg):
        cfg.training.plan.global_batch_size = 4
        cfg.training.topology.dp_size = 2
        cfg.training.topology.dp_replicate_size = 3
        with pytest.raises(ValueError, match="must be divisible by .* dp_replicate_size"):
            validate_training_batch_geometry(cfg)


class TestValidateWeightSyncContract:
    def test_default_passes(self, cfg):
        # Bare compose is fsdp + no sync = valid direct-sampling shape.
        validate_weight_sync_contract(cfg)

    def test_direct_sampling_forbids_sync_section(self, cfg):
        # Default is direct (fsdp). Add a sync section ⇒ violation.
        OmegaConf.update(cfg, "sync", {"_target_": _TENSOR_SYNC_TARGET}, force_add=True)
        with pytest.raises(ValueError, match="forbids a sync section"):
            validate_weight_sync_contract(cfg)

    def test_non_direct_requires_sync_section(self, cfg):
        # Flip engine to sglang; bare cfg has no sync ⇒ violation.
        cfg.rollout.engine._target_ = _SGLANG_ENGINE_TARGET
        with pytest.raises(ValueError, match="requires a sync variant"):
            validate_weight_sync_contract(cfg)

    def test_tensor_protocol_requires_sglang(self, cfg):
        # Engine is non-direct but not sglang ⇒ tensor sync fails the third rule.
        cfg.rollout.engine._target_ = "diffusionrl.samplers.fake.FakeRolloutEngine"
        OmegaConf.update(cfg, "sync", {"_target_": _TENSOR_SYNC_TARGET}, force_add=True)
        with pytest.raises(ValueError, match="requires the sglang rollout engine"):
            validate_weight_sync_contract(cfg)


class TestValidateRolloutLayout:
    def test_default_passes(self, cfg):
        validate_rollout_layout(cfg)

    def test_rejects_multi_gpu_colocate_without_sglang(self, cfg):
        cfg.placement.colocate = True
        cfg.placement.num_rollout_gpus_per_actor = 2
        cfg.rollout.engine._target_ = "diffusionrl.samplers.fake.FakeRolloutEngine"
        with pytest.raises(ValueError, match="multi-GPU colocated rollout .* requires the sglang engine"):
            validate_rollout_layout(cfg)


class TestValidateOffloadContract:
    def test_default_passes(self, cfg):
        validate_offload_contract(cfg)

    def test_rejects_direct_sampling_with_offload_train(self, cfg):
        cfg.rollout.engine._target_ = _FSDP_ENGINE_TARGET
        cfg.training.execution.offload_train = True
        with pytest.raises(ValueError, match="offload_train=True"):
            validate_offload_contract(cfg)

    def test_rejects_direct_sampling_with_offload_rollout(self, cfg):
        cfg.rollout.engine._target_ = _FSDP_ENGINE_TARGET
        cfg.training.execution.offload_rollout = True
        with pytest.raises(ValueError, match="offload_rollout=True"):
            validate_offload_contract(cfg)
