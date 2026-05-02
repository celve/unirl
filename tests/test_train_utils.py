"""Tests for ``diffusionrl.utils.train_utils``.

The resume-rollout-id resolver writes back into a sealed cfg in production
(``cfg.resume`` is registered with ``mutable=True``); previously this path
was uncovered. These tests pin the contract:

- ``checkpoint-N`` directory name → ``start_rollout_id == N + 1``.
- No checkpoint path / non-matching name / already-set start id → ``None``.
- The write succeeds against a freshly composed-and-frozen cfg.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# Trigger @register_config side effects.
import diffusionrl  # noqa: F401
from diffusionrl.algorithms import grpo as _grpo  # noqa: F401
from diffusionrl.algorithms import nft as _nft  # noqa: F401
from diffusionrl.config import debug_config as _debug_config  # noqa: F401
from diffusionrl.config import evaluation_config as _evaluation_config  # noqa: F401
from diffusionrl.config import logging_config as _logging_config  # noqa: F401
from diffusionrl.config import resume_config as _resume_config  # noqa: F401
from diffusionrl.config import run_config as _run_config  # noqa: F401
from diffusionrl.config.instantiate import freeze
from diffusionrl.distributed import weight_sync as _weight_sync  # noqa: F401
from diffusionrl.models import flux as _flux  # noqa: F401
from diffusionrl.ray import placement as _placement  # noqa: F401
from diffusionrl.ray.train_actor import TrainingExecutionConfig  # noqa: F401
from diffusionrl.reward import config as _reward_config  # noqa: F401
from diffusionrl.samplers.fsdp import engine as _fsdp_engine  # noqa: F401
from diffusionrl.samplers.sglang import engine as _sglang_engine  # noqa: F401
from diffusionrl.training.backends import fsdp as _fsdp  # noqa: F401
from diffusionrl.training.backends import veomni as _veomni  # noqa: F401
from diffusionrl.types import sampling as _sampling  # noqa: F401
from diffusionrl.utils.train_utils import maybe_restore_start_rollout_id_from_checkpoint

_CONF_DIR = str(Path(__file__).resolve().parent.parent / "conf")


@pytest.fixture
def frozen_cfg():
    """Compose the real cfg and apply freeze, mirroring train.main."""
    with initialize_config_dir(config_dir=_CONF_DIR, version_base=None):
        cfg = compose(config_name="train")
        OmegaConf.resolve(cfg)
        freeze(cfg)
        yield cfg


def test_returns_none_when_path_empty(frozen_cfg):
    assert maybe_restore_start_rollout_id_from_checkpoint(frozen_cfg, "") is None
    assert maybe_restore_start_rollout_id_from_checkpoint(frozen_cfg, None) is None


def test_returns_none_when_start_rollout_id_already_set(frozen_cfg):
    frozen_cfg.resume.start_rollout_id = 5
    assert maybe_restore_start_rollout_id_from_checkpoint(frozen_cfg, "/some/path/checkpoint-42") is None


def test_returns_none_when_path_does_not_match_pattern(frozen_cfg):
    assert maybe_restore_start_rollout_id_from_checkpoint(frozen_cfg, "/some/random/dir") is None


def test_writes_next_rollout_id_into_frozen_resume(frozen_cfg):
    """The whole point: cfg.resume must be writable after freeze."""
    assert frozen_cfg.resume.start_rollout_id == 0

    result = maybe_restore_start_rollout_id_from_checkpoint(frozen_cfg, "/work/runs/exp42/checkpoint-17")

    assert result == 18
    assert frozen_cfg.resume.start_rollout_id == 18


def test_handles_trailing_slash_in_path(frozen_cfg):
    result = maybe_restore_start_rollout_id_from_checkpoint(frozen_cfg, "/work/runs/exp42/checkpoint-3/")
    assert result == 4
    assert frozen_cfg.resume.start_rollout_id == 4
