"""Tests for ``diffusionrl.config.instantiate.validate``.

The helper walks the cfg tree and materializes every node backed by a
``@register_config`` schema (either with ``_target_`` or schema-only),
discarding the result. Cover:

- Default composition round-trips cleanly.
- Optional sections (``cfg.sampling``) that are absent are skipped.
- Every registered leaf in the default cfg gets visited — the walk
  picks them up automatically via ``OmegaConf.get_type``, with no
  hardcoded section list.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _helpers import unseal_for_testing
from hydra import compose, initialize_config_dir

# Trigger @register_config side effects.
import diffusionrl  # noqa: F401
from diffusionrl.algorithms import grpo as _grpo  # noqa: F401
from diffusionrl.algorithms import nft as _nft  # noqa: F401
from diffusionrl.config import debug_config as _debug_config  # noqa: F401
from diffusionrl.config import evaluation_config as _evaluation_config  # noqa: F401
from diffusionrl.config import instantiate as _instantiate
from diffusionrl.config import logging_config as _logging_config  # noqa: F401
from diffusionrl.config import resume_config as _resume_config  # noqa: F401
from diffusionrl.config import run_config as _run_config  # noqa: F401
from diffusionrl.config.instantiate import validate
from diffusionrl.distributed import weight_sync as _weight_sync  # noqa: F401
from diffusionrl.models import flux as _flux  # noqa: F401
from diffusionrl.models import sd3 as _sd3  # noqa: F401
from diffusionrl.ray import placement as _placement  # noqa: F401
from diffusionrl.ray.train_actor import TrainingExecutionConfig  # noqa: F401
from diffusionrl.reward import config as _reward_config  # noqa: F401
from diffusionrl.samplers.fsdp import engine as _fsdp_engine  # noqa: F401
from diffusionrl.samplers.sglang import engine as _sglang_engine  # noqa: F401
from diffusionrl.training.backends import fsdp as _fsdp  # noqa: F401
from diffusionrl.types import sampling as _sampling  # noqa: F401

_CONF_DIR = str(Path(__file__).resolve().parent.parent / "conf")


@pytest.fixture
def hydra_context():
    with initialize_config_dir(config_dir=_CONF_DIR, version_base=None):
        yield


# Several registered schemas have required fields without defaults
# (model.pretrained_model_ckpt_path, reward.aggregation_method,
# reward.components, ...). Experiment YAMLs supply them; tests compose with
# one so validate (which now exercises real OmegaConf.to_object
# missing-mandatory checks) sees a complete cfg.
def _compose_train(*extra_overrides: str):
    return compose(
        config_name="train",
        overrides=["+experiment=nft_sd3", *extra_overrides],
    )


def test_validate_accepts_default_composition(hydra_context):
    cfg = _compose_train()
    validate(cfg)  # must not raise


def test_validate_skips_absent_sampling(hydra_context):
    cfg = _compose_train()
    unseal_for_testing(cfg)
    cfg.sampling = None
    validate(cfg)  # must not raise — sampling is optional


def test_validate_rejects_bad_training_plan(hydra_context):
    """Mutating cfg.training.plan to violate TrainingPlan.__post_init__ must raise."""
    cfg = _compose_train()
    unseal_for_testing(cfg)
    cfg.training.plan.global_batch_size = 0
    with pytest.raises(ValueError, match="global_batch_size must be >= 1"):
        validate(cfg)


def test_validate_rejects_bad_sync_interval(hydra_context):
    """Mutating cfg.run.weight_sync_interval must raise."""
    cfg = _compose_train()
    unseal_for_testing(cfg)
    cfg.run.weight_sync_interval = 0
    with pytest.raises(ValueError, match="weight_sync_interval must be >= 1"):
        validate(cfg)


def test_validate_rejects_bad_forward_batch_size(hydra_context):
    """Mutating cfg.rollout.plan.forward_batch_size to violate RolloutPlan must raise."""
    cfg = _compose_train("rollout/engine=sglang")
    unseal_for_testing(cfg)
    cfg.rollout.plan.forward_batch_size = 0
    with pytest.raises(ValueError, match="forward_batch_size must be >= 1 when set"):
        validate(cfg)


def test_validate_visits_every_registered_leaf(hydra_context, monkeypatch):
    """Spy on materialize to confirm every known section is materialized."""
    visited: list[str] = []
    original = _instantiate.materialize

    def _spy(leaf):
        full_key = leaf._get_full_key(None)
        visited.append(full_key or "<root>")
        return original(leaf)

    monkeypatch.setattr(_instantiate, "materialize", _spy)

    cfg = _compose_train()
    validate(cfg)

    # `sync` is intentionally absent from the nft_sd3 experiment (direct-sampling
    # mode without a separate rollout group), so it's not in the cfg and not
    # visited. Tests that need to exercise the sync schema compose with a
    # different experiment.
    # ``_iter_typed_leaves`` stops at the outermost typed dataclass; nested
    # dataclass fields are validated transitively as part of the parent's
    # ``to_object`` round-trip, not as separate visits.
    expected = {
        "model",
        "algorithm",
        "reward",
        "sampling",
        "placement",
        "training.backend",
        "training.optimizer",
        "training.lr_scheduler",
        "training.execution",
        "training.plan",
        "training.topology",
        "rollout.engine",
        "logging",
        "debug",
        "evaluation",
        "resume",
    }
    missing = expected - set(visited)
    assert not missing, f"validate failed to visit sections: {missing}"
