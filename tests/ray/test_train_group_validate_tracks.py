"""Tests for ``_validate_cfg_for_train`` under the multi-track schema.

The validator is the preflight gate inside :class:`TrainActorGroup.__init__`.
It runs against the composed Hydra cfg before any actor is spawned, so we
test it directly via :class:`omegaconf.OmegaConf` constructions without
touching Ray or any placement infrastructure.

Coverage:

- Happy path: a minimal two-track cfg validates.
- Missing or empty ``cfg.training.tracks`` raises (clean break: the
  legacy flat-cfg shape is no longer accepted).
- Per-track required fields (``model._target_``, ``source_stage_attr``,
  ``policies`` non-empty, each policy carries ``_target_``, ``optimizer``
  / ``lr_scheduler`` set) each raise with a track-keyed message.
- ``cfg.algorithm.algorithms`` keys must equal ``cfg.training.tracks`` keys
  (every track needs exactly one algorithm; the key IS the track name).
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from diffusionrl.training.validate import _validate_cfg_for_train


def _make_track(
    *,
    model_target: str = "fake.Pipeline",
    source_stage_attr: str = "diffusion",
    policies=None,
    optimizer=None,
    lr_scheduler=None,
):
    return {
        "model": {"_target_": model_target},
        "source_stage_attr": source_stage_attr,
        "policies": policies
        if policies is not None
        else [{"_target_": "fake.LoRAPolicy"}, {"_target_": "fake.FSDPPolicy"}],
        "optimizer": optimizer if optimizer is not None else {"learning_rate": 1e-4},
        "lr_scheduler": lr_scheduler if lr_scheduler is not None else {"type": "constant"},
    }


def _make_cfg(*, tracks=None, algorithms=None):
    """Build a minimal OmegaConf with the new multi-track shape."""
    if tracks is None:
        tracks = {
            "image": _make_track(source_stage_attr="diffusion"),
            "refined": _make_track(source_stage_attr="ar", model_target="fake.LLMPipeline"),
        }
    if algorithms is None:
        algorithms = {
            "image": {"_target_": "fake.DiffusionGRPO"},
            "refined": {"_target_": "fake.ARGRPO"},
        }
    return OmegaConf.create({"training": {"tracks": tracks}, "algorithm": {"algorithms": algorithms}})


def test_validator_accepts_minimal_two_track_cfg():
    cfg = _make_cfg()
    _validate_cfg_for_train(cfg)  # must not raise


def test_validator_accepts_single_track_cfg():
    """The shape collapses cleanly to a single-track recipe."""
    cfg = _make_cfg(
        tracks={"image": _make_track(source_stage_attr="diffusion")},
        algorithms={"image": {"_target_": "fake.DiffusionGRPO"}},
    )
    _validate_cfg_for_train(cfg)


def test_validator_rejects_missing_tracks():
    cfg = OmegaConf.create({"training": {}, "algorithm": {"algorithms": {"image": {"_target_": "x"}}}})
    with pytest.raises(ValueError, match="cfg.training.tracks must be a non-empty mapping"):
        _validate_cfg_for_train(cfg)


def test_validator_rejects_empty_tracks():
    cfg = _make_cfg(tracks={}, algorithms={})
    with pytest.raises(ValueError, match="cfg.training.tracks must be a non-empty mapping"):
        _validate_cfg_for_train(cfg)


def test_validator_rejects_missing_algorithms():
    cfg = OmegaConf.create(
        {"training": {"tracks": {"image": _make_track()}}, "algorithm": {}},
    )
    with pytest.raises(ValueError, match="cfg.algorithm.algorithms must be a non-empty"):
        _validate_cfg_for_train(cfg)


def test_validator_rejects_key_mismatch_between_tracks_and_algorithms():
    cfg = _make_cfg(
        tracks={"image": _make_track(), "refined": _make_track(source_stage_attr="ar")},
        algorithms={"image": {"_target_": "x"}, "phantom": {"_target_": "y"}},
    )
    with pytest.raises(ValueError, match="cfg.algorithm.algorithms keys.*must match cfg.training.tracks keys"):
        _validate_cfg_for_train(cfg)


def test_validator_rejects_algorithm_without_target():
    cfg = _make_cfg(
        tracks={"image": _make_track()},
        algorithms={"image": {}},
    )
    with pytest.raises(ValueError, match=r"cfg.algorithm.algorithms.image must carry _target_"):
        _validate_cfg_for_train(cfg)


def test_validator_rejects_track_model_without_target():
    bad = _make_track()
    bad["model"] = {}  # no _target_
    cfg = _make_cfg(tracks={"image": bad}, algorithms={"image": {"_target_": "x"}})
    with pytest.raises(ValueError, match=r"cfg.training.tracks.image.model must carry _target_"):
        _validate_cfg_for_train(cfg)


def test_validator_rejects_missing_source_stage_attr():
    bad = _make_track()
    bad["source_stage_attr"] = ""
    cfg = _make_cfg(tracks={"image": bad}, algorithms={"image": {"_target_": "x"}})
    with pytest.raises(ValueError, match=r"cfg.training.tracks.image.source_stage_attr must be set"):
        _validate_cfg_for_train(cfg)


def test_validator_rejects_empty_policies():
    bad = _make_track(policies=[])
    cfg = _make_cfg(tracks={"image": bad}, algorithms={"image": {"_target_": "x"}})
    with pytest.raises(ValueError, match=r"cfg.training.tracks.image.policies must be a non-empty list"):
        _validate_cfg_for_train(cfg)


def test_validator_rejects_policy_without_target():
    bad = _make_track(policies=[{}])
    cfg = _make_cfg(tracks={"image": bad}, algorithms={"image": {"_target_": "x"}})
    with pytest.raises(ValueError, match=r"cfg.training.tracks.image.policies\[0\] must carry _target_"):
        _validate_cfg_for_train(cfg)


def test_validator_rejects_missing_optimizer():
    bad = _make_track()
    del bad["optimizer"]
    cfg = _make_cfg(tracks={"image": bad}, algorithms={"image": {"_target_": "x"}})
    with pytest.raises(ValueError, match=r"cfg.training.tracks.image.optimizer must be set"):
        _validate_cfg_for_train(cfg)


def test_validator_rejects_missing_lr_scheduler():
    bad = _make_track()
    del bad["lr_scheduler"]
    cfg = _make_cfg(tracks={"image": bad}, algorithms={"image": {"_target_": "x"}})
    with pytest.raises(ValueError, match=r"cfg.training.tracks.image.lr_scheduler must be set"):
        _validate_cfg_for_train(cfg)
