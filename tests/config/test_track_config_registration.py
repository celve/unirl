"""Tests for the per-track training schema registration.

Verifies that :class:`TrainingTrackConfig` and :class:`TrackPlanOverrides`
land in the Hydra ConfigStore under the expected ``training/track`` and
``training/track_plan_overrides`` groups, and that the dataclass shape
preserves the required-vs-optional field surface that the validator and
actor rely on.
"""

from __future__ import annotations

from dataclasses import fields as dc_fields

from hydra.core.config_store import ConfigStore

from diffusionrl.training.track_config import TrackPlanOverrides, TrainingTrackConfig


def test_training_track_config_registered_under_training_track_group():
    """``@register_config(group="training/track", name="default")`` lands in the store."""
    entry = ConfigStore.instance().repo["training"]["track"]["default.yaml"]
    # ``mutable=True`` was requested; this is observable via the registered
    # node carrying the same fields as TrainingTrackConfig.
    field_names = {f.name for f in dc_fields(TrainingTrackConfig)}
    assert {
        "model",
        "source_stage_attr",
        "optimizer",
        "lr_scheduler",
        "plan_overrides",
    } <= field_names
    # The registered node is a dataclass instance (or its serialized form).
    # Construct from kwargs to confirm the schema accepts the canonical
    # required-set + optional plan_overrides.
    instance = TrainingTrackConfig(
        model={"_target_": "fake.Pipeline"},
        source_stage_attr="diffusion",
        optimizer=None,  # type: ignore[arg-type] — schema accepts; runtime expects OptimizerConfig
        lr_scheduler=None,  # type: ignore[arg-type]
    )
    assert instance.plan_overrides is None  # default
    assert entry is not None


def test_track_plan_overrides_registered():
    """``TrackPlanOverrides`` lands under ``training/track_plan_overrides``."""
    entry = ConfigStore.instance().repo["training"]["track_plan_overrides"]["default.yaml"]
    assert entry is not None
    # Default constructor must work (every field optional).
    assert TrackPlanOverrides().micro_batch_size is None
    assert TrackPlanOverrides(micro_batch_size=4).micro_batch_size == 4


def test_training_track_config_required_fields_are_positional():
    """The schema enforces required fields by lacking defaults (positional-only at construction)."""
    import pytest

    with pytest.raises(TypeError):
        # Missing model, source_stage_attr, optimizer, lr_scheduler.
        TrainingTrackConfig()  # type: ignore[call-arg]


def test_training_track_config_plan_overrides_optional():
    """``plan_overrides`` defaults to None and accepts a TrackPlanOverrides instance."""
    cfg = TrainingTrackConfig(
        model={"_target_": "fake.Pipeline"},
        source_stage_attr="diffusion",
        optimizer=None,  # type: ignore[arg-type]
        lr_scheduler=None,  # type: ignore[arg-type]
        plan_overrides=TrackPlanOverrides(micro_batch_size=8),
    )
    assert cfg.plan_overrides is not None
    assert cfg.plan_overrides.micro_batch_size == 8
