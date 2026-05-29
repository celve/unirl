"""Structural + sampling validation for ``conf_v2/pe_trainside.yaml``.

CPU-only: loads the PE trainside recipe, checks the blocks ``PETrainer``
consumes resolve to the intended targets, and instantiates the
``ComposedSamplingParams`` to verify the N×M fan-out factor. Does NOT
instantiate models / open a pool (no GPU, no Ray) — the end-to-end run is a
GPU smoke.
"""

from __future__ import annotations

from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf

from diffusionrl.types.sampling import (
    ARSamplingParams,
    ComposedSamplingParams,
    DiffusionSamplingParams,
)

_RECIPE = Path(__file__).resolve().parents[1] / "conf_v2" / "pe_trainside.yaml"


def _load():
    return OmegaConf.load(_RECIPE)


def test_recipe_has_all_petrainer_blocks() -> None:
    cfg = _load()
    # Every kwarg PETrainer.__init__ consumes is present at the top level.
    for key in ("num_devices", "batch_size", "diffusion", "ar", "rollout", "reward", "data_source", "sampling"):
        assert key in cfg, f"missing top-level block: {key}"
    # Each trained side has the five sibling-role blocks _wire_side builds.
    for side in ("diffusion", "ar"):
        for role in ("bundle", "pipeline", "backend", "algorithm", "stack"):
            assert role in cfg[side], f"{side} missing role block: {role}"


def test_recipe_targets_and_stage_wiring() -> None:
    cfg = _load()
    assert cfg.diffusion.algorithm["_target_"].endswith("DiffusionGRPO")
    assert cfg.diffusion.algorithm.stage_attr == "diffusion"
    assert cfg.ar.algorithm["_target_"].endswith("ARGRPO")
    assert cfg.ar.algorithm.stage_attr == "ar"
    # Both backends train the same attr on their respective bundles.
    assert cfg.diffusion.backend.trainable_attr == "transformer"
    assert cfg.ar.backend.trainable_attr == "transformer"
    # Composed trainside rollout eval-scopes BOTH trained stages.
    assert cfg.rollout["_target_"].endswith("TrainsideRolloutEngine")
    assert list(cfg.rollout.stage_attrs) == ["diffusion", "ar"]
    # No weight-sync block — trainside shares the live modules.
    assert "sync" not in cfg


def test_sampling_fans_out_n_times_m() -> None:
    cfg = _load()
    sampling = instantiate(cfg.sampling)
    assert isinstance(sampling, ComposedSamplingParams)
    assert isinstance(sampling.ar, ARSamplingParams)
    assert isinstance(sampling.diffusion, DiffusionSamplingParams)
    n = int(sampling.ar.samples_per_prompt)
    m = int(sampling.diffusion.samples_per_prompt)
    assert (n, m) == (2, 2)
    # ComposedSamplingParams.__post_init__: per-prompt fan-out = N * M.
    assert int(sampling.samples_per_prompt) == n * m == 4
