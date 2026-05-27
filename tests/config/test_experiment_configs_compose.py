"""Compose-time regression test for every experiment recipe.

The config-preset refactor moved ~30 recipes onto shared Hydra group presets.
This test is the gate that keeps them loadable: every maintained recipe must
**compose and fully resolve** under Hydra struct mode. It catches the failure
classes the refactor addressed so they cannot silently regress:

* phantom ``sampling.sde_config`` / ``algorithm.eta`` keys (rejected by the
  closed-struct ``DiffusionSamplingParams`` / ``GRPORolloutControlConfig``),
* defaults-list ordering (Hydra requires ``override`` entries at the end),
* ``reward.components`` as a list instead of a dict.

Only ``DATA_PATH`` is referenced without a default across all recipes; the
fixture supplies it. Recipes with pre-existing, unrelated schema issues are
skipped explicitly (see ``_KNOWN_BROKEN``).
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from diffusionrl.config import register_all_configs
from diffusionrl.config.polymorphic import expand_polymorphic_fields
from diffusionrl.training.validate import _validate_cfg_for_train
from diffusionrl.types.sampling import get_diffusion_params

_CONF_DIR = str(Path(__file__).resolve().parents[2] / "conf")
_EXPERIMENTS = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(_CONF_DIR, "experiment", "*.yaml")))

# Recipes failing on pre-existing schema issues unrelated to the preset refactor
# (features not ported to the multi-track pipeline), tracked separately:
#   mixgrpo_sd35      -> algorithm.skip_last_timestep (MixGRPO window scheduler)
#   smoke_composed_pe -> rollout.engine.stage_id_map (composed PE engine)
_KNOWN_BROKEN = {"mixgrpo_sd35", "smoke_composed_pe"}

_REPRESENTATIVE = ["flowgrpo_fast_sd3_colocate", "nft_sd3", "hi3_think_recaption_colocate"]
_ADV_GROUP_SIZE_CASES = {
    "flowgrpo_fast_sd3": 8,
    "dancegrpo_fast_sd3": 16,
    "nft_sd3": 16,
    "grpo_flux2_klein9b_trainside_smoke": 2,
}
# Keep this list to recipes whose train-time model ownership is represented
# by today's per-track stack. HI3 dual-loss training needs a shared-backbone
# track design and should not be treated as train-entry green by shape alone.
_TRAIN_ENTRY_CASES = ["flowgrpo_fast_sd3_colocate_reproduce"]


@pytest.fixture(scope="module", autouse=True)
def _registered_configs():
    """Populate the ConfigStore with every ``@register_config`` group once."""
    register_all_configs()
    yield


@pytest.fixture(autouse=True)
def _required_env(monkeypatch):
    """``DATA_PATH`` is the only env var referenced without a default."""
    monkeypatch.setenv("DATA_PATH", "/tmp/diffusionrl-test/train.txt")


def _compose(experiment: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=_CONF_DIR):
        return compose(config_name="train", overrides=[f"+experiment={experiment}"])


@pytest.mark.parametrize("experiment", _EXPERIMENTS)
def test_experiment_composes_and_resolves(experiment: str):
    """Every maintained recipe composes and fully resolves under struct mode."""
    if experiment in _KNOWN_BROKEN:
        pytest.skip(f"{experiment}: pre-existing non-preset schema issue (tracked separately)")
    cfg = _compose(experiment)
    OmegaConf.to_container(cfg, resolve=True)  # force interpolation resolution


@pytest.mark.parametrize("experiment", _REPRESENTATIVE)
def test_schema_contracts(experiment: str):
    """Spot-check the schema contracts the refactor enforces (V1 + V3)."""
    cfg = _compose(experiment)
    sampling = cfg.sampling.diffusion if "diffusion" in cfg.sampling else cfg.sampling
    # V1: the flattened SDE schema — no nested sde_config, no dead algorithm.eta.
    assert sampling.get("sde_config") is None
    assert cfg.algorithm.get("sde_config") is None
    assert cfg.algorithm.get("eta") is None
    assert sampling.get("eta") is not None
    # V3: reward.components is a mapping, not a list.
    components = cfg.reward.get("components")
    assert components is not None and not isinstance(OmegaConf.to_container(components), list)


@pytest.mark.parametrize(("experiment", "expected"), _ADV_GROUP_SIZE_CASES.items())
def test_actor_advantage_group_size_uses_sampling_contract(experiment: str, expected: int):
    """Actor-side grouped advantage normalization follows sampling, not the old algorithm alias."""
    cfg = _compose(experiment)
    sampling = get_diffusion_params(cfg.sampling)
    assert int(sampling.samples_per_prompt) == expected


@pytest.mark.parametrize("experiment", _TRAIN_ENTRY_CASES)
def test_train_entry_preprocess_contracts(experiment: str):
    """Representative trainable recipes survive train.py's config preprocessing."""
    cfg = _compose(experiment)
    OmegaConf.resolve(cfg)
    expand_polymorphic_fields(cfg)
    _validate_cfg_for_train(cfg)
    assert set(cfg.training.tracks.keys()) == set(cfg.algorithm.algorithms.keys())
