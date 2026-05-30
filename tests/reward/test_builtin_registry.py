"""Built-in reward scorer / spec registry invariants.

Locks the contract that every name in ``_BUILTIN_SCORERS`` has both a
resolvable scorer class AND a matching spec class. Drift between the two
maps (a typo in one, a missing entry in the other) is the most common
bug class in this registry; this test catches it without needing any
scorer to actually load weights.
"""

from __future__ import annotations

from diffusionrl.reward.base import BaseRewardComponentSpec, RewardBackend
from diffusionrl.reward.local.registry import (
    _BUILTIN_SCORERS,
    _BUILTIN_SPECS,
    available_builtin_reward_models,
    resolve_builtin_reward_scorer_class,
    resolve_builtin_reward_spec_class,
)


def test_scorer_and_spec_maps_have_same_keys() -> None:
    assert set(_BUILTIN_SCORERS.keys()) == set(_BUILTIN_SPECS.keys()), (
        "Scorer and Spec registries drifted: "
        f"only-scorer={set(_BUILTIN_SCORERS) - set(_BUILTIN_SPECS)}, "
        f"only-spec={set(_BUILTIN_SPECS) - set(_BUILTIN_SCORERS)}"
    )


def test_every_builtin_name_resolves_to_concrete_classes() -> None:
    for name in available_builtin_reward_models():
        scorer_cls = resolve_builtin_reward_scorer_class(name)
        spec_cls = resolve_builtin_reward_spec_class(name)
        assert issubclass(scorer_cls, RewardBackend), name
        assert issubclass(spec_cls, BaseRewardComponentSpec), name


def test_videopickscore_resolves_scorer_and_spec() -> None:
    """Specific assertion for the videopickscore registry entry — guards
    against a regression where main's PR #96/#101 used ``PickScoreSpec``
    here (registry reuse) while the path uses the dedicated
    ``VideoPickScoreSpec`` (one-Spec-per-name in the Hydra
    ``reward/component`` registry)."""
    scorer_cls = resolve_builtin_reward_scorer_class("videopickscore")
    spec_cls = resolve_builtin_reward_spec_class("videopickscore")
    assert scorer_cls.__name__ == "VideoPickScoreScorer"
    assert spec_cls.__name__ == "VideoPickScoreSpec"
