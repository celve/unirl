"""Built-in reward scorer / spec registry invariants."""

from diffusionrl.reward.scorers.registry import resolve_builtin_reward_scorer_class, resolve_builtin_reward_spec_class


def test_videopickscore_resolves_scorer_and_spec():
    scorer_cls = resolve_builtin_reward_scorer_class("videopickscore")
    spec_cls = resolve_builtin_reward_spec_class("videopickscore")
    assert scorer_cls.__name__ == "VideoPickScoreScorer"
    assert spec_cls.__name__ == "PickScoreSpec"
