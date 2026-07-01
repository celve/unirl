"""CPU tests for env-sourced agentic RL plumbing (LIN-519 ALFWorld).

Covers the two seams that make env rewards work end-to-end without a reward backend:
the engine's ``_attach_env_reward`` (puts the episode return on the last gen Part) and
``AgenticEnvTrainer._rewards_and_groups`` (reads it back + groups by root id).
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from unirl.rollout.engine.agentic.engine import AgenticRolloutEngine  # noqa: E402
from unirl.trainer.agentic_env import AgenticEnvTrainer  # noqa: E402
from unirl.types.primitives import Texts  # noqa: E402
from unirl.types.sample import Part, Sample  # noqa: E402
from unirl.types.sampling import ARSamplingParams  # noqa: E402

_SP = ARSamplingParams(samples_per_prompt=1, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=8)


def _traj(sid: str, answer: str = "a") -> Sample:
    """A minimal one-turn trajectory: input root + one filled gen Part."""
    return (
        Sample.request(Part.input([sid], primitive=Texts(texts=["q"])))
        .fork(1, sampling_params=_SP)
        .with_filled_frontier(primitive=Texts(texts=[answer]))
    )


def test_attach_env_reward_sets_last_gen_part():
    s = _traj("r0")
    out = AgenticRolloutEngine._attach_env_reward(s, 1.0)
    rewards = out.gen_parts()[-1].rewards
    assert rewards is not None
    assert float(rewards[0].item()) == 1.0


def test_attach_env_reward_none_is_noop():
    s = _traj("r0")
    out = AgenticRolloutEngine._attach_env_reward(s, None)
    assert out.gen_parts()[-1].rewards is None


def test_attach_env_reward_no_gen_parts_is_noop():
    # input-only sample (reset failed before any generation) — must not raise
    s = Sample.request(Part.input(["r0"], primitive=Texts(texts=["q"])))
    out = AgenticRolloutEngine._attach_env_reward(s, 1.0)
    assert out.parts == s.parts


def test_rewards_and_groups_reads_env_reward_and_groups_by_root():
    trajs = [
        AgenticRolloutEngine._attach_env_reward(_traj("g0"), 1.0),
        AgenticRolloutEngine._attach_env_reward(_traj("g0"), 0.0),
        AgenticRolloutEngine._attach_env_reward(_traj("g1"), 0.5),
    ]
    # unbound call: the method reads only trajs (self unused for logging path here)
    rewards, group_ids = AgenticEnvTrainer._rewards_and_groups(object(), None, trajs, 0)
    assert [round(float(x), 3) for x in rewards.tolist()] == [1.0, 0.0, 0.5]
    assert group_ids == ["g0", "g0", "g1"]


def test_rewards_and_groups_missing_reward_scores_zero():
    trajs = [_traj("g0")]  # no env reward attached
    rewards, group_ids = AgenticEnvTrainer._rewards_and_groups(object(), None, trajs, 0)
    assert float(rewards[0].item()) == 0.0
    assert group_ids == ["g0"]
