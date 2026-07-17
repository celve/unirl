from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

import unirl.train.stack.sao as sao_stack
from unirl.algorithms import AlgorithmStepResult
from unirl.algorithms.gae import compute_action_token_gae
from unirl.train.stack.sao import SAOTrainStack, plan_trajectory_micros
from unirl.types.conditions import TextTokenCondition
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams
from unirl.types.segments import TextSegment


def _condition(start: int, length: int = 2) -> dict[str, Any]:
    ids = torch.arange(start, start + length).unsqueeze(0)
    return {
        "prompt": TextTokenCondition(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
        )
    }


def _gen_part(sample_id: str, tokens: list[int], *, version: int) -> Part:
    n = len(tokens)
    return Part(
        sample_ids=[sample_id],
        segment=TextSegment.pack(
            tokens=[torch.tensor(tokens)],
            log_probs=[torch.zeros(n)],
            loss_mask=[torch.ones(n)],
        ),
        conditions=_condition(tokens[0] + 20),
        sampling_params=ARSamplingParams(samples_per_prompt=1),
        weight_version=version,
    )


def _two_turn_trajectory() -> Sample:
    return Sample(
        parts=[
            Part.input(["root"]),
            _gen_part("root/0", [1, 2], version=3),
            Part(sample_ids=["root/0/0"], role="tool"),
            _gen_part("root/0/0/0", [3], version=4),
        ]
    )


class _Backend:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events
        self.model = nn.Linear(1, 1, bias=False)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.steps = 0
        self.rollout_ends = 0

    def trainable_module(self) -> nn.Module:
        return self.model

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def set_grad_sync(self, enabled: bool) -> None:
        self.events.append(f"{self.name}:sync={enabled}")

    def optimizer_step(self, *, max_grad_norm: float) -> float:
        del max_grad_norm
        self.optimizer.step()
        self.steps += 1
        self.events.append(f"{self.name}:step")
        return 0.5

    def on_rollout_end(self) -> None:
        self.rollout_ends += 1
        self.events.append(f"{self.name}:end")


class _CriticAlgorithm:
    def __init__(self, backend: _Backend, events: list[str]) -> None:
        self.backend = backend
        self.events = events
        self.targets: list[torch.Tensor] = []
        self.loss_scales: list[float] = []
        self.train_calls = 0

    def compute_loss_and_backward(
        self,
        *,
        conditions: Any,
        segment: TextSegment,
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        del conditions, advantages, training_progress
        assert segment.value_targets is not None
        assert segment.value_mask is not None
        self.targets.append(segment.value_targets.detach().clone())
        self.loss_scales.append(float(loss_scale))
        self.train_calls += 1
        self.events.append("critic:loss")
        (self.backend.model.weight.sum() * float(loss_scale)).backward()
        return AlgorithmStepResult(
            loss=1.0,
            metrics={"value_mse": 1.0},
            num_steps_or_tokens=int(segment.value_targets.numel()),
            has_backward=True,
        )

    def predict_values(self, conditions: Any, segment: TextSegment) -> torch.Tensor:
        del conditions
        assert self.train_calls == 2
        assert segment.tokens is not None
        self.events.append("critic:predict")
        return torch.zeros_like(segment.tokens, dtype=torch.float32)


class _ActorAlgorithm:
    def __init__(self, backend: _Backend, events: list[str]) -> None:
        self.backend = backend
        self.events = events
        self.advantages: list[torch.Tensor] = []
        self.loss_scales: list[float] = []

    def compute_loss_and_backward(
        self,
        *,
        conditions: Any,
        segment: TextSegment,
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        del conditions, advantages, training_progress
        assert segment.token_advantages is not None
        self.advantages.append(segment.token_advantages.detach().clone())
        self.loss_scales.append(float(loss_scale))
        self.events.append("actor:loss")
        (self.backend.model.weight.sum() * float(loss_scale)).backward()
        return AlgorithmStepResult(
            loss=2.0,
            metrics={"dis_accept_fraction": 1.0},
            num_steps_or_tokens=int(segment.token_advantages.numel()),
            has_backward=True,
        )


def test_composite_stack_runs_critic_twice_then_fresh_gae_then_actor() -> None:
    events: list[str] = []
    actor_backend = _Backend("actor", events)
    critic_backend = _Backend("critic", events)
    critic = _CriticAlgorithm(critic_backend, events)
    actor = _ActorAlgorithm(actor_backend, events)
    stack = SAOTrainStack(
        actor_backend=actor_backend,
        actor_algorithm=actor,
        critic_backend=critic_backend,
        critic_algorithm=critic,
        actor_micro_batch_size=1,
        critic_micro_batch_size=1,
        critic_updates_per_actor=2,
        gamma=0.5,
        gae_alpha=1.5,
        critic_lambda=1.0,
    )

    result = stack.train_trajectories(
        [_two_turn_trajectory()],
        torch.tensor([1.0]),
        training_progress=0.25,
    )

    assert critic_backend.steps == 2
    assert actor_backend.steps == 1
    assert events.index("critic:predict") > events.index("critic:step")
    assert events.index("actor:loss") > events.index("critic:predict")
    assert critic_backend.rollout_ends == actor_backend.rollout_ends == 1

    # Observations never enter either packed train signal.  Critic lambda=1
    # gives Monte-Carlo returns over exactly the three generated tokens.
    expected_targets = torch.tensor([0.25, 0.5, 1.0])
    assert len(critic.targets) == 2
    assert all(torch.allclose(target, expected_targets) for target in critic.targets)

    expected_advantages = compute_action_token_gae(
        values=torch.zeros(3),
        action_lengths=[2, 1],
        terminal_reward=1.0,
        gamma=0.5,
        gae_lambda=1.0 - 1.0 / (1.5 * 3),
    ).advantages
    assert len(actor.advantages) == 1
    assert torch.allclose(actor.advantages[0], expected_advantages)
    assert result.metrics["critic/updates_per_actor"] == 2
    assert result.metrics["actor/tokens"] == 3
    assert result.metrics["critic/tokens"] == 3


def test_trajectory_micro_planner_keeps_trajectories_whole() -> None:
    groups = plan_trajectory_micros(
        [90, 50, 40, 10],
        micro_batch_size=2,
        token_budget=100,
    )
    assert sorted(i for group in groups for i in group) == [0, 1, 2, 3]
    assert all(len(group) <= 2 for group in groups)
    assert all(sum([90, 50, 40, 10][i] for i in group) <= 100 for group in groups)

    # A DP peer may require more collectives. Rebalancing changes only bins,
    # never splits or duplicates a trajectory.
    rebalanced = plan_trajectory_micros(
        [90, 50, 40, 10],
        micro_batch_size=4,
        target_micros=3,
    )
    assert len(rebalanced) == 3
    assert sorted(i for group in rebalanced for i in group) == [0, 1, 2, 3]


def test_trajectory_micro_planner_rejects_singleton_over_token_budget() -> None:
    with pytest.raises(
        ValueError,
        match=r"whole trajectory at index 1 has replay cost 101.*token_budget=100.*indivisible",
    ):
        plan_trajectory_micros(
            [40, 101, 20],
            micro_batch_size=3,
            token_budget=100,
        )


def test_composite_stack_reports_global_dp_statistics_without_changing_loss_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate an unequal two-rank shard without starting a process group."""

    def fake_global_int(value: int, *, op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.SUM) -> int:
        if op == torch.distributed.ReduceOp.MAX:
            return int(value)
        assert value == 3
        return 5  # local rank has 3 tokens; its peer has 2

    peer_lambda = 1.0 - 1.0 / (1.5 * 2)

    def fake_all_reduce_named(
        values: dict[str, float],
        *,
        op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.SUM,
    ) -> dict[str, float]:
        del op
        reduced = dict(values)
        if "actor_tokens" in values:
            reduced.update(
                actor_tokens=values["actor_tokens"] + 2.0,
                critic_tokens=values["critic_tokens"] + 2.0,
                gae_lambda_count=values["gae_lambda_count"] + 1.0,
                gae_lambda_sum=values["gae_lambda_sum"] + peer_lambda,
                replay_tokens=values["replay_tokens"] + 5.0,
            )
        elif "dis_accepted" in values:
            reduced["loss_sum"] += 8.0  # peer actor mean 4.0 over 2 tokens
            reduced["ratio_count"] += 2.0
            # The peer rejects both tokens, so dis_accepted stays unchanged.
        elif "value_finite" in values:
            reduced["loss_sum"] += 6.0  # peer critic mean 3.0 over 2 tokens
            reduced["value_finite"] += 2.0
        return reduced

    monkeypatch.setattr(sao_stack, "_global_int", fake_global_int)
    monkeypatch.setattr(sao_stack, "_dp_world_size", lambda: 2)
    monkeypatch.setattr(sao_stack, "_all_reduce_named", fake_all_reduce_named)

    events: list[str] = []
    actor_backend = _Backend("actor", events)
    critic_backend = _Backend("critic", events)
    critic = _CriticAlgorithm(critic_backend, events)
    actor = _ActorAlgorithm(actor_backend, events)
    stack = SAOTrainStack(
        actor_backend=actor_backend,
        actor_algorithm=actor,
        critic_backend=critic_backend,
        critic_algorithm=critic,
        actor_micro_batch_size=1,
        critic_micro_batch_size=1,
        critic_updates_per_actor=2,
        gamma=0.5,
        gae_alpha=1.5,
        critic_lambda=1.0,
    )

    result = stack.train_trajectories(
        [_two_turn_trajectory()],
        torch.tensor([1.0]),
        training_progress=0.25,
    )

    # The existing gradient normalization remains local_tokens * dp/global.
    assert actor.loss_scales == pytest.approx([3 * 2 / 5])
    assert critic.loss_scales == pytest.approx([3 * 2 / 5, 3 * 2 / 5])

    # Returned scalars represent the complete DP batch, not rank zero's shard.
    assert result.loss == pytest.approx((2.0 * 3 + 4.0 * 2) / 5)
    assert result.metrics["dis_accept_fraction"] == pytest.approx(3 / 5)
    assert result.metrics["critic/loss"] == pytest.approx((1.0 * 3 + 3.0 * 2) / 5)
    assert result.metrics["critic/value_mse"] == pytest.approx((1.0 * 3 + 3.0 * 2) / 5)
    assert result.metrics["actor/tokens"] == 5
    assert result.metrics["critic/tokens"] == 5
    assert result.metrics["gae/lambda_mean"] == pytest.approx(((1.0 - 1.0 / (1.5 * 3)) + peer_lambda) / 2)
    assert result.metrics["replay/token_amplification"] == pytest.approx((7 + 5) / 5)
