from __future__ import annotations

import pytest
import torch

from unirl.algorithms.gae import (
    adaptive_policy_lambda,
    compute_action_token_gae,
    concatenate_action_tokens,
    split_action_tokens,
)


def test_critic_lambda_one_produces_discounted_terminal_returns() -> None:
    output = compute_action_token_gae(
        values=torch.tensor([0.2, -1.0, 7.0]),
        action_lengths=[2, 1],
        terminal_reward=2.0,
        gamma=0.5,
        gae_lambda=1.0,
    )

    assert torch.allclose(output.value_targets, torch.tensor([0.5, 1.0, 2.0]), atol=1e-6)
    assert torch.allclose(output.advantages, output.value_targets - torch.tensor([0.2, -1.0, 7.0]))


def test_action_boundary_jumps_over_observation_without_extra_discount() -> None:
    # The middle transition is last-token(action 0) -> first-token(action 1).
    # There is no observation slot in the value chain and therefore no extra
    # gamma factor at the turn boundary.
    values = torch.zeros(3)
    output = compute_action_token_gae(
        values=values,
        action_lengths=[2, 1],
        terminal_reward=1.0,
        gamma=0.5,
        gae_lambda=1.0,
    )

    first_action, second_action = output.split_value_targets()
    assert torch.equal(first_action, torch.tensor([0.25, 0.5]))
    assert torch.equal(second_action, torch.tensor([1.0]))


def test_general_gae_matches_direct_reverse_recurrence() -> None:
    values = torch.tensor([0.5, 1.0, -0.5, 0.25])
    gamma = 0.9
    gae_lambda = 0.6
    reward = 2.0
    output = compute_action_token_gae(
        values=values,
        action_lengths=[1, 0, 3],
        terminal_reward=reward,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    rewards = torch.tensor([0.0, 0.0, 0.0, reward])
    next_values = torch.tensor([1.0, -0.5, 0.25, 0.0])
    deltas = rewards + gamma * next_values - values
    expected = torch.empty_like(values)
    carry = torch.tensor(0.0)
    for index in range(values.numel() - 1, -1, -1):
        carry = deltas[index] + gamma * gae_lambda * carry
        expected[index] = carry

    assert torch.allclose(output.advantages, expected, atol=1e-6)
    split = output.split_advantages()
    assert [part.numel() for part in split] == [1, 0, 3]


def test_nonterminal_trajectory_uses_explicit_bootstrap() -> None:
    output = compute_action_token_gae(
        values=torch.tensor([1.0, 2.0]),
        action_lengths=[2],
        terminal_reward=0.0,
        gamma=0.5,
        gae_lambda=1.0,
        terminal=False,
        bootstrap_value=4.0,
    )
    assert torch.allclose(output.value_targets, torch.tensor([1.0, 2.0]))

    with pytest.raises(ValueError, match="bootstrap_value is required"):
        compute_action_token_gae(
            values=torch.ones(1),
            action_lengths=[1],
            terminal_reward=0.0,
            terminal=False,
        )


def test_action_concat_and_split_preserve_zero_length_turns() -> None:
    actions = [torch.tensor([1.0, 2.0]), torch.empty(0), torch.tensor([3.0])]
    packed, lengths = concatenate_action_tokens(actions)
    assert lengths == (2, 0, 1)
    assert torch.equal(packed, torch.tensor([1.0, 2.0, 3.0]))
    unpacked = split_action_tokens(packed, lengths)
    assert len(unpacked) == 3
    assert all(torch.equal(actual, expected) for actual, expected in zip(unpacked, actions))


def test_adaptive_policy_lambda_uses_total_action_tokens() -> None:
    assert adaptive_policy_lambda(10, alpha=1.5) == pytest.approx(1.0 - 1.0 / 15.0)
    with pytest.raises(ValueError):
        adaptive_policy_lambda(0)
    with pytest.raises(ValueError, match="outside"):
        adaptive_policy_lambda(1, alpha=0.5)


def test_chunked_scan_matches_long_reference_recurrence() -> None:
    generator = torch.Generator().manual_seed(7)
    values = torch.randn(4097, generator=generator)
    gamma = 0.99
    gae_lambda = 0.997
    output = compute_action_token_gae(
        values=values,
        action_lengths=[1024, 2048, 1025],
        terminal_reward=1.25,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    rewards = torch.zeros_like(values)
    rewards[-1] = 1.25
    next_values = torch.cat([values[1:], values.new_zeros(1)])
    deltas = rewards + gamma * next_values - values
    reference = torch.empty_like(values)
    carry = values.new_zeros(())
    for index in range(values.numel() - 1, -1, -1):
        carry = deltas[index] + gamma * gae_lambda * carry
        reference[index] = carry
    assert torch.allclose(output.advantages, reference, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize(
    ("values", "lengths", "message"),
    [
        (torch.empty(0), [], "at least one"),
        (torch.ones(2), [1], "sum to"),
        (torch.ones(2), [1, -1, 2], "non-negative"),
    ],
)
def test_gae_rejects_invalid_action_geometry(
    values: torch.Tensor,
    lengths: list[int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_action_token_gae(
            values=values,
            action_lengths=lengths,
            terminal_reward=1.0,
        )
