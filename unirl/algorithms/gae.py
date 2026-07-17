"""Pure action-token GAE utilities for multi-turn agent trajectories.

Callers concatenate generated action spans only and pass their lengths.  The
resulting flat chain links the last token of one action directly to the first
token of the next action, so observation tokens add neither value/loss entries
nor extra discount steps while still having conditioned the next action's
critic forward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Union

import torch

Scalar = Union[float, int, torch.Tensor]


def _normalize_lengths(action_lengths: Sequence[int] | torch.Tensor) -> tuple[int, ...]:
    if isinstance(action_lengths, torch.Tensor):
        if action_lengths.ndim != 1:
            raise ValueError(f"action_lengths tensor must be one-dimensional; got shape={tuple(action_lengths.shape)}")
        raw = action_lengths.tolist()
    else:
        raw = action_lengths
    lengths_list: list[int] = []
    for length in raw:
        try:
            numeric = float(length)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"action lengths must be integers; got {length!r}") from exc
        if not math.isfinite(numeric) or numeric != int(numeric):
            raise ValueError(f"action lengths must be integers; got {length!r}")
        lengths_list.append(int(numeric))
    lengths = tuple(lengths_list)
    if any(length < 0 for length in lengths):
        raise ValueError(f"action_lengths must be non-negative; got {lengths!r}")
    return lengths


def split_action_tokens(tensor: torch.Tensor, action_lengths: Sequence[int] | torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Split one packed action-only tensor back into its turn spans."""

    lengths = _normalize_lengths(action_lengths)
    if sum(lengths) != int(tensor.shape[0]):
        raise ValueError(f"action_lengths sum to {sum(lengths)}, but tensor has {int(tensor.shape[0])} tokens")
    return tuple(torch.split(tensor, lengths, dim=0))


def concatenate_action_tokens(action_tensors: Sequence[torch.Tensor]) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Concatenate per-turn action tensors and return their lossless split sizes."""

    if not action_tensors:
        raise ValueError("concatenate_action_tokens requires at least one action tensor")
    reference = action_tensors[0]
    if reference.ndim < 1:
        raise ValueError("action tensors must have a token dimension")
    trailing_shape = tuple(reference.shape[1:])
    for index, tensor in enumerate(action_tensors):
        if tensor.ndim < 1 or tuple(tensor.shape[1:]) != trailing_shape:
            raise ValueError(
                f"action tensor {index} shape={tuple(tensor.shape)} is incompatible with "
                f"trailing shape={trailing_shape}"
            )
        if tensor.dtype != reference.dtype or tensor.device != reference.device:
            raise ValueError("all action tensors must share dtype and device")
    lengths = tuple(int(tensor.shape[0]) for tensor in action_tensors)
    return torch.cat(tuple(action_tensors), dim=0), lengths


def adaptive_policy_lambda(total_action_tokens: int, *, alpha: float = 1.5) -> float:
    """SAO's trajectory-length-adaptive policy GAE lambda."""

    length = int(total_action_tokens)
    resolved_alpha = float(alpha)
    if length < 1:
        raise ValueError(f"total_action_tokens must be >= 1; got {total_action_tokens!r}")
    if not (resolved_alpha > 0.0 and math.isfinite(resolved_alpha)):
        raise ValueError(f"alpha must be finite and > 0; got {alpha!r}")
    value = 1.0 - 1.0 / (resolved_alpha * length)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"adaptive lambda={value} is outside [0, 1]; require alpha * total_action_tokens >= 1")
    return value


def _scalar_like(value: Scalar, reference: torch.Tensor, *, name: str) -> torch.Tensor:
    scalar = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if scalar.numel() != 1:
        raise ValueError(f"{name} must be scalar; got shape={tuple(scalar.shape)}")
    scalar = scalar.reshape(()).detach()
    if not bool(torch.isfinite(scalar).item()):
        raise ValueError(f"{name} must be finite; got {scalar.item()!r}")
    return scalar


def _discounted_reverse_sum(deltas: torch.Tensor, discount: float) -> torch.Tensor:
    """Stable reverse recurrence ``out[t] = x[t] + discount*out[t+1]``.

    The scan is vectorized in bounded chunks.  This avoids one Python/GPU
    launch per token for 128k trajectories while also avoiding powers small
    enough to underflow in the weighted-cumsum identity.
    """

    if deltas.numel() == 0:
        return deltas.clone()
    if discount == 0.0:
        return deltas.clone()
    reversed_deltas = deltas.flip(0)
    if discount == 1.0:
        return reversed_deltas.cumsum(dim=0).flip(0)

    # Keep the smallest divisor in a chunk above ~1e-12.  At the paper's
    # adaptive lambda this chooses 1024 tokens; small discounts use shorter
    # chunks rather than amplifying round-off through division by tiny powers.
    safe_by_discount = max(1, int(math.floor(math.log(1.0e-12) / math.log(discount))))
    chunk_size = min(1024, safe_by_discount)
    chunks: list[torch.Tensor] = []
    carry = reversed_deltas.new_zeros(())
    for start in range(0, int(reversed_deltas.shape[0]), chunk_size):
        chunk = reversed_deltas[start : start + chunk_size]
        exponents = torch.arange(chunk.shape[0], device=chunk.device, dtype=chunk.dtype)
        powers = torch.pow(chunk.new_tensor(discount), exponents)
        scanned = powers * torch.cumsum(chunk / powers, dim=0)
        scanned = scanned + (powers * discount) * carry
        carry = scanned[-1]
        chunks.append(scanned)
    return torch.cat(chunks, dim=0).flip(0)


@dataclass(frozen=True)
class ActionTokenGAE:
    """Flat action-only GAE and lambda-return targets."""

    advantages: torch.Tensor
    value_targets: torch.Tensor
    action_lengths: tuple[int, ...]

    def split_advantages(self) -> tuple[torch.Tensor, ...]:
        return split_action_tokens(self.advantages, self.action_lengths)

    def split_value_targets(self) -> tuple[torch.Tensor, ...]:
        return split_action_tokens(self.value_targets, self.action_lengths)


def compute_action_token_gae(
    *,
    values: torch.Tensor,
    action_lengths: Sequence[int] | torch.Tensor,
    terminal_reward: Scalar,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    terminal: bool = True,
    bootstrap_value: Scalar | None = None,
) -> ActionTokenGAE:
    """Compute GAE over a concatenated action-only token chain.

    ``terminal_reward`` is placed on the final action token; all earlier token
    rewards are zero.  For a terminal trajectory the final next-state value is
    zero.  A non-terminal/truncated trajectory must supply ``bootstrap_value``.
    Values and scalar inputs are detached, so returned advantages/targets are
    fixed learner signals rather than paths into the critic graph.

    With ``gae_lambda=1`` the returned ``value_targets`` are discounted Monte
    Carlo terminal returns, independent of the supplied intermediate values.
    """

    if values.ndim != 1:
        raise ValueError(f"values must be a flat [action_tokens] tensor; got shape={tuple(values.shape)}")
    if not torch.is_floating_point(values):
        raise TypeError(f"values must be floating point; got {values.dtype}")
    lengths = _normalize_lengths(action_lengths)
    total = sum(lengths)
    if total < 1:
        raise ValueError("GAE requires at least one generated action token")
    if total != int(values.shape[0]):
        raise ValueError(f"action_lengths sum to {total}, but values has {int(values.shape[0])} tokens")

    resolved_gamma = float(gamma)
    resolved_lambda = float(gae_lambda)
    if not (math.isfinite(resolved_gamma) and 0.0 <= resolved_gamma <= 1.0):
        raise ValueError(f"gamma must be finite and in [0, 1]; got {gamma!r}")
    if not (math.isfinite(resolved_lambda) and 0.0 <= resolved_lambda <= 1.0):
        raise ValueError(f"gae_lambda must be finite and in [0, 1]; got {gae_lambda!r}")

    work_dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
    detached_values = values.detach().to(dtype=work_dtype)
    if not bool(torch.isfinite(detached_values).all().item()):
        raise ValueError("values must all be finite")
    reward = _scalar_like(terminal_reward, detached_values, name="terminal_reward")
    if terminal:
        if bootstrap_value is not None:
            raise ValueError("bootstrap_value must be None for a terminal trajectory")
        bootstrap = detached_values.new_zeros(())
    else:
        if bootstrap_value is None:
            raise ValueError("bootstrap_value is required for a non-terminal trajectory")
        bootstrap = _scalar_like(bootstrap_value, detached_values, name="bootstrap_value")

    rewards = torch.zeros_like(detached_values)
    rewards[-1] = reward
    next_values = torch.empty_like(detached_values)
    next_values[:-1] = detached_values[1:]
    next_values[-1] = bootstrap
    deltas = rewards + resolved_gamma * next_values - detached_values
    advantages = _discounted_reverse_sum(deltas, resolved_gamma * resolved_lambda)
    value_targets = advantages + detached_values
    return ActionTokenGAE(
        advantages=advantages,
        value_targets=value_targets,
        action_lengths=lengths,
    )


# Aliases make call sites self-documenting while keeping one implementation.
action_token_gae = compute_action_token_gae
compute_skip_observation_gae = compute_action_token_gae


__all__ = [
    "ActionTokenGAE",
    "action_token_gae",
    "adaptive_policy_lambda",
    "compute_action_token_gae",
    "compute_skip_observation_gae",
    "concatenate_action_tokens",
    "split_action_tokens",
]
