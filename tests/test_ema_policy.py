"""Unit tests for :class:`diffusionrl.training_new.ema_policy.EMAPolicy`.

Builds a fake Stage-shaped source so we can exercise the EMA shadow
init / step / swap / state-dict surface without needing a real model
or a torch.distributed process group.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from diffusionrl.training_new.ema_policy import EMAPolicy, EMAPolicyConfig
from diffusionrl.utils.ema import EMAModuleWrapper

# ---------------------------------------------------------------------------
# Test scaffolding: a fake Stage-shaped source
# ---------------------------------------------------------------------------


class _FakeStage(nn.Module):
    """Stage-shaped: nn.Module + ``trainable_module()`` + counter for
    ``post_materialize_init`` so tests can verify the inward chain ran.
    """

    def __init__(self, in_dim: int = 4, out_dim: int = 4) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self._post_materialize_init_calls = 0

    def trainable_module(self) -> nn.Module:
        return self

    def post_materialize_init(self) -> None:
        self._post_materialize_init_calls += 1


def _make_policy(decay: float = 0.5, **kwargs) -> EMAPolicy:
    stage = _FakeStage()
    # Initialize params to known values (zeros) so EMA math is easy to verify.
    with torch.no_grad():
        stage.lin.weight.zero_()
    cfg = EMAPolicyConfig(decay=decay, **kwargs)
    return EMAPolicy(cfg, stage)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_init_raises_when_source_lacks_trainable_module():
    class _Bad:
        pass

    with pytest.raises(TypeError, match=r"trainable_module"):
        EMAPolicy(EMAPolicyConfig(), _Bad())


def test_init_does_not_build_shadow():
    """Shadow is built lazily in post_materialize_init — at __init__
    time, source params may still be on meta device, so cloning would
    produce a junk shadow.
    """

    policy = _make_policy()
    assert policy.ema is None


def test_step_raises_before_post_materialize_init():
    policy = _make_policy()
    with pytest.raises(RuntimeError, match=r"shadow not initialized"):
        policy.step(optimization_step=1)


def test_use_ema_parameters_raises_before_post_materialize_init():
    policy = _make_policy()
    with pytest.raises(RuntimeError, match=r"shadow not initialized"):
        with policy.use_ema_parameters():
            pass  # pragma: no cover — never reached


def test_load_ema_state_dict_raises_before_post_materialize_init():
    policy = _make_policy()
    with pytest.raises(RuntimeError, match=r"shadow not initialized"):
        policy.load_ema_state_dict({})


def test_ema_state_dict_is_empty_before_post_materialize_init():
    policy = _make_policy()
    assert policy.ema_state_dict() == {}


# ---------------------------------------------------------------------------
# post_materialize_init
# ---------------------------------------------------------------------------


def test_post_materialize_init_builds_shadow_with_correct_param_count():
    policy = _make_policy()
    policy.post_materialize_init()

    assert isinstance(policy.ema, EMAModuleWrapper)
    # Default _FakeStage has all params with requires_grad=True, so
    # the trainable count matches the total.
    n_params = sum(1 for _ in policy.source.parameters())
    n_trainable = sum(1 for p in policy.source.parameters() if p.requires_grad)
    assert n_params == n_trainable
    assert len(policy.ema.ema_parameters) == n_trainable


def test_post_materialize_init_skips_frozen_params():
    """When some params have ``requires_grad=False`` (e.g. peft has
    frozen the base under a LoRA stack), the shadow must skip them. A
    full snapshot would blow up GPU memory by ~size-of-base on
    HI3-scale models.
    """

    stage = _FakeStage()
    extra = nn.Linear(8, 8, bias=True)
    extra.weight.requires_grad_(False)
    extra.bias.requires_grad_(False)
    stage.frozen = extra  # adds 2 frozen params

    cfg = EMAPolicyConfig()
    policy = EMAPolicy(cfg, stage)
    policy.post_materialize_init()

    n_total = sum(1 for _ in policy.source.parameters())
    n_trainable = sum(1 for p in policy.source.parameters() if p.requires_grad)
    assert n_total == n_trainable + 2  # the two frozen params we just added
    assert len(policy.ema.ema_parameters) == n_trainable


def test_post_materialize_init_chains_to_source():
    policy = _make_policy()
    policy.post_materialize_init()
    # PolicyBase.post_materialize_init walks inward via getattr → calls
    # source.post_materialize_init() exactly once.
    assert policy.source._post_materialize_init_calls == 1


def test_post_materialize_init_snapshots_current_params():
    """The shadow's tensors should equal the source's params at the
    moment ``post_materialize_init`` is called, not whatever they later
    drift to."""

    policy = _make_policy()
    # Set known param values BEFORE post_materialize_init so the
    # snapshot captures them.
    with torch.no_grad():
        policy.source.lin.weight.fill_(3.0)
    policy.post_materialize_init()

    assert policy.ema is not None
    snap = policy.ema.ema_parameters[0]
    assert torch.allclose(snap, torch.full_like(snap, 3.0))

    # Now mutate source — shadow should still hold the snapshot.
    with torch.no_grad():
        policy.source.lin.weight.fill_(7.0)
    snap_after = policy.ema.ema_parameters[0]
    assert torch.allclose(snap_after, torch.full_like(snap_after, 3.0))


# ---------------------------------------------------------------------------
# step
# ---------------------------------------------------------------------------


def test_step_updates_shadow_toward_new_params():
    """After step(), shadow moves from snapshot toward the current
    params per the warmup decay schedule.
    """

    policy = _make_policy(decay=0.99)  # high target; warmup dominates early
    with torch.no_grad():
        policy.source.lin.weight.fill_(0.0)
    policy.post_materialize_init()  # shadow = 0.0

    # Mutate params to 1.0 to simulate an optimizer step.
    with torch.no_grad():
        policy.source.lin.weight.fill_(1.0)

    # At optimization_step=1 the warmup formula gives
    # decay = min((1+1)/(10+1), 0.99) = 2/11 ≈ 0.1818.
    # shadow_new = 0.1818 * 0 + 0.8182 * 1 = 0.8182.
    policy.step(optimization_step=1)

    snap = policy.ema.ema_parameters[0]
    expected = 1.0 - 2.0 / 11.0  # 1 - decay = 1 - 2/11
    assert torch.allclose(snap, torch.full_like(snap, expected), atol=1e-5)


def test_step_with_default_optimization_step_uses_internal_counter():
    policy = _make_policy()
    with torch.no_grad():
        policy.source.lin.weight.fill_(0.0)
    policy.post_materialize_init()

    with torch.no_grad():
        policy.source.lin.weight.fill_(1.0)

    # No explicit step number; wrapper increments _step_counter on each call.
    initial_counter = policy.ema._step_counter
    policy.step()
    assert policy.ema._step_counter == initial_counter + 1


# ---------------------------------------------------------------------------
# use_ema_parameters
# ---------------------------------------------------------------------------


def test_use_ema_parameters_swaps_and_restores():
    policy = _make_policy()
    with torch.no_grad():
        policy.source.lin.weight.fill_(2.0)  # snapshot value
    policy.post_materialize_init()

    with torch.no_grad():
        policy.source.lin.weight.fill_(7.0)  # current value (post-training)

    # Inside the context: source params should match the snapshot (2.0).
    with policy.use_ema_parameters():
        assert torch.allclose(
            policy.source.lin.weight,
            torch.full_like(policy.source.lin.weight, 2.0),
        )

    # After exit: source params are restored to current value (7.0).
    assert torch.allclose(
        policy.source.lin.weight,
        torch.full_like(policy.source.lin.weight, 7.0),
    )


def test_use_ema_parameters_restores_on_exception():
    policy = _make_policy()
    with torch.no_grad():
        policy.source.lin.weight.fill_(5.0)
    policy.post_materialize_init()

    with torch.no_grad():
        policy.source.lin.weight.fill_(11.0)

    with pytest.raises(RuntimeError, match=r"deliberate"):
        with policy.use_ema_parameters():
            raise RuntimeError("deliberate")

    # Even on exception, the original params must be restored.
    assert torch.allclose(
        policy.source.lin.weight,
        torch.full_like(policy.source.lin.weight, 11.0),
    )


# ---------------------------------------------------------------------------
# state_dict round-trip
# ---------------------------------------------------------------------------


def test_ema_state_dict_round_trip_preserves_shadow_tensors():
    policy = _make_policy(decay=0.9)
    with torch.no_grad():
        policy.source.lin.weight.fill_(4.0)
    policy.post_materialize_init()

    saved = policy.ema_state_dict()
    assert "ema_parameters" in saved
    assert "decay" in saved
    assert "step_counter" in saved

    # ``EMAModuleWrapper.state_dict`` returns the live ``ema_parameters``
    # list by reference (no clone). In production this is fine because
    # ``torch.save`` serializes through and breaks the reference; for the
    # in-memory round-trip here we have to clone manually so subsequent
    # in-place ``step()`` updates don't also mutate the snapshot.
    snapshot = {
        "decay": saved["decay"],
        "ema_parameters": [t.clone() for t in saved["ema_parameters"]],
        "step_counter": saved["step_counter"],
    }

    # Now mutate the shadow indirectly by stepping.
    with torch.no_grad():
        policy.source.lin.weight.fill_(0.0)
    policy.step(optimization_step=1)
    drifted = policy.ema.ema_parameters[0].clone()
    assert not torch.allclose(drifted, torch.full_like(drifted, 4.0))

    # Reload — shadow goes back to the saved 4.0.
    policy.load_ema_state_dict(snapshot)
    restored = policy.ema.ema_parameters[0]
    assert torch.allclose(restored, torch.full_like(restored, 4.0))


# ---------------------------------------------------------------------------
# Stack composition (smoke check that EMAPolicy behaves as a Policy)
# ---------------------------------------------------------------------------


def test_ema_policy_satisfies_policy_protocol():
    from diffusionrl.training_new.policy import Policy

    policy = _make_policy()
    assert isinstance(policy, Policy)


def test_trainable_module_returns_same_object_as_source():
    policy = _make_policy()
    assert policy.trainable_module() is policy.source.trainable_module()
    assert policy.model is policy.source
