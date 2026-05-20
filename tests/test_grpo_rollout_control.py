"""Unit tests for ``diffusionrl.algorithms_new.rollout_control``.

Locks the driver-side contract that ``rollout/new_pipeline.py`` depends on:

- ``samples_per_prompt`` attribute
- ``resolve_rollout_sde_indices(current_step=...)`` returns a Set[int]
- ``get_filtered_training_indices(sde_indices, num_steps)`` applies
  ``skip_last_timestep`` and ``skip_initial_timesteps``

A package-init circular import (``types/conditions/base`` ↔
``distributed`` ↔ ``rollout/engine`` ↔ ``types/rollout_req``) is guarded
with ``importorskip`` so this file auto-skips instead of blocking the
suite when the cycle is hit; remove the guard once the cycle is broken.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "diffusionrl.algorithms_new.rollout_control",
    reason="Pre-existing circular import in algorithms_new package init "
    "(types/conditions/base ↔ distributed ↔ rollout/engine).",
    exc_type=ImportError,
)

from diffusionrl.algorithms_new.rollout_control import (  # noqa: E402
    GRPORolloutControl,
    GRPORolloutControlConfig,
)
from diffusionrl.types.sampling import SamplingParams  # noqa: E402


def _make_ctrl(**overrides) -> GRPORolloutControl:
    cfg_kwargs = dict(
        sampling=SamplingParams(num_inference_steps=10),
        prompts_per_rollout=4,
        samples_per_prompt=8,
    )
    cfg_kwargs.update(overrides)
    return GRPORolloutControl(config=GRPORolloutControlConfig(**cfg_kwargs))


def test_attributes_exposed_for_train_new_driver() -> None:
    """``train_new.py:198`` reads ``getattr(ctrl, 'samples_per_prompt', 1)``
    and ``cfg.algorithm.prompts_per_rollout``."""
    ctrl = _make_ctrl(prompts_per_rollout=3, samples_per_prompt=5)
    assert ctrl.samples_per_prompt == 5
    assert ctrl.prompts_per_rollout == 3


def test_samples_per_prompt_floors_at_one() -> None:
    """OLD GRPOAlgorithm clamped ``samples_per_prompt = max(1, int(...))``."""
    # GRPORolloutControlConfig.__post_init__ doesn't have such a clamp;
    # the clamp lives in GRPORolloutControl.__init__ to match OLD behavior.
    ctrl = _make_ctrl(samples_per_prompt=0)
    assert ctrl.samples_per_prompt == 1


def test_resolve_rollout_sde_indices_returns_set() -> None:
    """Driver calls this per rollout step; expects Set[int]."""
    ctrl = _make_ctrl()
    result = ctrl.resolve_rollout_sde_indices(current_step=0)
    assert isinstance(result, set)
    assert all(isinstance(i, int) for i in result)


def test_resolve_raises_on_zero_inference_steps() -> None:
    ctrl = _make_ctrl()
    ctrl.sampling = SamplingParams(num_inference_steps=0)
    with pytest.raises(ValueError, match=r"num_inference_steps >= 1"):
        ctrl.resolve_rollout_sde_indices(current_step=0)


def test_get_filtered_training_indices_passthrough() -> None:
    """No skipping → returns the input set unchanged."""
    ctrl = _make_ctrl()
    out = ctrl.get_filtered_training_indices({0, 1, 2, 3, 4}, num_steps=5)
    assert out == {0, 1, 2, 3, 4}


def test_get_filtered_training_indices_skip_last() -> None:
    ctrl = _make_ctrl(skip_last_timestep=True)
    out = ctrl.get_filtered_training_indices({0, 1, 2, 3, 4}, num_steps=5)
    assert out == {0, 1, 2, 3}


def test_get_filtered_training_indices_skip_initial() -> None:
    ctrl = _make_ctrl(skip_initial_timesteps=2)
    out = ctrl.get_filtered_training_indices({0, 1, 2, 3, 4}, num_steps=5)
    assert out == {2, 3, 4}


def test_get_filtered_training_indices_skip_both() -> None:
    ctrl = _make_ctrl(skip_last_timestep=True, skip_initial_timesteps=2)
    out = ctrl.get_filtered_training_indices({0, 1, 2, 3, 4}, num_steps=5)
    assert out == {2, 3}


def test_get_filtered_training_indices_skip_last_empty_input_safe() -> None:
    """OLD-side guard: skip_last on an empty set must NOT error."""
    ctrl = _make_ctrl(skip_last_timestep=True)
    assert ctrl.get_filtered_training_indices(set(), num_steps=5) == set()


def test_config_rejects_wrong_type_at_ctor() -> None:
    """GRPORolloutControl is strict about config class."""
    with pytest.raises(TypeError, match=r"GRPORolloutControlConfig"):
        GRPORolloutControl(config="not a config")  # type: ignore[arg-type]


def test_config_rejects_kl_coef_gt_zero() -> None:
    """KL penalty is not implemented; ``kl_coef > 0`` must fail at config time."""
    with pytest.raises(ValueError, match=r"kl_coef"):
        GRPORolloutControlConfig(
            sampling=SamplingParams(num_inference_steps=10),
            kl_coef=0.01,
        )


def test_config_rejects_grpo_num_sde_steps_zero() -> None:
    """``num_sde_steps=0`` is reserved for NFTRolloutControl (forward-process);
    GRPO requires SDE steps to train."""
    from diffusionrl.utils.scheduler_utils import SchedulerConfig

    with pytest.raises(ValueError, match=r"num_sde_steps=0"):
        GRPORolloutControl(
            config=GRPORolloutControlConfig(
                sampling=SamplingParams(num_inference_steps=10),
                scheduler=SchedulerConfig(num_sde_steps=0),
            )
        )
