"""Unit tests for ``SGLangRolloutRequest`` mode discrimination.

The SGLang engine has two rollout modes — discriminated entirely by
``sampling_params.sde_indices``:

  - **SDE mode** (``sde_indices`` is a non-empty set, e.g. GRPO):
    ``to_kwargs()`` pins ``rollout=True`` and the ``rollout_sde_*``
    family so SGLang runs stochastic SDE sampling.

  - **ODE / non-SDE mode** (``sde_indices is None``, e.g. eval and
    NFT-train): ``to_kwargs()`` omits the rollout family entirely so
    SGLang runs deterministic sampling. ``return_trajectory_latents``
    stays True regardless — downstream consumers derive
    ``final_latents`` from ``trajectory_latents[:, -1]``.

These tests pin both shapes so a future refactor can't silently break
either branch without test churn. No SGLang runtime needed.
"""

from __future__ import annotations

import pytest
import torch

from diffusionrl.samplers.sglang.request import SGLangRolloutRequest
from diffusionrl.types.prompts import Prompts
from diffusionrl.types.request import RolloutRequest
from diffusionrl.types.sampling import SamplingParams, SDEConfig


def _build_request(
    *,
    sde_indices,
    num_samples_per_prompt: int = 1,
    sampler_kwargs: dict | None = None,
) -> RolloutRequest:
    sp = SamplingParams(
        num_inference_steps=4,
        guidance_scale=4.5,
        height=64,
        width=64,
        num_frames=1,
        seed=0,
        num_samples_per_prompt=num_samples_per_prompt,
        sde_config=SDEConfig(eta=0.7, shift=3.0),
        sde_indices=sde_indices,
        sampler_kwargs=dict(sampler_kwargs or {}),
    )
    prompts = Prompts.from_unique_prompts(["a photo"], prompt_ids=["p0"])
    if num_samples_per_prompt > 1:
        prompts = prompts.expand(num_samples_per_prompt, init_same_noise=False)
    return RolloutRequest(prompts=prompts, sampling_params=sp)


def _populate_engine_state(req: SGLangRolloutRequest, *, sde_type: str | None) -> None:
    """Mirror the engine's pre-``to_kwargs`` population step."""
    req.initial_noise = torch.zeros(1)
    if sde_type is not None:
        req.rollout_sde_type = sde_type


# ---------------------------------------------------------------------------
# ODE / non-SDE mode (sde_indices=None) — eval, NFT-train
# ---------------------------------------------------------------------------


def test_from_rollout_request_accepts_none_sde_indices():
    """Eval and NFT-train pass sde_indices=None; this used to crash."""
    req = SGLangRolloutRequest.from_rollout_request(_build_request(sde_indices=None))
    assert req.rollout_sde_indices is None


def test_to_kwargs_ode_mode_omits_rollout_block():
    req = SGLangRolloutRequest.from_rollout_request(_build_request(sde_indices=None))
    _populate_engine_state(req, sde_type=None)  # not required in ODE mode

    kwargs = req.to_kwargs()

    # The full rollout-kernel family must be absent — SGLang then runs ODE.
    assert "rollout" not in kwargs
    assert "rollout_sde_type" not in kwargs
    assert "rollout_noise_level" not in kwargs
    assert "rollout_sde_indices" not in kwargs

    # Trajectory return must stay on regardless of mode — downstream reads
    # ``trajectory_latents[:, -1]`` for the final clean latent.
    assert kwargs["return_trajectory_latents"] is True
    assert kwargs["return_prompt_embeds"] is True


# ---------------------------------------------------------------------------
# SDE mode (sde_indices non-empty) — GRPO
# ---------------------------------------------------------------------------


def test_to_kwargs_sde_mode_pins_rollout_block():
    req = SGLangRolloutRequest.from_rollout_request(_build_request(sde_indices=[0, 2, 1]))
    _populate_engine_state(req, sde_type="sde")

    kwargs = req.to_kwargs()

    assert kwargs["rollout"] is True
    assert kwargs["rollout_sde_type"] == "sde"
    assert kwargs["rollout_noise_level"] == pytest.approx(0.7)
    # sde_indices are stored sorted so SGLang sees a deterministic list
    assert kwargs["rollout_sde_indices"] == [0, 1, 2]
    assert kwargs["return_trajectory_latents"] is True


def test_to_kwargs_sde_mode_requires_rollout_sde_type():
    """Engine must populate rollout_sde_type before to_kwargs() in SDE mode."""
    req = SGLangRolloutRequest.from_rollout_request(_build_request(sde_indices=[0]))
    _populate_engine_state(req, sde_type=None)  # forgotten

    with pytest.raises(Exception, match="rollout_sde_type"):
        req.to_kwargs()


# ---------------------------------------------------------------------------
# Shared invariants
# ---------------------------------------------------------------------------


def test_to_kwargs_requires_initial_noise_in_either_mode():
    for sde_indices in (None, [0]):
        req = SGLangRolloutRequest.from_rollout_request(_build_request(sde_indices=sde_indices))
        if sde_indices is not None:
            req.rollout_sde_type = "sde"
        # initial_noise deliberately not populated
        with pytest.raises(Exception, match="initial_noise"):
            req.to_kwargs()


# ---------------------------------------------------------------------------
# sampler_kwargs escape-hatch (issue 4)
# ---------------------------------------------------------------------------


def test_sampler_kwargs_pass_through_for_unknown_keys():
    """Arbitrary sampler_kwargs reach SGLang verbatim — escape-hatch contract."""
    req = SGLangRolloutRequest.from_rollout_request(
        _build_request(
            sde_indices=None,
            sampler_kwargs={
                "negative_prompt": "ugly, blurry",
                "return_negative_prompt_embeds": True,
                "fps": 24,
            },
        )
    )
    _populate_engine_state(req, sde_type=None)

    kwargs = req.to_kwargs()

    assert kwargs["negative_prompt"] == "ugly, blurry"
    assert kwargs["return_negative_prompt_embeds"] is True
    assert kwargs["fps"] == 24


def test_negative_prompt_without_return_embeds_raises():
    """``negative_prompt`` without ``return_negative_prompt_embeds=True`` is rejected.

    Without the guard, rollout uses the negative prompt for CFG (gated by
    ``guidance_scale>1``, not by ``return_negative_prompt_embeds``), but the
    training-side replay falls back to zero negative embeds — a silent GRPO
    ratio mismatch.
    """
    req = SGLangRolloutRequest.from_rollout_request(
        _build_request(
            sde_indices=None,
            sampler_kwargs={"negative_prompt": "ugly, blurry"},
        )
    )
    _populate_engine_state(req, sde_type=None)
    with pytest.raises(ValueError, match="return_negative_prompt_embeds"):
        req.to_kwargs()


def test_engine_pins_override_sampler_kwargs():
    """User cannot smash the trajectory / shape contract via sampler_kwargs.

    Two-layer defense (do not collapse into one):
      - typed fields (``init_same_noise``, ``seed``, ``num_inference_steps``,
        ...) are rejected at ``SamplingParams.__post_init__`` — covered by
        ``tests/test_post_init_invariants.py``.
      - non-typed engine-pinned keys (``return_trajectory_latents``,
        ``save_output``, ``return_prompt_embeds``) are overridden at the
        SGLang request layer; that's what this test pins.
    """
    req = SGLangRolloutRequest.from_rollout_request(
        _build_request(
            sde_indices=None,
            sampler_kwargs={
                # Hostile overrides: each one would break a downstream invariant.
                "return_trajectory_latents": False,  # NFT/eval need final latent
                "save_output": True,
                "return_prompt_embeds": False,  # forward_context needs them
            },
        )
    )
    _populate_engine_state(req, sde_type=None)

    kwargs = req.to_kwargs()

    assert kwargs["return_trajectory_latents"] is True
    assert kwargs["save_output"] is False
    assert kwargs["return_prompt_embeds"] is True
    # Engine also pins these unconditionally — the typed-field defaults flow
    # through the SamplingParams layer and the engine still overrides them.
    assert kwargs["init_same_noise"] is False
    assert kwargs["seed"] is None


def test_sde_mode_kwargs_override_sampler_kwargs():
    """In SDE mode, the rollout block overrides sampler_kwargs values."""
    req = SGLangRolloutRequest.from_rollout_request(
        _build_request(
            sde_indices=[0, 1],
            sampler_kwargs={
                "rollout": False,  # would silently degrade GRPO to ODE
                "rollout_sde_type": "wrong",
                "rollout_noise_level": 99.0,
                "rollout_sde_indices": [42],
            },
        )
    )
    _populate_engine_state(req, sde_type="sde")

    kwargs = req.to_kwargs()

    assert kwargs["rollout"] is True
    assert kwargs["rollout_sde_type"] == "sde"
    assert kwargs["rollout_noise_level"] == pytest.approx(0.7)
    assert kwargs["rollout_sde_indices"] == [0, 1]


def test_empty_sampler_kwargs_default():
    """Default empty sampler_kwargs leaves output identical to before issue 4."""
    req = SGLangRolloutRequest.from_rollout_request(_build_request(sde_indices=None))
    _populate_engine_state(req, sde_type=None)
    kwargs = req.to_kwargs()
    # No leaked entries from sampler_kwargs.
    assert "negative_prompt" not in kwargs
    assert "return_negative_prompt_embeds" not in kwargs
    assert "fps" not in kwargs
