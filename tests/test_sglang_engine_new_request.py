"""Unit tests for the new-protocol SGLang request translator.

Pins the contract of ``_to_sglang_kwargs`` (file:
``diffusionrl/rollout/engine/sglang/request.py``) without requiring a live
SGLang runtime.

Coverage:

- Initial-noise sources (pre-shipped via ``request_conditions`` vs absent).
- K-expansion → de-expansion when ``group_ids`` partition cleanly; passthrough
  when heterogeneous.
- SDE-mode kwargs (``rollout``, ``rollout_sde_type``, ``rollout_sde_indices``)
  gated on ``stage_params['diffusion']['sde_indices']``.
- ODE-mode (``sde_indices`` absent) — rollout family omitted entirely.
- Negative-prompt CFG invariant raises when ``return_negative_prompt_embeds``
  is not pinned alongside ``negative_prompt``.
- Engine pins (return_trajectory_latents, return_prompt_embeds, init_same_noise=False,
  seed=None, etc.) — locked in to detect protocol drift.
- ``noise_group_ids`` forwarded from ``req.group_ids``.
"""

from __future__ import annotations

import pytest
import torch

from diffusionrl.rollout.engine.sglang.config import SGLangEngineConfig
from diffusionrl.rollout.engine.sglang.request import (
    _deexpand_prompts_from_groups,
    _to_sglang_kwargs,
)
from diffusionrl.types.conditions.image import ImageLatentCondition
from diffusionrl.types.primitives import Texts
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.sampling import SamplingParams, SDEConfig


def _make_cfg(*, sampler_kwargs: dict | None = None) -> SGLangEngineConfig:
    sampling = SamplingParams(
        num_inference_steps=4,
        guidance_scale=4.5,
        height=64,
        width=64,
        num_frames=1,
        seed=0,
        num_samples_per_prompt=1,
        sde_config=SDEConfig(eta=0.7),
        sde_indices=None,
        sampler_kwargs=dict(sampler_kwargs or {}),
    )
    return SGLangEngineConfig(
        sampling=sampling,
        model_family="sd3",
        populate_conditions=True,
        init_same_noise=False,
        logprob_source="replay",
    )


def _make_req(
    *,
    prompts: list[str],
    group_ids: list[str] | None = None,
    sde_indices: list[int] | None = None,
    initial_latents: torch.Tensor | None = None,
    num_samples_per_prompt: int = 1,
) -> RolloutReq:
    sample_ids = [f"s{i}" for i in range(len(prompts))]
    if group_ids is None:
        group_ids = [f"g{i}" for i in range(len(prompts))]
    diffusion = {
        "height": 64,
        "width": 64,
        "num_inference_steps": 4,
        "guidance_scale": 4.5,
        "eta": 0.7,
        "seed": 12345,
        "sde_indices": sde_indices,
        "num_samples_per_prompt": int(num_samples_per_prompt),
    }
    rc: dict = {}
    if initial_latents is not None:
        rc["initial_latents"] = ImageLatentCondition(latents=initial_latents)
    return RolloutReq(
        sample_ids=sample_ids,
        group_ids=group_ids,
        primitives={"text": Texts(texts=list(prompts))},
        request_conditions=rc,
        stage_params={"diffusion": diffusion},
    )


# ---------------------------------------------------------------------------
# De-expansion
# ---------------------------------------------------------------------------


def test_deexpand_uniform_k_collapses():
    unique, k = _deexpand_prompts_from_groups(
        prompts=["a", "a", "b", "b"],
        group_ids=["g0", "g0", "g1", "g1"],
    )
    assert unique == ["a", "b"]
    assert k == 2


def test_deexpand_k_equals_one_passthrough():
    unique, k = _deexpand_prompts_from_groups(
        prompts=["a", "b", "c"],
        group_ids=["g0", "g1", "g2"],
    )
    assert unique == ["a", "b", "c"]
    assert k == 1


def test_deexpand_mismatched_prompts_in_group_passthrough():
    """If two samples claim the same group but disagree on prompt, no collapse."""
    unique, k = _deexpand_prompts_from_groups(
        prompts=["a", "DIFFERENT", "b", "b"],
        group_ids=["g0", "g0", "g1", "g1"],
    )
    assert unique == ["a", "DIFFERENT", "b", "b"]
    assert k == 1


def test_deexpand_heterogeneous_k_passthrough():
    """When groups have different K, no collapse."""
    unique, k = _deexpand_prompts_from_groups(
        prompts=["a", "a", "a", "b", "b"],
        group_ids=["g0", "g0", "g0", "g1", "g1"],
    )
    assert unique == ["a", "a", "a", "b", "b"]
    assert k == 1


# ---------------------------------------------------------------------------
# Initial noise plumbing
# ---------------------------------------------------------------------------


def test_initial_noise_pre_shipped_forwarded_verbatim():
    noise = torch.arange(2 * 4 * 8 * 8, dtype=torch.float32).reshape(2, 4, 8, 8)
    req = _make_req(prompts=["a", "b"], initial_latents=noise)
    kwargs = _to_sglang_kwargs(req, cfg=_make_cfg(), sde_label="sde", initial_noise=noise)
    assert "initial_noise" in kwargs
    assert torch.equal(kwargs["initial_noise"], noise)


def test_initial_noise_absent_when_caller_supplies_none():
    req = _make_req(prompts=["a", "b"])
    kwargs = _to_sglang_kwargs(req, cfg=_make_cfg(), sde_label="sde", initial_noise=None)
    assert "initial_noise" not in kwargs


# ---------------------------------------------------------------------------
# SDE / ODE mode gating
# ---------------------------------------------------------------------------


def test_sde_mode_kwargs_present_when_sde_indices_set():
    req = _make_req(prompts=["a"], sde_indices=[0, 2])
    kwargs = _to_sglang_kwargs(req, cfg=_make_cfg(), sde_label="sde", initial_noise=None)
    assert kwargs["rollout"] is True
    assert kwargs["rollout_sde_type"] == "sde"
    assert kwargs["rollout_noise_level"] == 0.7
    assert kwargs["rollout_sde_indices"] == [0, 2]


def test_ode_mode_omits_rollout_family():
    req = _make_req(prompts=["a"], sde_indices=None)
    kwargs = _to_sglang_kwargs(req, cfg=_make_cfg(), sde_label="sde", initial_noise=None)
    assert "rollout" not in kwargs
    assert "rollout_sde_type" not in kwargs
    assert "rollout_noise_level" not in kwargs
    assert "rollout_sde_indices" not in kwargs


def test_sde_mode_requires_sde_label():
    req = _make_req(prompts=["a"], sde_indices=[0])
    with pytest.raises(Exception):
        _to_sglang_kwargs(req, cfg=_make_cfg(), sde_label=None, initial_noise=None)


# ---------------------------------------------------------------------------
# Engine pins + noise_group_ids
# ---------------------------------------------------------------------------


def test_engine_pins_locked_in():
    req = _make_req(prompts=["a"])
    kwargs = _to_sglang_kwargs(req, cfg=_make_cfg(), sde_label=None, initial_noise=None)
    assert kwargs["return_trajectory_latents"] is True
    assert kwargs["return_prompt_embeds"] is True
    assert kwargs["return_trajectory_decoded"] is False
    assert kwargs["save_output"] is False
    assert kwargs["return_file_paths_only"] is False
    assert kwargs["init_same_noise"] is False  # SGLang-side group sharing disabled
    assert kwargs["seed"] is None


def test_noise_group_ids_forwarded_from_req():
    req = _make_req(prompts=["a", "b"], group_ids=["alpha", "beta"])
    kwargs = _to_sglang_kwargs(req, cfg=_make_cfg(), sde_label=None, initial_noise=None)
    assert kwargs["noise_group_ids"] == ["alpha", "beta"]


def test_num_outputs_per_prompt_set_when_deexpanded():
    req = _make_req(
        prompts=["a", "a", "b", "b"],
        group_ids=["g0", "g0", "g1", "g1"],
    )
    kwargs = _to_sglang_kwargs(req, cfg=_make_cfg(), sde_label=None, initial_noise=None)
    assert kwargs["prompt"] == ["a", "b"]
    assert kwargs["num_outputs_per_prompt"] == 2


def test_num_outputs_per_prompt_omitted_when_k_is_one():
    req = _make_req(prompts=["a", "b"], group_ids=["g0", "g1"])
    kwargs = _to_sglang_kwargs(req, cfg=_make_cfg(), sde_label=None, initial_noise=None)
    # single-prompt-per-group: prompt is the bare list (length 2), no num_outputs_per_prompt
    assert "num_outputs_per_prompt" not in kwargs


# ---------------------------------------------------------------------------
# Negative-prompt CFG invariant
# ---------------------------------------------------------------------------


def test_negative_prompt_without_return_flag_raises():
    cfg = _make_cfg(sampler_kwargs={"negative_prompt": "blurry"})
    req = _make_req(prompts=["a"])
    with pytest.raises(Exception):
        _to_sglang_kwargs(req, cfg=cfg, sde_label=None, initial_noise=None)


def test_negative_prompt_with_return_flag_passes():
    cfg = _make_cfg(
        sampler_kwargs={
            "negative_prompt": "blurry",
            "return_negative_prompt_embeds": True,
        }
    )
    req = _make_req(prompts=["a"])
    kwargs = _to_sglang_kwargs(req, cfg=cfg, sde_label=None, initial_noise=None)
    assert kwargs["negative_prompt"] == "blurry"
    assert kwargs["return_negative_prompt_embeds"] is True
