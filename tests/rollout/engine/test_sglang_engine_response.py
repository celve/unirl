"""Unit tests for the SGLang response translator.

Pins the contract of ``_to_rollout_resp`` (file:
``diffusionrl/rollout/engine/sglang/response.py``) without requiring a live
SGLang runtime. Mock ``GenerationResult`` objects are built with
``SimpleNamespace`` carrying just the fields the translator reads.

Coverage:

- T+1 trajectory invariant raises on shape mismatch.
- Sigma cross-check raises on value drift in ``trajectory_timesteps``.
- Selective trim with subset ``sde_indices`` → ``LatentSegment.indices``
  matches ``compute_trajectory_positions``.
- ``conditions['text']`` populated when prompt embeds present.
- ``conditions['negative_text']`` populated when CFG negative embeds present;
  absent when not.
- ``conditions`` empty when ``cfg.populate_conditions=False``.
- ``tracks['image'].decoded`` is Images float32 [B, C, H, W] in [0, 1].
- Native vs replay logp policy: ``sde_logp`` populated in native mode, None
  in replay.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from diffusionrl.rollout.engine.sglang.config import SGLangEngineConfig
from diffusionrl.rollout.engine.sglang.response import _to_rollout_resp
from diffusionrl.sde.runtime import get_sigma_schedule
from diffusionrl.types.primitives import Texts
from diffusionrl.types.rollout_req import RolloutReq
from diffusionrl.types.sampling import DiffusionSamplingParams

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_cfg(*, populate_conditions: bool = True) -> SGLangEngineConfig:
    sampling = DiffusionSamplingParams(
        num_inference_steps=4,
        guidance_scale=4.5,
        height=64,
        width=64,
        num_frames=1,
        seed=0,
        num_samples_per_prompt=1,
        eta=0.7,
        sde_indices=None,
        sampler_kwargs={},
    )
    return SGLangEngineConfig(
        sampling=sampling,
        model_family="sd3",
        populate_conditions=populate_conditions,
    )


def _make_req(batch: int = 2) -> RolloutReq:
    return RolloutReq(
        sample_ids=[f"s{i}" for i in range(batch)],
        group_ids=[f"g{i}" for i in range(batch)],
        primitives={"text": Texts(texts=[f"prompt_{i}" for i in range(batch)])},
        sampling_params=DiffusionSamplingParams(
            num_inference_steps=4,
            samples_per_prompt=1,
        ),
    )


def _make_result(
    *,
    num_steps: int = 4,
    shift: float = 3.0,
    batch_per_result: int = 1,
    channels: int = 4,
    h: int = 8,
    w: int = 8,
    with_log_probs: bool = False,
    with_neg_embeds: bool = False,
    sample_h: int = 64,
    sample_w: int = 64,
    text_tokens: int = 7,
    text_hidden: int = 12,
) -> SimpleNamespace:
    """Mock one SGLang ``GenerationResult``."""
    sigmas = get_sigma_schedule(num_steps, shift=shift)
    traj_latents = torch.zeros(batch_per_result, num_steps + 1, channels, h, w)
    traj_log_probs = torch.zeros(batch_per_result, num_steps) if with_log_probs else None
    # Decoded sample: per-sample [C, H, W] float in [0, 1] — SGLang's
    # decode_sample contract surfaces a [C, H, W] tensor per-sample. Wrap as
    # a list of per-sample tensors so decode_sample can iterate.
    samples = [torch.rand(3, sample_h, sample_w) for _ in range(batch_per_result)]
    prompt_embeds = torch.randn(batch_per_result, text_tokens, text_hidden)
    pooled = torch.randn(batch_per_result, text_hidden)
    encoder_attention_mask = torch.ones(batch_per_result, text_tokens)
    neg_embeds = torch.randn(batch_per_result, text_tokens, text_hidden) if with_neg_embeds else None
    neg_pooled = torch.randn(batch_per_result, text_hidden) if with_neg_embeds else None
    return SimpleNamespace(
        trajectory_latents=traj_latents,
        trajectory_log_probs=traj_log_probs,
        trajectory_timesteps=sigmas,
        samples=samples[0] if batch_per_result == 1 else samples,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled,
        encoder_attention_mask=encoder_attention_mask,
        negative_prompt_embeds=neg_embeds,
        neg_pooled_prompt_embeds=neg_pooled,
    )


# ---------------------------------------------------------------------------
# Trajectory invariants
# ---------------------------------------------------------------------------


def test_t_plus_one_invariant_raises_on_short_traj():
    cfg = _make_cfg()
    bad_result = _make_result()
    # Trim one column so traj has T columns instead of T+1.
    bad_result.trajectory_latents = bad_result.trajectory_latents[:, :-1]
    with pytest.raises(Exception):
        _to_rollout_resp(
            _make_req(batch=1),
            [bad_result],
            cfg=cfg,
            num_steps=4,
            shift=3.0,
            sde_indices=None,
            use_native_logprob=False,
        )


def test_sigma_value_drift_raises():
    cfg = _make_cfg()
    bad_result = _make_result()
    # Inject a clearly-wrong sigma schedule on the result.
    bad_result.trajectory_timesteps = torch.linspace(0.5, 0.1, 5)
    with pytest.raises(Exception):
        _to_rollout_resp(
            _make_req(batch=1),
            [bad_result],
            cfg=cfg,
            num_steps=4,
            shift=3.0,
            sde_indices=None,
            use_native_logprob=False,
        )


def test_sgl_timesteps_diffusers_scale_normalized_before_compare():
    """SGLang may emit trajectory_timesteps in [0, num_train_timesteps] (the
    diffusers convention), not the [0, 1] normalized scale ``get_sigma_schedule``
    returns. The cross-check must divide by 1000 before comparing — same
    schedule, different units.
    """
    cfg = _make_cfg()
    result = _make_result()
    # Take the normalized sigmas and multiply by 1000 to mimic SGLang's emission.
    result.trajectory_timesteps = result.trajectory_timesteps * 1000.0
    # Should NOT raise: after the engine normalizes the units, schedules match.
    resp = _to_rollout_resp(
        _make_req(batch=1),
        [result],
        cfg=cfg,
        num_steps=4,
        shift=3.0,
        sde_indices=None,
        use_native_logprob=False,
    )
    # Sanity: segment built; sigmas on the segment stay in normalized [0,1]
    # (they come from get_sigma_schedule, not SGLang).
    seg = resp.tracks["image"].segment
    assert float(seg.sigmas.max().item()) <= 1.0
    assert float(seg.sigmas.min().item()) >= 0.0


# ---------------------------------------------------------------------------
# LatentSegment population
# ---------------------------------------------------------------------------


def test_segment_full_trajectory_when_no_subset_trim():
    cfg = _make_cfg()
    results = [_make_result(batch_per_result=1) for _ in range(2)]
    resp = _to_rollout_resp(
        _make_req(batch=2),
        results,
        cfg=cfg,
        num_steps=4,
        shift=3.0,
        sde_indices=None,
        use_native_logprob=False,
    )
    seg = resp.tracks["image"].segment
    assert seg.latents.shape == (2, 5, 4, 8, 8)
    assert seg.sigmas.shape == (5,)
    assert torch.equal(seg.indices, torch.arange(5))
    assert torch.equal(seg.sample_indices, torch.arange(2))
    assert seg.sde_logp is None
    assert seg.sde_indices is None
    # Diffusion-side fields not in LatentSegment's contract.
    assert seg.log_probs is None
    assert seg.loss_mask is None


def test_segment_selective_trim_with_subset_sde_indices():
    cfg = _make_cfg()
    results = [_make_result(batch_per_result=1) for _ in range(2)]
    resp = _to_rollout_resp(
        _make_req(batch=2),
        results,
        cfg=cfg,
        num_steps=4,
        shift=3.0,
        sde_indices=[1],  # needs positions [1, 2]
        use_native_logprob=False,
    )
    seg = resp.tracks["image"].segment
    # Trimmed to positions [1, 2] → 2 columns out of 5.
    assert seg.latents.shape == (2, 2, 4, 8, 8)
    assert torch.equal(seg.indices, torch.tensor([1, 2]))


def test_native_logp_lands_on_sde_logp():
    cfg = _make_cfg()
    results = [_make_result(batch_per_result=1, with_log_probs=True) for _ in range(2)]
    # Set deterministic log_probs so we can check the concat.
    results[0].trajectory_log_probs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    results[1].trajectory_log_probs = torch.tensor([[5.0, 6.0, 7.0, 8.0]])
    resp = _to_rollout_resp(
        _make_req(batch=2),
        results,
        cfg=cfg,
        num_steps=4,
        shift=3.0,
        sde_indices=None,
        use_native_logprob=True,
    )
    seg = resp.tracks["image"].segment
    assert seg.sde_logp is not None
    assert seg.sde_logp.shape == (2, 4)
    assert torch.equal(seg.sde_logp[0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.equal(seg.sde_logp[1], torch.tensor([5.0, 6.0, 7.0, 8.0]))
    assert seg.sde_indices is not None
    assert torch.equal(seg.sde_indices, torch.arange(4))


def test_replay_mode_leaves_both_sde_logp_and_sde_indices_none():
    """Replay mode (``use_native_logprob=False``) intentionally leaves both
    ``sde_logp`` and ``sde_indices`` ``None`` on the segment. The two travel
    together — segments produced this way are NOT consumable by trainer-side
    ``SD3DiffusionStage.replay`` (which requires ``sde_indices`` non-None).
    Training rollouts must use ``logprob_source='native'``.
    """
    cfg = _make_cfg()
    results = [_make_result(batch_per_result=1, with_log_probs=True) for _ in range(2)]
    resp = _to_rollout_resp(
        _make_req(batch=2),
        results,
        cfg=cfg,
        num_steps=4,
        shift=3.0,
        sde_indices=[0, 1, 2, 3],  # request SDE mode
        use_native_logprob=False,
    )
    seg = resp.tracks["image"].segment
    assert seg.sde_logp is None
    assert seg.sde_indices is None


# ---------------------------------------------------------------------------
# Conditions packing
# ---------------------------------------------------------------------------


def test_conditions_text_populated_when_populate_on():
    cfg = _make_cfg(populate_conditions=True)
    results = [_make_result(batch_per_result=1) for _ in range(2)]
    resp = _to_rollout_resp(
        _make_req(batch=2),
        results,
        cfg=cfg,
        num_steps=4,
        shift=3.0,
        sde_indices=None,
        use_native_logprob=False,
    )
    assert "text" in resp.tracks["image"].conditions
    text_cond = resp.tracks["image"].conditions["text"]
    assert text_cond.embeds.shape == (2, 7, 12)
    assert text_cond.pooled.shape == (2, 12)
    assert text_cond.attn_mask.shape == (2, 7)


def test_conditions_negative_text_populated_only_when_cfg_active():
    cfg = _make_cfg(populate_conditions=True)
    # CFG active: results carry negative_prompt_embeds.
    results = [_make_result(batch_per_result=1, with_neg_embeds=True) for _ in range(2)]
    resp = _to_rollout_resp(
        _make_req(batch=2),
        results,
        cfg=cfg,
        num_steps=4,
        shift=3.0,
        sde_indices=None,
        use_native_logprob=False,
    )
    assert "negative_text" in resp.tracks["image"].conditions
    neg = resp.tracks["image"].conditions["negative_text"]
    assert neg.embeds.shape == (2, 7, 12)
    assert neg.pooled.shape == (2, 12)

    # CFG inactive: no negative embeds emitted.
    results_no_neg = [_make_result(batch_per_result=1, with_neg_embeds=False) for _ in range(2)]
    resp2 = _to_rollout_resp(
        _make_req(batch=2),
        results_no_neg,
        cfg=cfg,
        num_steps=4,
        shift=3.0,
        sde_indices=None,
        use_native_logprob=False,
    )
    assert "negative_text" not in resp2.conditions


def test_populate_conditions_off_yields_empty_dict():
    cfg = _make_cfg(populate_conditions=False)
    results = [_make_result(batch_per_result=1, with_neg_embeds=True) for _ in range(2)]
    resp = _to_rollout_resp(
        _make_req(batch=2),
        results,
        cfg=cfg,
        num_steps=4,
        shift=3.0,
        sde_indices=None,
        use_native_logprob=False,
    )
    assert resp.tracks["image"].conditions == {}


# ---------------------------------------------------------------------------
# Decoded images
# ---------------------------------------------------------------------------


def test_decoded_images_shape_and_range():
    cfg = _make_cfg()
    results = [_make_result(batch_per_result=1) for _ in range(2)]
    resp = _to_rollout_resp(
        _make_req(batch=2),
        results,
        cfg=cfg,
        num_steps=4,
        shift=3.0,
        sde_indices=None,
        use_native_logprob=False,
    )
    decoded = resp.tracks["image"].decoded
    assert decoded.pixels.dtype == torch.float32
    assert decoded.pixels.shape == (2, 3, 64, 64)
    assert float(decoded.pixels.min().item()) >= 0.0
    assert float(decoded.pixels.max().item()) <= 1.0


# ---------------------------------------------------------------------------
# Sample/group id round-trip
# ---------------------------------------------------------------------------


def test_sample_and_group_ids_carried_through():
    cfg = _make_cfg()
    req = _make_req(batch=2)
    results = [_make_result(batch_per_result=1) for _ in range(2)]
    resp = _to_rollout_resp(
        req,
        results,
        cfg=cfg,
        num_steps=4,
        shift=3.0,
        sde_indices=None,
        use_native_logprob=False,
    )
    assert resp.tracks["image"].sample_ids == req.sample_ids
    assert resp.tracks["image"].group_ids == req.group_ids
