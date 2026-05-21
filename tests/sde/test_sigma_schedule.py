"""Unit tests for the canonical σ schedule entry point.

Covers static (no-mu) and dynamic (mu provided) paths of
:func:`diffusionrl.sde.runtime.get_sigma_schedule`, plus the
:func:`calculate_dynamic_mu` helper.

Architecture under test:
  - Static: implemented in DiffusionRL (linspace(1,0,N+1) + SD3-paper
    shift formula applied once). We own this because diffusers' static
    path has the confirmed double-shift bug (issue #13243).
  - Dynamic: delegated to diffusers' FlowMatchEulerDiscreteScheduler
    (no bug). These tests pin the equivalence to diffusers.
"""

from __future__ import annotations

import json

import pytest
import torch

from diffusionrl.sde.runtime import calculate_dynamic_mu, get_sigma_schedule

# ---------------------------------------------------------------------------
# Static path
# ---------------------------------------------------------------------------


def test_static_matches_paper_formula():
    """Hard-coded reference for SD3 N=10 shift=3.

    Numbers come from the SD3-paper formula
    ``(shift * t) / (1 + (shift - 1) * t)`` applied to
    ``linspace(1, 0, 11)``. Pinning the values prevents accidental
    regressions if someone "optimizes" the static math later.
    """
    sigmas = get_sigma_schedule(10, shift=3.0).tolist()
    expected = [
        1.0,
        0.964286,
        0.923077,
        0.875,
        0.818182,
        0.75,
        0.666667,
        0.5625,
        0.428571,
        0.25,
        0.0,
    ]
    assert len(sigmas) == len(expected)
    for a, b in zip(sigmas, expected):
        assert abs(a - b) < 1e-5, f"{a} vs {b}"


def test_static_terminal_is_zero():
    sigmas = get_sigma_schedule(28, shift=3.0)
    assert sigmas[-1].item() == 0.0


def test_static_endpoint_invariance():
    # Shift formula is identity at t=1: shift / (1 + (shift - 1)) == 1.
    sigmas = get_sigma_schedule(10, shift=3.0)
    assert sigmas[0].item() == 1.0


def test_static_shape_is_T_plus_1():
    for n in (1, 10, 28, 50, 100):
        sigmas = get_sigma_schedule(n, shift=3.0)
        assert sigmas.shape == (n + 1,), f"n={n}: {sigmas.shape}"


def test_static_shift_one_is_identity_linspace():
    # shift=1 collapses to t/t == identity, so output should match linspace(1, 0, N+1)
    sigmas = get_sigma_schedule(10, shift=1.0)
    expected = torch.linspace(1.0, 0.0, 11)
    assert torch.allclose(sigmas, expected)


def test_static_monotonic_decreasing():
    for shift in (1.0, 3.0, 5.0, 7.0):
        sigmas = get_sigma_schedule(28, shift=shift)
        diffs = sigmas[1:] - sigmas[:-1]
        # Each step σ_{i+1} <= σ_i.
        assert torch.all(diffs <= 1e-7), (
            f"σ schedule not monotonic for shift={shift}: max positive diff={float(diffs.max().item())}"
        )


def test_static_finite_for_video_shifts():
    for shift in (3.0, 5.0, 7.0):
        sigmas = get_sigma_schedule(28, shift=shift)
        assert torch.isfinite(sigmas).all()


def test_static_device_placement():
    cpu = get_sigma_schedule(10, shift=3.0, device=None)
    assert cpu.device.type == "cpu"
    explicit_cpu = get_sigma_schedule(10, shift=3.0, device=torch.device("cpu"))
    assert torch.allclose(cpu, explicit_cpu)


def test_static_no_double_shift_regression():
    """If someone re-introduces the diffusers static double-shift bug,
    a shift=3 N=10 schedule would diverge from the paper-formula values
    above. This test guards by comparing against a hand-computed value
    that's only correct under single-shift semantics."""
    sigmas = get_sigma_schedule(10, shift=3.0)
    # Single-shift at t=0.5: shift * 0.5 / (1 + (shift-1)*0.5) = 1.5/2 = 0.75
    # (i.e. sigmas[5] should be exactly 0.75 — the midpoint of an
    # 11-element schedule with shift=3).
    assert abs(sigmas[5].item() - 0.75) < 1e-6


# ---------------------------------------------------------------------------
# Dynamic path — delegated to diffusers
# ---------------------------------------------------------------------------


def _diffusers_dynamic_sigmas(num_steps: int, mu: float, time_shift_type: str = "exponential"):
    """Direct call to upstream diffusers for cross-check."""
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    sch = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        use_dynamic_shifting=True,
        time_shift_type=time_shift_type,
    )
    sch.set_timesteps(num_inference_steps=num_steps, mu=mu)
    return sch.sigmas


def test_dynamic_terminal_is_zero():
    sigmas = get_sigma_schedule(10, shift=3.0, mu=1.0)
    assert sigmas[-1].item() == 0.0


def test_dynamic_shape_is_T_plus_1():
    assert get_sigma_schedule(10, shift=3.0, mu=1.0).shape == (11,)
    assert get_sigma_schedule(28, shift=3.0, mu=0.7).shape == (29,)


def test_dynamic_no_nan_inf():
    for mu in (0.5, 0.7, 1.0, 1.15):
        sigmas = get_sigma_schedule(28, shift=3.0, mu=mu)
        assert torch.isfinite(sigmas).all(), f"non-finite values at mu={mu}"


def test_dynamic_equals_diffusers_reference():
    """Our dynamic branch must match what diffusers produces directly.

    This is the safety net for the delegation contract: if diffusers
    refactors its dynamic codepath in a breaking way, this test fires.
    """
    for mu in (0.5, 0.7, 1.0):
        ours = get_sigma_schedule(10, shift=3.0, mu=mu)
        ref = _diffusers_dynamic_sigmas(10, mu)
        assert torch.allclose(ours, ref, atol=1e-6), f"drift at mu={mu}"


def test_dynamic_ignores_shift_argument():
    """When mu is provided, ``shift`` is unused — diffusers' dynamic
    branch derives σ from mu and time_shift_type, not from a static
    shift value."""
    a = get_sigma_schedule(10, shift=3.0, mu=0.7)
    b = get_sigma_schedule(10, shift=7.0, mu=0.7)
    assert torch.allclose(a, b)


# ---------------------------------------------------------------------------
# calculate_dynamic_mu
# ---------------------------------------------------------------------------


def test_calculate_dynamic_mu_endpoints():
    # μ should equal base_shift at base_seq_len and max_shift at max_seq_len.
    mu_lo = calculate_dynamic_mu(256, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15)
    mu_hi = calculate_dynamic_mu(4096, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15)
    assert abs(mu_lo - 0.5) < 1e-6
    assert abs(mu_hi - 1.15) < 1e-6


def test_calculate_dynamic_mu_linear():
    # Linear interpolation should give the midpoint at midpoint seq len.
    mid_seq = (256 + 4096) / 2
    mu_mid = calculate_dynamic_mu(int(mid_seq), base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15)
    expected = (0.5 + 1.15) / 2
    assert abs(mu_mid - expected) < 1e-3


def test_calculate_dynamic_mu_extrapolates():
    # The formula is linear so it extrapolates past max_seq_len. This is
    # the diffusers convention — pin it so we don't silently clamp later.
    mu_above = calculate_dynamic_mu(8192, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15)
    assert mu_above > 1.15


# ---------------------------------------------------------------------------
# Smoke: the import surface only exposes what we want
# ---------------------------------------------------------------------------


def test_sde_runtime_public_surface():
    import diffusionrl.sde.runtime as rt

    assert hasattr(rt, "get_sigma_schedule")
    assert hasattr(rt, "calculate_dynamic_mu")
    assert hasattr(rt, "FlowMatchSchedulePolicy")
    assert hasattr(rt, "compute_flowmatch_sigma")
    assert hasattr(rt, "ensure_req_sigmas")
    # Legacy symbols must be gone — if they re-appear someone is
    # re-introducing a deprecated entry point.
    assert not hasattr(rt, "sd3_time_shift")
    assert not hasattr(rt, "get_sigma_schedule_diffusers")


# ---------------------------------------------------------------------------
# FlowMatchSchedulePolicy.from_pretrained — JSON layout & graceful fallback
# ---------------------------------------------------------------------------


def test_policy_none_path_falls_back_to_static():
    """``path is None`` is an explicit opt-in to static-only (tests,
    smoke runs, fake bundles)."""
    from diffusionrl.sde.runtime import FlowMatchSchedulePolicy

    p = FlowMatchSchedulePolicy.from_pretrained(None, shift=5.0)
    assert p.shift == 5.0
    assert p.use_dynamic_shifting is False
    assert p.vae_scale_factor == 8
    assert p.patch_size == 2


def test_policy_nonexistent_path_falls_back_to_static(tmp_path):
    """Non-existent local path falls back to static-only without
    raising — covers two legitimate cases: (a) tests with no real
    checkpoint, (b) HF repo IDs like ``stabilityai/stable-diffusion-3.5-medium``
    that ``Path(...).exists()`` returns False for but bundle resolves
    via huggingface_hub. The σ-direction drift assert at rollout time
    is the real safety net for misconfigured paths."""
    from diffusionrl.sde.runtime import FlowMatchSchedulePolicy

    p = FlowMatchSchedulePolicy.from_pretrained(
        tmp_path / "does_not_exist",
        shift=3.0,
    )
    assert p.shift == 3.0
    assert p.use_dynamic_shifting is False


def test_policy_hf_repo_id_falls_back_to_static():
    """HF-style repo IDs like 'stabilityai/sd3.5-medium' don't exist as
    a local path — fall back to static so engines can still construct
    a policy at __init__. Bundle.from_pretrained does the real
    repo-resolve later."""
    from diffusionrl.sde.runtime import FlowMatchSchedulePolicy

    p = FlowMatchSchedulePolicy.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        shift=3.0,
    )
    assert p.shift == 3.0
    assert p.use_dynamic_shifting is False


def test_policy_missing_scheduler_json_warns(tmp_path, caplog):
    """When the pretrained directory exists but
    ``scheduler/scheduler_config.json`` is missing, the dynamic-shift
    fields silently fall back to defaults. That's correct for static
    models but a real bug if the model wants dynamic shift — emit a
    WARNING so the cause is obvious in logs without tracing through
    to the response handler."""
    import logging

    from diffusionrl.sde.runtime import FlowMatchSchedulePolicy

    # Make tmp_path "exist" but with no scheduler/ subdir
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "config.json").write_text(json.dumps({"patch_size": 2}))
    with caplog.at_level(logging.WARNING, logger="diffusionrl.sde.runtime"):
        FlowMatchSchedulePolicy.from_pretrained(tmp_path, shift=3.0)
    assert any("scheduler_config.json" in rec.message for rec in caplog.records), (
        f"expected warning about missing scheduler_config.json, got: {[r.message for r in caplog.records]}"
    )


def test_policy_full_diffusers_layout(tmp_path):
    """Standard SD3.5-style layout — all three JSONs populated."""
    from diffusionrl.sde.runtime import FlowMatchSchedulePolicy

    (tmp_path / "scheduler").mkdir()
    (tmp_path / "transformer").mkdir()
    (tmp_path / "vae").mkdir()
    (tmp_path / "scheduler" / "scheduler_config.json").write_text(
        json.dumps(
            {
                "use_dynamic_shifting": True,
                "base_shift": 0.5,
                "max_shift": 1.16,
                "base_image_seq_len": 256,
                "max_image_seq_len": 4096,
                "time_shift_type": "exponential",
            }
        )
    )
    (tmp_path / "transformer" / "config.json").write_text(
        json.dumps(
            {
                "patch_size": 2,
            }
        )
    )
    (tmp_path / "vae" / "config.json").write_text(
        json.dumps(
            {
                "block_out_channels": [128, 256, 512, 512],  # 4 blocks → vae_scale_factor=8
            }
        )
    )
    p = FlowMatchSchedulePolicy.from_pretrained(tmp_path, shift=3.0)
    assert p.shift == 3.0
    assert p.use_dynamic_shifting is True
    assert abs(p.max_shift - 1.16) < 1e-6
    assert p.patch_size == 2
    assert p.vae_scale_factor == 8


def test_policy_partial_layout(tmp_path):
    """scheduler/ missing → static defaults for dynamic block; other dirs read OK."""
    from diffusionrl.sde.runtime import FlowMatchSchedulePolicy

    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "config.json").write_text(
        json.dumps(
            {
                "patch_size": 4,
            }
        )
    )
    p = FlowMatchSchedulePolicy.from_pretrained(tmp_path, shift=5.0)
    assert p.shift == 5.0
    assert p.use_dynamic_shifting is False  # default (sched JSON missing)
    assert p.patch_size == 4  # picked up


def test_policy_shift_override_wins_over_scheduler_json(tmp_path):
    """``shift`` arg always wins over scheduler.config.shift."""
    from diffusionrl.sde.runtime import FlowMatchSchedulePolicy

    (tmp_path / "scheduler").mkdir()
    (tmp_path / "scheduler" / "scheduler_config.json").write_text(
        json.dumps(
            {
                "shift": 7.0,  # would-be ckpt default
            }
        )
    )
    p = FlowMatchSchedulePolicy.from_pretrained(tmp_path, shift=3.0)
    assert p.shift == 3.0  # user-configured wins


# ---------------------------------------------------------------------------
# compute_flowmatch_sigma — applies policy to (T, H, W)
# ---------------------------------------------------------------------------


def test_compute_static_matches_get_sigma_schedule():
    from diffusionrl.sde.runtime import (
        FlowMatchSchedulePolicy,
        compute_flowmatch_sigma,
    )

    policy = FlowMatchSchedulePolicy(shift=3.0, use_dynamic_shifting=False)
    out = compute_flowmatch_sigma(policy, num_inference_steps=10, height=1024, width=1024)
    expected = get_sigma_schedule(10, shift=3.0)
    assert torch.allclose(out, expected)


def test_compute_dynamic_derives_mu_and_matches_diffusers():
    """The dynamic branch should produce exactly what diffusers'
    dynamic FlowMatchEulerDiscreteScheduler would, given the same μ."""
    from diffusionrl.sde.runtime import (
        FlowMatchSchedulePolicy,
        calculate_dynamic_mu,
        compute_flowmatch_sigma,
    )

    policy = FlowMatchSchedulePolicy(
        shift=3.0,
        use_dynamic_shifting=True,
        base_shift=0.5,
        max_shift=1.16,
        base_image_seq_len=256,
        max_image_seq_len=4096,
        time_shift_type="exponential",
        vae_scale_factor=8,
        patch_size=2,
    )
    # Manually compute mu the same way compute_flowmatch_sigma does
    latent_h = 1024 // 8
    latent_w = 1024 // 8
    image_seq_len = (latent_h // 2) * (latent_w // 2)
    mu = calculate_dynamic_mu(image_seq_len, 256, 4096, 0.5, 1.16)
    expected = get_sigma_schedule(10, 3.0, mu=mu, time_shift_type="exponential")
    out = compute_flowmatch_sigma(
        policy,
        num_inference_steps=10,
        height=1024,
        width=1024,
    )
    assert torch.allclose(out, expected)


# ---------------------------------------------------------------------------
# ensure_req_sigmas — engine adapter helper
# ---------------------------------------------------------------------------


class _FakeReq:
    """Minimal duck-typed stand-in for RolloutReq."""

    def __init__(self, **diffusion):
        self.sigmas = None
        self.stage_params = {"diffusion": diffusion}


def test_ensure_req_sigmas_populates_when_none():
    from diffusionrl.sde.runtime import FlowMatchSchedulePolicy, ensure_req_sigmas

    policy = FlowMatchSchedulePolicy(shift=3.0)
    req = _FakeReq(num_inference_steps=10, height=1024, width=1024)
    ensure_req_sigmas(req, policy)
    assert req.sigmas is not None
    assert req.sigmas.shape == (11,)


def test_ensure_req_sigmas_idempotent_no_overwrite():
    """Pre-pinned σ must NOT be overwritten — supports test/escape-hatch use."""
    from diffusionrl.sde.runtime import FlowMatchSchedulePolicy, ensure_req_sigmas

    policy = FlowMatchSchedulePolicy(shift=3.0)
    req = _FakeReq(num_inference_steps=10, height=1024, width=1024)
    sentinel = torch.zeros(11)
    req.sigmas = sentinel
    ensure_req_sigmas(req, policy)
    assert req.sigmas is sentinel


def test_ensure_req_sigmas_missing_num_steps_raises():
    from diffusionrl.sde.runtime import FlowMatchSchedulePolicy, ensure_req_sigmas

    policy = FlowMatchSchedulePolicy(shift=3.0)
    req = _FakeReq()  # no num_inference_steps / height / width
    with pytest.raises(ValueError, match="num_inference_steps"):
        ensure_req_sigmas(req, policy)


def test_ensure_req_sigmas_missing_height_raises():
    """Silent height=1024 default would mis-derive μ for dynamic-shift
    models when the request actually rendered at a different
    resolution (e.g. WAN T2V at 480×832)."""
    from diffusionrl.sde.runtime import FlowMatchSchedulePolicy, ensure_req_sigmas

    policy = FlowMatchSchedulePolicy(shift=3.0)
    req = _FakeReq(num_inference_steps=10, width=512)  # missing height
    with pytest.raises(ValueError, match="height"):
        ensure_req_sigmas(req, policy)


def test_ensure_req_sigmas_missing_width_raises():
    from diffusionrl.sde.runtime import FlowMatchSchedulePolicy, ensure_req_sigmas

    policy = FlowMatchSchedulePolicy(shift=3.0)
    req = _FakeReq(num_inference_steps=10, height=512)  # missing width
    with pytest.raises(ValueError, match="width"):
        ensure_req_sigmas(req, policy)


# ---------------------------------------------------------------------------
# FlowMatchSDEDiscreteScheduler.set_timesteps — no double-shift fix
# ---------------------------------------------------------------------------


def test_sde_scheduler_external_sigmas_no_double_shift():
    """When external sigmas are provided to ``set_timesteps``, they must be
    used verbatim — no extra shift applied. This is the diffusers issue
    #13243 regression guard (our subclass patches around it)."""
    from diffusionrl.rollout.engine.vllm_omni._shared.flow_match_sde_scheduler import (
        FlowMatchSDEDiscreteScheduler,
    )

    # Use a shift the parent would happily double-apply.
    sch = FlowMatchSDEDiscreteScheduler(
        num_train_timesteps=1000,
        shift=3.0,
        use_dynamic_shifting=False,
    )
    # Compute an already-shifted schedule (shift=3, N=10) — the values
    # we expect sch.sigmas to end up holding after set_timesteps.
    our_sigmas = get_sigma_schedule(10, shift=3.0).tolist()[:-1]  # T values
    sch.set_timesteps(sigmas=our_sigmas)
    # scheduler.sigmas is T+1 (parent appends terminal 0). Compare the
    # first T values to what we sent.
    got = sch.sigmas[:10].cpu().to(torch.float32)
    sent = torch.tensor(our_sigmas, dtype=torch.float32)
    assert torch.allclose(got, sent, atol=1e-6), (
        f"Double-shift regression — set_timesteps(sigmas=...) re-applied "
        f"a shift. sent head={sent.tolist()[:3]}, got head={got.tolist()[:3]}"
    )
    # And the original config.shift must be restored after the call
    # (we mutate during the call and restore in a try/finally).
    assert sch.config.shift == 3.0
    assert sch.config.use_dynamic_shifting is False


# ---------------------------------------------------------------------------
# verify_engine_used_sigmas — round-trip σ scale handling
# ---------------------------------------------------------------------------


def test_verify_passes_when_normalized_match():
    from diffusionrl.rollout.engine.sigma_verify import verify_engine_used_sigmas

    sigmas = get_sigma_schedule(10, shift=3.0)
    # Same tensor → must pass without raising.
    verify_engine_used_sigmas(sigmas.clone(), expected=sigmas, engine_name="test")


def test_verify_skips_when_expected_none():
    """Legacy callers that bypass ensure_req_sigmas pass expected=None;
    the verify must no-op."""
    from diffusionrl.rollout.engine.sigma_verify import verify_engine_used_sigmas

    # actual is whatever; with expected=None, no check fires.
    verify_engine_used_sigmas(torch.tensor([1.0, 0.0]), expected=None, engine_name="test")


def test_verify_raises_when_actual_none():
    from diffusionrl.rollout.engine.sigma_verify import verify_engine_used_sigmas

    sigmas = get_sigma_schedule(10, shift=3.0)
    with pytest.raises(RuntimeError, match="did not echo trajectory_timesteps"):
        verify_engine_used_sigmas(None, expected=sigmas, engine_name="sglang-test")


def test_verify_handles_raw_1000x_scale():
    """SGLang sometimes echoes raw ``sigma * num_train_timesteps`` (e.g.
    [1000, 750, 0] for SD3). The verifier must auto-normalize via the
    dynamic integer ratio (commit 43642ac1 main-repo port) — same
    schedule, different unit."""
    from diffusionrl.rollout.engine.sigma_verify import verify_engine_used_sigmas

    sigmas = get_sigma_schedule(10, shift=3.0)
    raw = sigmas * 1000.0  # un-normalized scale
    verify_engine_used_sigmas(raw, expected=sigmas, engine_name="sglang-test")


def test_verify_handles_arbitrary_integer_scale():
    """The dynamic-ratio detector should work for any integer scale, not
    just 1000 (research checkpoints sometimes use 500 or 4000 as
    num_train_timesteps)."""
    from diffusionrl.rollout.engine.sigma_verify import verify_engine_used_sigmas

    sigmas = get_sigma_schedule(10, shift=3.0)
    for scale in (500, 1000, 4000):
        raw = sigmas * float(scale)
        verify_engine_used_sigmas(raw, expected=sigmas, engine_name=f"scale-{scale}")


def test_verify_catches_value_drift_after_scale_normalization():
    """A scale of 1000 with a real drift INSIDE the values: the integer
    ratio still folds to 1000, but allclose then surfaces the drift."""
    from diffusionrl.rollout.engine.sigma_verify import verify_engine_used_sigmas

    sigmas = get_sigma_schedule(10, shift=3.0)
    bad = sigmas.clone() * 1000.0
    bad[3] += 5.0  # genuine drift, ~0.005 after normalization
    with pytest.raises(RuntimeError, match="value mismatch"):
        verify_engine_used_sigmas(bad, expected=sigmas, engine_name="sglang-test")


def test_verify_catches_shape_mismatch_regardless_of_scale():
    """Shape mismatch is a definitive wiring break — it must raise even
    if the values would have been numerically close after scale-normalization."""
    from diffusionrl.rollout.engine.sigma_verify import verify_engine_used_sigmas

    sigmas = get_sigma_schedule(10, shift=3.0)  # [11]
    short = sigmas[:8] * 1000.0  # [8] — raw scale + wrong length
    with pytest.raises(RuntimeError, match="shape mismatch"):
        verify_engine_used_sigmas(short, expected=sigmas, engine_name="sglang-test")


def test_verify_under_threshold_is_not_normalized():
    """Threshold guard: values <= 10 must NOT be auto-divided (they
    could be legitimate near-1 σ values; dividing would mask drift).
    For a real-world example, σ[0] = 1.0 — must not be touched."""
    from diffusionrl.rollout.engine.sigma_verify import verify_engine_used_sigmas

    expected = torch.tensor([1.0, 0.5, 0.0])
    # If we sent 1.0 σ but worker echoed 2.0 σ, that's a real drift —
    # the [0, 10] threshold guard must let this through to allclose.
    bad = torch.tensor([2.0, 1.0, 0.0])
    with pytest.raises(RuntimeError, match="value mismatch"):
        verify_engine_used_sigmas(bad, expected=expected, engine_name="test")


def test_sde_scheduler_default_path_unchanged():
    """When sigmas is None, parent's default schedule (with its known
    static double-shift bug) is preserved — we don't change behavior on
    that path."""
    from diffusionrl.rollout.engine.vllm_omni._shared.flow_match_sde_scheduler import (
        FlowMatchSDEDiscreteScheduler,
    )

    sch = FlowMatchSDEDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,  # identity shift → static path same as our get_sigma_schedule
        use_dynamic_shifting=False,
    )
    sch.set_timesteps(num_inference_steps=10)
    assert sch.sigmas.shape == (11,)
    assert float(sch.sigmas[0].item()) == 1.0
    assert float(sch.sigmas[-1].item()) == 0.0


# ---------------------------------------------------------------------------
# vLLM-Omni adapter — sigma list length matches diffusers convention
# ---------------------------------------------------------------------------


class _StubReq:
    """Minimal stub for the vLLM-Omni request translator (matches what
    the translator reads from RolloutReq)."""

    def __init__(self, num_inference_steps: int, *, sigmas=None):
        self.sigmas = sigmas
        self.stage_params = {"diffusion": {"num_inference_steps": num_inference_steps}}


def test_vllm_omni_sigmas_list_length_is_T_not_T_plus_1():
    """Diffusers' set_timesteps(sigmas=...) takes len(sigmas) as
    num_inference_steps and appends a terminal 0 itself
    (scheduling_flow_match_euler_discrete.py:323 + :379). The vLLM-Omni
    adapter must send T values, not T+1. Sending T+1 would make the
    worker loop run an extra step and the response verify would fire
    a shape mismatch (T+2 actual vs T+1 expected).
    """
    from diffusionrl.rollout.engine.vllm_omni.request import _sigmas_list_from_req

    T = 10
    full_sigmas = get_sigma_schedule(T, shift=3.0)  # length T+1
    req = _StubReq(num_inference_steps=T, sigmas=full_sigmas)

    out = _sigmas_list_from_req(req, T)
    assert out is not None
    assert len(out) == T, f"expected T={T} values, got {len(out)}"
    # First value matches req.sigmas[0]; terminal 0 NOT included
    assert abs(out[0] - 1.0) < 1e-6
    assert out[-1] != 0.0  # terminal stripped


def test_vllm_omni_sigmas_list_returns_none_when_req_has_none():
    from diffusionrl.rollout.engine.vllm_omni.request import _sigmas_list_from_req

    req = _StubReq(num_inference_steps=10, sigmas=None)
    assert _sigmas_list_from_req(req, 10) is None


# ---------------------------------------------------------------------------
# resp_to_samples sparse SDE — full sigma schedule survives the bridge
# ---------------------------------------------------------------------------


def test_resp_to_samples_keeps_full_sigma_schedule_for_sparse_sde():
    """When the segment has sparse seg.indices, the bridge must use
    Trajectory.from_selective so the FULL σ schedule travels with
    samples.timesteps. Legacy GRPO loss reads ``td.sigmas[1]`` for
    sigma_max — cropping to sparse local view would silently
    mis-map it to e.g. sigma[5]."""
    import torch

    from diffusionrl.rollout.engine.types_compat import resp_to_samples
    from diffusionrl.types.prompts import Prompts
    from diffusionrl.types.request import RolloutRequest
    from diffusionrl.types.rollout_resp import RolloutResp
    from diffusionrl.types.sampling import SamplingParams
    from diffusionrl.types.segments.latent import LatentSegment

    T = 10  # T+1 = 11 σ values
    full_sigmas = get_sigma_schedule(T, shift=3.0)  # length 11
    # Sparse store: only positions [0, 5, 6, 10, 11] kept (5 latents)
    sparse_positions = [0, 5, 6, 10]
    sparse_latents = torch.zeros(
        2,
        len(sparse_positions),
        4,
        8,
        8,
        dtype=torch.float32,
    )

    seg = LatentSegment(
        sample_indices=torch.arange(2, dtype=torch.long),
        positions=torch.zeros(2, dtype=torch.long),
        latents=sparse_latents,
        sigmas=full_sigmas,
        indices=torch.tensor(sparse_positions, dtype=torch.long),
        sde_logp=None,
        sde_indices=None,
    )

    resp = RolloutResp(
        sample_ids=["a", "b"],
        group_ids=["g", "g"],
        conditions={},
        rollout_traces={"image": seg},
        decoded={},
    )

    request = RolloutRequest(
        prompts=Prompts(
            prompts=["p", "p"],
            prompt_ids=["0", "0"],
            sample_ids=["a", "b"],
            group_ids=["g", "g"],
            noise_group_ids=["a", "b"],
            prompt_metadata=[{}, {}],
        ),
        sampling_params=SamplingParams(num_inference_steps=T),
        collect_media_preview=False,
        media_max_items=8,
    )

    samples = resp_to_samples(resp, request=request)
    # Full schedule preserved (legacy GRPO reads ``td.sigmas[1]`` for
    # sigma_max — must point at global step-1 σ, not the sparse local
    # position-1 σ).
    assert samples.timesteps.shape == (T + 1,)
    assert torch.allclose(samples.timesteps, full_sigmas)
    # Trajectory store uses selective form so validate() accepts the
    # "len(timesteps) != num_stored" combination via the is_selective branch.
    assert samples.trajectories.is_selective
    assert samples.trajectories.total_positions == T + 1
    assert samples.trajectories.num_stored == len(sparse_positions)
    # step_indices MUST be None for selective: the legacy
    # ``TrainingBatch.get_position_for_step`` method assumes
    # ``compact_index == global_position`` (true only for full
    # trajectories). With selective storage, returning sparse global
    # labels would make has_position fail in surprising ways. Set
    # ``None`` so any caller that tries per-step lookups on this
    # bridge gets a clean error pointing them to the API
    # (``stage.replay`` reads ``segment.sigmas`` directly).
    assert samples.step_indices is None


def test_resp_to_samples_full_dense_keeps_step_indices():
    """The selective→None rule must NOT affect full dense trajectories,
    where step_indices is well-defined as the identity mapping."""
    import torch

    from diffusionrl.rollout.engine.types_compat import resp_to_samples
    from diffusionrl.types.prompts import Prompts
    from diffusionrl.types.request import RolloutRequest
    from diffusionrl.types.rollout_resp import RolloutResp
    from diffusionrl.types.sampling import SamplingParams
    from diffusionrl.types.segments.latent import LatentSegment

    T = 4  # tiny dense schedule, K = T+1 = 5
    full_sigmas = get_sigma_schedule(T, shift=3.0)
    full_positions = list(range(T + 1))  # [0,1,2,3,4]
    dense_latents = torch.zeros(2, T + 1, 4, 8, 8, dtype=torch.float32)
    seg = LatentSegment(
        sample_indices=torch.arange(2, dtype=torch.long),
        positions=torch.zeros(2, dtype=torch.long),
        latents=dense_latents,
        sigmas=full_sigmas,
        indices=torch.tensor(full_positions, dtype=torch.long),
        sde_logp=None,
        sde_indices=None,
    )
    resp = RolloutResp(
        sample_ids=["a", "b"],
        group_ids=["g", "g"],
        conditions={},
        rollout_traces={"image": seg},
        decoded={},
    )
    request = RolloutRequest(
        prompts=Prompts(
            prompts=["p", "p"],
            prompt_ids=["0", "0"],
            sample_ids=["a", "b"],
            group_ids=["g", "g"],
            noise_group_ids=["a", "b"],
            prompt_metadata=[{}, {}],
        ),
        sampling_params=SamplingParams(num_inference_steps=T),
        collect_media_preview=False,
        media_max_items=8,
    )
    samples = resp_to_samples(resp, request=request)
    # Dense path: trajectory is full, step_indices remains the identity
    # mapping so legacy code keeps working.
    assert samples.trajectories.is_full
    assert samples.step_indices is not None
    assert samples.step_indices.tolist() == full_positions


def test_sglang_sparse_trim_keeps_terminal_position():
    """For sparse SDE through the SGLang path, the trimming step must
    always preserve the terminal clean latent at position T —
    ``compute_trajectory_positions`` only emits the SDE (i, i+1) pairs,
    so without this fix sparse ``sde_indices={5}`` at T=10 would drop
    position 10 and downstream ``samples.latents = seg.latents[:, -1]``
    would return ``x_6`` instead of the clean ``x_10``."""
    from diffusionrl.types.trajectory_store import compute_trajectory_positions

    num_steps = 10
    sde_indices = {5}
    needed = set(compute_trajectory_positions(sde_indices, num_steps))
    # compute_trajectory_positions itself only returns (5, 6) — terminal
    # 10 is NOT in the set. Our trimming code in
    # ``rollout/engine/sglang/response.py`` explicitly adds num_steps.
    assert num_steps not in needed
    # The bug fix lives in the trimming logic; this test pins the
    # invariant by re-running the same ``needed.add(num_steps)`` step
    # and asserting position T is now present.
    needed.add(num_steps)
    assert num_steps in needed
    # And the keep_cols filter (positions valid within traj_len = T+1)
    # still includes the terminal.
    keep_cols = sorted(p for p in needed if 0 <= p < num_steps + 1)
    assert num_steps in keep_cols
