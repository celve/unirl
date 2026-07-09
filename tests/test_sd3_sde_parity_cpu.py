"""Bitwise SDE-math parity: engine scheduler vs trainer strategy (CPU).

The vLLM-Omni rollout computes per-step SDE log-probs in
``FlowMatchSDEDiscreteScheduler`` (an in-repo reimplementation), while the
trainer replays them through ``unirl.sde.kernels.FlowSDEStrategy``. The two
must agree **bitwise** given identical network outputs — any formula or
dtype drift here lands directly in the GRPO ratio, amplified by ``1/(2σ²)``.

This test drives a full T-step rollout through the scheduler on random
tensors with a deterministic pseudo-"network", then replays every SDE
transition through the trainer strategy and asserts ``torch.equal`` on the
log-probs, plus bitwise Euler parity on the non-SDE steps. Runs on CPU —
no GPU, no vllm/vllm-omni import (the scheduler module only needs
torch + diffusers).

Run: ``pytest tests/test_sd3_sde_parity_cpu.py -q``
"""

from __future__ import annotations

import torch

from unirl.rollout.engine.vllm_omni.pipelines._shared.flow_match_sde_scheduler import (
    FlowMatchSDEDiscreteScheduler,
)
from unirl.sde.kernels import FlowSDEStrategy

# Schedule chosen to exercise the σ==1 clamp (index 0), interior steps, and a
# small-σ tail where the 1/(2σ²) amplification is strongest.
SIGMAS = [1.0, 0.85, 0.65, 0.45, 0.25, 0.1]
SDE_INDICES = [0, 2, 4]
ETA = 0.7
SHAPE = (2, 4, 8, 8)  # [B, C, H, W] — small but multi-dim so the mean-reduce matters
DTYPE = torch.bfloat16  # trajectory/model dtype in the SD3 recipes


def _velocity(x: torch.Tensor, step_index: int) -> torch.Tensor:
    """Deterministic stand-in for the DiT: same bf16 input -> same bf16 output.

    Rollout and replay call this with the same stored ``x_t``, mirroring the
    real system where the shared-kernel forward makes v_theta(x_t) identical
    on both sides. Only the SDE math is under test here.
    """
    return torch.tanh(x.float() * (0.3 + 0.1 * step_index)).to(x.dtype)


def _engine_rollout():
    sched = FlowMatchSDEDiscreteScheduler(eta=ETA)
    sched.set_timesteps(sigmas=list(SIGMAS))
    sched.arm(eta=ETA, sde_indices=list(SDE_INDICES))

    gen = torch.Generator().manual_seed(1234)
    x = torch.randn(*SHAPE, generator=gen, dtype=torch.float32).to(DTYPE)
    noise_gen = torch.Generator().manual_seed(4321)

    for i, t in enumerate(sched.timesteps):
        v = _velocity(x, i)
        x = sched.step(v, t, x, generator=noise_gen)[0]
        assert x.dtype == DTYPE

    latents = [sched._initial_latent] + list(sched._traj_latents)  # x_0 .. x_T
    return sched, latents


def test_sde_logp_bitwise_parity():
    sched, latents = _engine_rollout()
    assert sched._traj_sde_step_indices == SDE_INDICES
    assert len(sched._traj_log_probs) == len(SDE_INDICES)

    strategy = FlowSDEStrategy()
    sigmas = sched.sigmas  # fp32, length T+1 (terminal 0 appended by diffusers)
    sigma_max = sigmas[1]

    for k, i in enumerate(SDE_INDICES):
        sample = latents[i]
        prev_sample = latents[i + 1]
        v = _velocity(sample, i)
        _, log_prob, _ = strategy.denoise(
            noise_pred=v,
            sample=sample,
            sigma=sigmas[i],
            sigma_next=sigmas[i + 1],
            eta=ETA,
            prev_sample=prev_sample,
            sigma_max=sigma_max,
            step_index=i,
        )
        engine_logp = sched._traj_log_probs[k]
        assert log_prob is not None
        assert log_prob.shape == engine_logp.shape == (SHAPE[0],)
        assert torch.equal(log_prob.float(), engine_logp.float()), (
            f"SDE step {i}: replay logp != rollout logp; "
            f"max|Δ|={(log_prob.float() - engine_logp.float()).abs().max().item():.3e}"
        )


def test_euler_step_bitwise_parity():
    """Non-SDE steps: trainer's eta=0 Euler must land on the engine trajectory."""
    sched, latents = _engine_rollout()
    strategy = FlowSDEStrategy()
    sigmas = sched.sigmas
    sigma_max = sigmas[1]
    ode_indices = [i for i in range(len(SIGMAS)) if i not in set(SDE_INDICES)]
    assert ode_indices, "test schedule must contain at least one ODE step"

    for i in ode_indices:
        sample = latents[i]
        v = _velocity(sample, i)
        prev_out, log_prob, _ = strategy.denoise(
            noise_pred=v,
            sample=sample,
            sigma=sigmas[i],
            sigma_next=sigmas[i + 1],
            eta=0.0,
            prev_sample=None,
            sigma_max=sigma_max,
            step_index=i,
        )
        assert log_prob is None
        assert torch.equal(prev_out.to(DTYPE), latents[i + 1]), f"Euler step {i} diverged from engine trajectory"
