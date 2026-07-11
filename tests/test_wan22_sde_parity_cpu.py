"""Bitwise SDE-math parity, wan22 flavor: engine scheduler vs trainer strategy (CPU).

Same contract as ``tests/test_sd3_sde_parity_cpu.py`` (which documents the
rationale), exercised at the wan22 parity recipe's operating point instead:

- σ grid: the wan static shift-5.0 flow-match schedule over 20 steps
  (``shift*t / (1 + (shift-1)*t)``, ``t = linspace(1, 0, 21)`` — identical
  formula in the engine's ``scheduling_wan_euler.py`` and unirl's
  ``get_sigma_schedule`` static branch);
- SDE window ``timestep_fraction [0, 0.2]`` → steps 0..3 run the SDE branch;
- **fp32 trajectory dtype**: the wan engine's denoise loop carries fp32
  master latents (SD3 carried bf16) — this exercises the scheduler's
  original-dtype roundtrip as a no-op;
- 5-D video-latent shape ``[B, C, T, H, W]``.

Run: ``pytest tests/test_wan22_sde_parity_cpu.py -q``
"""

from __future__ import annotations

import torch

from unirl.rollout.engine.vllm_omni.pipelines._shared.flow_match_sde_scheduler import (
    FlowMatchSDEDiscreteScheduler,
)
from unirl.sde.kernels import FlowSDEStrategy

_SHIFT = 5.0
_NUM_STEPS = 20
_T = torch.linspace(1.0, 0.0, _NUM_STEPS + 1, dtype=torch.float32)
SIGMAS = ((_SHIFT * _T) / (1.0 + (_SHIFT - 1.0) * _T)).tolist()[:-1]  # T values; terminal 0 appended by diffusers
SDE_INDICES = [0, 1, 2, 3]  # timestep_fraction [0, 0.2] of 20 steps
ETA = 0.7
SHAPE = (1, 16, 1, 8, 8)  # [B, C, T_lat, H, W] — wan 5-D video latent
DTYPE = torch.float32  # wan trajectory/master-latent dtype (SD3 used bf16)


def _velocity(x: torch.Tensor, step_index: int) -> torch.Tensor:
    """Deterministic stand-in for the DiT (see the SD3 test's rationale)."""
    return torch.tanh(x.float() * (0.3 + 0.1 * step_index)).to(x.dtype)


def _engine_rollout():
    sched = FlowMatchSDEDiscreteScheduler(num_train_timesteps=1000, shift=_SHIFT, eta=ETA)
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


def test_wan_sde_logp_bitwise_parity():
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


def test_wan_euler_step_bitwise_parity():
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
