# SDE Runtime

`diffusionrl.sde` owns the per-step diffusion math shared by rollout and
training:

- **σ schedule** — what noise level each inference index sees;
- **step kernels** — given a denoiser output, advance one step *and* (for SDE
  strategies) return the per-step log-probability that policy-gradient losses
  consume during replay;
- **initial noise** — deterministic per-rollout, per-group `x_T` generation
  so rollout and train-side replay sample matching trajectories.

This package does **not** decide which inference indices get SDE treatment.
That choice lives in `diffusionrl.algorithms.rollout_control` (driver-side
controller) backed by `diffusionrl.utils.scheduler_utils` (`AllSDEScheduler`,
`WindowScheduler`). Per-step math here; "which steps?" there.

## Key Files

| File | Owns |
|---|---|
| `kernels.py` | `StepStrategy` base, `SDEStrategy` mixin, and four concrete kernels: `FlowSDEStrategy`, `DanceSDEStrategy`, `CPSSDEStrategy`, `DPM2Strategy`. Each `step(...)` returns `(prev_sample, prev_sample_mean, std_var)`; SDE variants also expose log-prob computation. |
| `runtime.py` | `FlowMatchSchedulePolicy` (model-owned static **+** dynamic schedule config, incl. the `compute_mu` / `compute_sigma` methods — see the "σ Schedule Policy" section below), `get_sigma_schedule`, `ensure_req_sigmas` (pins σ onto each outbound `RolloutReq`). |
| `noise.py` | Deterministic seed mixing and `generate_latents` / `generate_shared_noise` for per-group noise sharing across rollouts. |
| `rules.py` | String-level `sde_type` normalization at engine wire boundaries (e.g. SGLang kwargs). Typed Python paths read `cfg.sampling.sde_strategy` directly. |

## Registered Strategies

Step strategies register under Hydra group `sampling/sde_strategy`:

| Recipe slug | Class | Used by |
|---|---|---|
| `flow` | `FlowSDEStrategy` | FlowGRPO, default for SD3 / Qwen-Image / WAN-style flow-matching recipes |
| `dance` | `DanceSDEStrategy` | DanceGRPO |
| `cps` | `CPSSDEStrategy` | CPS-style updates |
| `dpm2` | `DPM2Strategy` | Deterministic ODE solver — evaluation only; no log-prob, so no GRPO-style training |

Pick one with:

```yaml
defaults:
  - override /sampling/sde_strategy: flow   # or dance / cps / dpm2
```

`MixGRPO` does **not** introduce a new strategy. It composes `flow` with a
windowed index schedule selected through `algorithm.scheduler.timestep_strategy: window`.

## How a Rollout Step Uses a Strategy

Each rollout engine (and train-side replay) builds the strategy once and
calls it inside the inference loop. The shape is the same across backends:

```text
sigmas   = policy.compute_sigma(num_inference_steps=..., height=..., width=...)
strategy = build(cfg.sampling.sde_strategy)
sde_idx  = control.resolve_rollout_sde_indices(current_rollout_step)

for i in range(num_inference_steps):
    eps = model.predict_noise(x_t, sigmas[i])
    if i in sde_idx:
        x_t, log_prob_i, mean_i = strategy.denoise(
            eps, x_t, sigma=sigmas[i], sigma_next=sigmas[i+1], eta=...
        )
        segment.sde_indices.append(i)
        segment.sde_logp[i] = log_prob_i
    else:
        x_t, _, _ = strategy.step(eps, x_t, sigmas[i], sigmas[i+1])
```

Two-mode contract:

- **SDE indices** record sample / mean / std and a per-step log-prob — this is
  what `DiffusionGRPO` replays during training to form the importance ratio.
- **Non-SDE indices** are pure inference; they produce no log-prob and never
  contribute gradient.

`DPM2Strategy` is deterministic at every index. It is fine for sampling-only
evaluation but cannot be the trained strategy for any GRPO recipe.

## σ Schedule Policy

`FlowMatchSchedulePolicy` is loaded from the model checkpoint once at startup
(`from_pretrained` reads the `scheduler/transformer/vae` JSONs) and shared
across rollout and train actors. It carries:

- `shift` — **static** FlowMatch time-shift scalar; used only on the static
  branch (`use_dynamic_shifting=False`);
- `use_dynamic_shifting` — selects the static vs dynamic branch;
- `base_shift`, `max_shift`, `base_image_seq_len`, `max_image_seq_len`,
  `time_shift_type` — the dynamic-shift block;
- `vae_scale_factor`, `patch_size` — derive `image_seq_len` from
  `height x width`.

On the dynamic branch the per-request μ is computed by
`policy.compute_mu(image_seq_len, num_inference_steps)` — **the single
per-model override point**. The default delegates to `calculate_dynamic_mu`
(linear in `image_seq_len`); a model whose μ differs subclasses the policy and
overrides `compute_mu` (e.g. FLUX.2-klein's empirical μ, which also depends on
`num_inference_steps`). The schedule *application* (base grid + diffusers
time-shift) stays shared — only the μ value is model-specific.

`ensure_req_sigmas(req, policy)` pins σ onto each outbound `RolloutReq` so
every rollout actor (including dedicated backends that do not re-read the
checkpoint) samples on the same schedule the trainer will replay.

## Initial Noise

`noise.py` produces deterministic per-rollout latents. The driver mixes
`run.seed` with `rollout_id`, then derives one noise tensor per explicit
`noise_group_id`:

```text
per_rollout_seed = mix_rollout_base_seed(run.seed, rollout_id)
x_T = generate_shared_noise(..., base_seed=per_rollout_seed, noise_group_ids=...)
```

- `base_seed` from `run.seed` is mixed with `rollout_id` so different
  rollout steps draw different noise.
- All samples in one GRPO group share noise if `sampling.init_same_noise=True`;
  otherwise each sample draws an independent slice.
- When supported by the rollout engine, the driver ships this tensor through
  `RolloutReq.request_conditions["initial_latents"]` so rollout and train-side
  replay see aligned trajectories.

## What Lives Elsewhere

| Concern | Owner |
|---|---|
| Which inference indices run SDE on a given rollout step | `diffusionrl.algorithms.rollout_control.GRPORolloutControl.resolve_rollout_sde_indices(...)` |
| Window / progressive / random index scheduling (MixGRPO) | `diffusionrl.utils.scheduler_utils.WindowScheduler` |
| Mapping `algorithm.scheduler.*` YAML into a scheduler object | `diffusionrl.utils.scheduler_utils.create_indices_scheduler` |
| Per-model σ override (e.g. video models with custom shift) | `diffusionrl/models/<name>/diffusion.py` |
| Filtering trained steps after rollout (`skip_last_timestep`, `skip_initial_timesteps`) | `GRPORolloutControl.get_filtered_training_indices(...)` |

See `diffusionrl/algorithms/README.md` for the algorithm-side view of this
boundary.
