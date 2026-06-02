# diffusionNFT — Negative-aware Fine-Tuning (forward-process diffusion RL)

`DiffusionNFT` trains the policy **across the whole noise spectrum without running
an SDE rollout for the loss**. Instead of replaying a trajectory, it re-noises the
rollout's clean final latent at many timesteps and trains a *dual adapter* —
a trainable adapter vs. an EMA-frozen "old" adapter — toward a reward-weighted blend
of a **positive** and a **negative** prediction.

- **Code:** [`unirl/algorithms/nft.py`](../../unirl/algorithms/nft.py)
- **Recipe:** [`recipes/diffusion_rl/sd3_nft.yaml`](../../recipes/diffusion_rl/sd3_nft.yaml)
- **Config extract:** [`config.yaml`](config.yaml)

This is the most different tutorial. flowGRPO/flowDPPO ask "how likely was the
sampled reverse SDE step under the new policy?" NFT asks "given the clean latent
that the old policy produced, how should the model's forward-process denoising
prediction move for good vs. bad samples?" There is no old/new log-prob ratio in
the loss.

## Intuition

GRPO/DPPO need a stochastic rollout so per-step log-probs exist. NFT sidesteps that:
take the rollout's clean image latent `x_0`, re-noise it to `x_t` at a chosen
timestep, and ask the model to denoise it. In this repo, the scalar used by the
loss is not the raw reward; the rollout first computes a group-relative advantage
`A`, then NFT clips and remaps that advantage to a weight `r ∈ [0,1]`. High-`r`
samples are pulled toward a **positive** target (lean into the trainable adapter)
and low-`r` samples toward a **negative** target (lean away). Sweeping many
timesteps per micro-step updates the same clean sample across the rollout's noise
schedule.

Because the rollout only needs to produce a good `x_0`, NFT runs it under
**EMA-smoothed weights** in this implementation. The algorithm class declares
`requires_ema_rollout = True`, and the trainer swaps to the EMA/shadow adapter for
sampling, then restores the trainable adapter for the loss. That is the codebase's
off-policy variant of the DiffusionNFT paper's forward-process idea.

```mermaid
flowchart LR
    EMA["EMA rollout → clean latent x_0"] --> RN["re-noise:<br/>x_t = (1-t)·x_0 + t·ε"]
    R["advantage A → r = clip(A)/(2·c) + 0.5 ∈ [0,1]"] --> Blend
    RN --> New["trainable adapter → new_pred"]
    RN --> Old["EMA 'old' adapter → old_pred (detached)"]
    New --> Blend["positive / negative blend → reconstruct x̂_0"]
    Old --> Blend
    Blend --> Loss["r·MSE(x̂_0^+, x_0) + (1-r)·MSE(x̂_0^-, x_0)"]
    Loss --> Bk["backward (scaled 1/K over K timesteps)"]
```

## The math

Advantage → weight (advantages clipped to `±c = adv_clip_max`, linearly remapped):

$$ r = \mathrm{clip}\!\Big(\tfrac{\mathrm{clip}(A,-c,c)}{2c} + \tfrac12,\ 0,\ 1\Big) $$

For each of `K` timesteps `t`, with `x_t = (1-t)\,x_0 + t\,\varepsilon`, blend the
trainable prediction `new` and the EMA prediction `old` (β = `beta`):

$$ \text{pos} = \beta\,\text{new} + (1-\beta)\,\text{old}, \qquad \text{neg} = (1+\beta)\,\text{old} - \beta\,\text{new} $$

reconstruct `x̂_0 = x_t - t·pred` from each, and weight the two reconstruction MSEs
by `r`:

$$ \mathcal{L} = \mathbb{E}\Big[\, \tfrac{r}{\beta}\lVert \hat x_0^{+} - x_0\rVert^2 + \tfrac{1-r}{\beta}\lVert \hat x_0^{-} - x_0\rVert^2 \,\Big]\cdot c $$

In code (`unirl/algorithms/nft.py` · `_compute_loss_at_t`; `c = adv_clip_max`,
`mse` = per-sample mean over latent dims, casts elided):

```python
positive = beta * new_pred + (1 - beta) * old_pred    # β=1 → new_pred
negative = (1 + beta) * old_pred - beta * new_pred     # β=1 → 2·old_pred − new_pred
x0_pos = xt - t * positive                             # reconstruct x̂₀ from each branch
x0_neg = xt - t * negative
loss = (r * mse(x0_pos, x0) / beta + (1 - r) * mse(x0_neg, x0) / beta).mean() * c
```

With `use_adaptive_weight`, each per-sample MSE is divided by its mean-abs-error so
noise levels contribute on a comparable scale. The `K` timesteps come from the
rollout's own σ schedule (`train_timestep_mode: all`, terminal `t=0` dropped); each
contributes one backward scaled by `1/K`.

```mermaid
flowchart LR
    x0(["x_0<br/>one clean latent"]) -->|"re-noise → t1"| a["x_{t1}"]:::n
    x0 -->|"t2"| b["x_{t2}"]:::n
    x0 -->|"tK"| c["x_{tK}"]:::n
    a --> L["dual-adapter loss at each t<br/>backward ×1/K"]
    b --> L
    c --> L
    classDef n fill:#e8f0ff,stroke:#3b73c4,color:#000;
```

No trajectory replay: the rollout's single clean `x_0` is re-noised to each of the
`K` noise levels in its own σ schedule, and every level runs one dual-adapter
forward/backward scaled by `1/K` — so one micro-step trains the sample across the
whole noise spectrum.

The positive/negative names are easiest to read at `beta = 1.0`, the shipped
default: `pos = new`, while `neg = 2*old - new`. A high-advantage sample mostly
trains the current adapter to reconstruct `x_0`; a low-advantage sample mostly
trains against the mirrored negative branch.

## Code map

| Step | Where |
|---|---|
| Advantage → `r ∈ [0,1]` | `DiffusionNFT.compute_loss_and_backward` (clamp + linear remap) |
| Resolve the `K` timesteps | `DiffusionNFT._resolve_timesteps` (`all` ⇒ from `segment.sigmas`) |
| Re-noise, dual forward, blend, reconstruct, MSE | `DiffusionNFT._compute_loss_at_t` |
| Trainable vs. EMA "old" prediction | `stage.predict_noise_at_step(...)` + `nft_lora_policy.use_shadow()` |
| EMA handle wiring | resolved off `backend.ema` by the trainer (see `__init__`) |

## Key knobs ([`config.yaml`](config.yaml))

| Knob | Meaning |
|---|---|
| `beta` | Dual-blend coefficient. `1.0` ⇒ positive = new, negative = `2·old − new`. |
| `adv_clip_max` | Clip range `c` for the advantage→`r` remap (and the gradient-scale restore). |
| `use_adaptive_weight` | Normalize each MSE by its mean-abs-error (matches the original NFT recipe). |
| `train_timestep_mode` | `all` (rollout's σ schedule) or `random` (fresh uniforms per micro-step). |
| `training_timestep_fraction` | Fraction of the schedule kept after dropping terminal `t=0`. |
| `sampling.eta` / `num_sde_steps` | Both `0` — NFT does **not** record an SDE trajectory. |
| `kl_coef` | KL-to-base penalty — not implemented; any `> 0` fails fast. |

## Common pitfalls

- NFT still uses online rewards, but the loss itself is not a policy-gradient
  likelihood-ratio loss.
- `eta: 0.0` is intentional here. Turning on SDE log-probs does not make NFT more
  correct; it just changes the rollout path away from this recipe.
- The `r` in the loss is a clipped/remapped advantage, not the raw PickScore value.

## Run it

```bash
PRETRAINED_MODEL=stabilityai/stable-diffusion-3.5-medium \
python -m unirl.train_diffusion --config-name=diffusion_rl/sd3_nft num_devices=8
```

NFT requires the backend's EMA ("dual adapter"); the recipe wires it automatically.

## vs. the other tutorials

- **[flowGRPO](../flowGRPO/)** / **[flowDPPO](../flowDPPO/)** are on-policy and need
  the SDE rollout's per-step log-probs; NFT is off-policy and trains the forward
  process directly, so it has no ratio and no trajectory replay.

<!-- Figures: drop PNG/SVG into ./assets/ and reference e.g. ![](assets/nft_dual_adapter.png) -->
