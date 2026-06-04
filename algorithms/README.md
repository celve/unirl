# Algorithm Tutorials

Code-grounded walkthroughs of the core RL algorithms in this repo: three **diffusion**
(image) algorithms and one **LLM** (autoregressive, token-level) algorithm. The Python
package is still named `unirl`, so all code links point into `unirl/...`.

These tutorials do more than restate each paper. Each one answers three questions:

1. **What** mathematical objective does the algorithm optimize?
2. **Which** rollout fields carry the mathematical objects (advantage, log-prob, ...)?
3. **Where** — which trainer, algorithm class, stage method, and config knobs implement
   it, and what to watch when it misbehaves?

If you are new here, read this page first, then **[`FlowGRPO/`](FlowGRPO/)** — it
establishes the diffusion reverse-process vocabulary (SDE rollout, per-step log-prob,
old/new ratio, group-relative advantage). **[`FlowDPPO/`](FlowDPPO/)** changes only the
trust-region rule. **[`DiffusionNFT/`](DiffusionNFT/)** is the different one — it does not
optimize the reverse-trajectory likelihood at all. **[`DRPO/`](DRPO/)** is the LLM track
and reads independently.

| Tutorial | Domain | Main objective | Implementation | Canonical recipe |
|---|---|---|---|---|
| [`FlowGRPO/`](FlowGRPO/) | Diffusion / flow | PPO-style clipped ratio over sampled SDE transitions | [`unirl/algorithms/diffusion_grpo.py`](../unirl/algorithms/diffusion_grpo.py), loss in [`base.py`](../unirl/algorithms/base.py) | [`recipes/diffusion_rl/sd3_trainside.yaml`](../recipes/diffusion_rl/sd3_trainside.yaml) |
| [`FlowDPPO/`](FlowDPPO/) | Diffusion / flow | Unclipped `−A·ratio`, masked only when exact Gaussian KL is high **and** the update is over-aggressive | [`unirl/algorithms/dppo.py`](../unirl/algorithms/dppo.py) | [`recipes/diffusion_rl/sd3_flowdppo.yaml`](../recipes/diffusion_rl/sd3_flowdppo.yaml) |
| [`DiffusionNFT/`](DiffusionNFT/) | Diffusion / flow | Forward-process positive/negative reconstruction, weighted by reward-derived optimality | [`unirl/algorithms/nft.py`](../unirl/algorithms/nft.py) | [`recipes/diffusion_rl/sd3_nft.yaml`](../recipes/diffusion_rl/sd3_nft.yaml) |
| [`DRPO/`](DRPO/) | LLM / AR | Token-level importance-weighted PG + smooth advantage-weighted Binary-TV quadratic regularizer | [`unirl/algorithms/drpo.py`](../unirl/algorithms/drpo.py) | [`recipes/llm_rl/ar_drpo_qwen3_4b_base_dpao_sglang.yaml`](../recipes/llm_rl/ar_drpo_qwen3_4b_base_dpao_sglang.yaml) |

## Shared execution chain

All four are `StageAlgorithm` subclasses
([`unirl/algorithms/base.py`](../unirl/algorithms/base.py)) driven by the same trainer
loop. One rollout→train iteration:

1. A trainer builds a `RolloutReq` from prompts + sampling config.
2. A rollout engine returns `RolloutResp.tracks[name]` (`conditions`, `segment`).
3. `RewardService.score_and_attach(req=, track=)` writes `track.rewards`.
4. `RolloutTrack.compute_advantages(...)` writes `track.advantages`.
5. `TrainStack.train_track(track)` calls `algorithm.prepare_segment(...)` **once**
   (materializes any pre-update anchor, e.g. a frozen `old_logp`).
6. The stack slices micro-batches and calls
   `StageAlgorithm.compute_loss_and_backward(...)` per micro-batch.
7. The algorithm replays/forwards the stage at current weights, computes its loss, and
   `backward()`s; the FSDP backend steps the optimizer.

Only **steps 6–7** differ between algorithms. Given the per-sample advantage `A`:

| Algorithm | Loss on `A` | Records at rollout |
|---|---|---|
| flowGRPO | `−min(ρA, clip(ρ,1±ε)A)` over the trained SDE steps | `sde_logp`, `sde_indices` |
| flowDPPO | `−ρA`, **zeroed** when per-step Gaussian KL is high **and** over-aggressive | `sde_logp`, `sde_means`, `sigmas` |
| diffusionNFT | remap `A → r ∈ [0,1]`, positive/negative reconstruction MSE (no `ρ`) | `latents` (clean `x_0`), `sigmas` |
| DRPO (LLM) | `−Â·r + \|Â\|·µ·(r−1)²/(2ε)`: smooth quadratic regularizer (paper Eq. 8) | `tokens`, `log_probs`, `loss_mask` |

- **flowGRPO** / **flowDPPO** are reverse-process policy-gradient methods: they replay the
  SDE trajectory the rollout sampled and compare new vs. old per-step log-probs. GRPO
  **clips** the ratio; DPPO **masks** high-KL updates already moving too aggressively in
  the reward-improving direction.
- **diffusionNFT** is *off-policy* (`requires_ema_rollout = True`): no SDE rollout for the
  loss. It re-noises the clean latent at many timesteps and trains a dual adapter
  (trainable vs. EMA-frozen) toward a reward-weighted positive/negative blend.
- **DRPO** is the LLM analogue of flowDPPO: it replaces ratio-based clipping with a smooth
  Binary-TV-aligned quadratic regularizer. The per-token loss is
  `−Â·r + |Â|·µ·(r−1)²/(2ε)` (paper Eq. 8), whose gradient induces a bounded, continuous
  weight that attenuates diverging updates and provides corrective signals beyond the
  trust-region boundary — see [`DRPO/`](DRPO/).

## Core files

| Concept | Code |
|---|---|
| Per-track container: `conditions`, `segment`, `rewards`, `advantages` | [`unirl/types/rollout_resp.py`](../unirl/types/rollout_resp.py) |
| Diffusion rollout trajectory fields | [`unirl/types/segments/latent.py`](../unirl/types/segments/latent.py) |
| AR packed-token fields | [`unirl/types/segments/text.py`](../unirl/types/segments/text.py) |
| Algorithm base contract + shared `_grpo_clip_loss` | [`unirl/algorithms/base.py`](../unirl/algorithms/base.py) |
| Optimizer loop + `prepare_segment` timing + multi-update gate | [`unirl/train/stack.py`](../unirl/train/stack.py) |
| Diffusion trainer (reward→advantage→train) | [`unirl/trainer/diffusion.py`](../unirl/trainer/diffusion.py) |
| LLM/VLM trainer (reward→advantage→train) | [`unirl/trainer/vlm.py`](../unirl/trainer/vlm.py) |
| SDE transition math + per-step log-prob | [`unirl/sde/kernels.py`](../unirl/sde/kernels.py) |

## Data structures to keep in mind

The trainable rollout data for diffusion is a `LatentSegment`:

| Field | Meaning in the math | Used by |
|---|---|---|
| `segment.latents` | Latent trajectory; `latents[:, -1]` is the clean `x_0`. | NFT (and GRPO/DPPO replay) |
| `segment.sigmas` | σ/time schedule; `sigmas[i]`, `sigmas[i+1]` give `dt` and the SDE variance. | GRPO, DPPO, NFT |
| `segment.sde_indices` | Which denoising steps are stochastic SDE steps (the trainable ones). | GRPO, DPPO |
| `segment.sde_logp` | Old-policy log-prob of the sampled SDE transition, aligned with `sde_indices`. | GRPO, DPPO |
| `segment.sde_means` | Old Gaussian transition means `µ_old`, captured by `DiffusionDPPO.prepare_segment`. | DPPO |

The trainable rollout data for AR is a packed `TextSegment`:

| Field | Meaning in the math | Used by |
|---|---|---|
| `segment.tokens` | Sampled response tokens. | DRPO |
| `segment.log_probs` | Rollout/behavior log-probs `log µ(y_t \| s_t)`. | DRPO |
| `segment.lengths` | Per-sample span lengths, used to repeat advantages over token spans. | DRPO |
| `segment.loss_mask` | Token mask for padding/eos exclusions. | DRPO |

## Algorithm differences

| Question | flowGRPO | flowDPPO | diffusionNFT | DRPO |
|---|---|---|---|---|
| What is the action? | Next latent on a selected SDE step | Same latent transition | No reverse-process action; train on re-noised `x_0` | Next sampled token |
| Needs rollout log-prob? | Yes (`sde_logp`) | Yes, plus `sde_means` | No | Yes (`log_probs`) |
| New probability via replay? | `stage.replay(...).log_probs` | `.log_probs` + `.prev_sample_means` | No log-prob; `predict_noise_at_step` | `stage.replay(..., temperature=...)` |
| Old-policy anchor | Frozen in `prepare_segment` | Frozen `old_logp` + `old_means` | EMA/shadow adapter | Rollout log-prob only (single update) |
| Advantage use | broadcast `[B]→[B, S]` | broadcast `[B]→[B, S]` | clip + remap to `r ∈ [0,1]` | expand `[B]→[total_tokens]` |
| Trust region | PPO clip range | Exact Gaussian-KL mask | EMA old + positive/negative target | Smooth Binary-TV quadratic regularizer (bounded weight `w_t`) |
| `num_updates_per_batch > 1`? | Yes | Yes | No validated path | No (`TrainStack` raises) |

## Shared advantage normalization

All four center each reward on its **prompt group's mean** (the group is the critic-free
baseline) through [`RolloutTrack.compute_advantages`](../unirl/types/rollout_resp.py). They
differ in the **std**, via two different trainer knobs feeding the same method:

- **Diffusion** (`unirl/trainer/diffusion.py`) reads `adv_use_global_std`. The code
  *default* (flag absent) is per-prompt std, but **every shipped diffusion recipe sets
  `adv_use_global_std: true`** (PR #239): per-group mean, then divide by **one batch-wide**
  std — `(r − group_mean) / (batch_std + ε)` — for v1 parity. Implemented as
  `compute_advantages(normalize=True, use_global_std=True)`.
- **LLM/AR** (`unirl/trainer/vlm.py`) reads `adv_normalization_scope` +
  `normalize_adv_by_std`. The DRPO recipe sets `scope: group`, `normalize_adv_by_std:
  false`, so it mean-centers within each prompt group with **no std division** (`r −
  group_mean`).

## Papers

- **Flow-GRPO** — *"Flow-GRPO: Training Flow Matching Models via Online RL"*, Liu et al.,
  NeurIPS 2026 ([arXiv:2505.05470](https://arxiv.org/abs/2505.05470)). ODE-to-SDE
  conversion + denoising reduction.
- **Flow-DPPO** — *"Flow-DPPO: Divergence Proximal Policy Optimization for Flow Matching
  Models."* Replaces noisy ratio clipping with an exact equal-covariance Gaussian KL mask.
- **DiffusionNFT** — *"DiffusionNFT: Online Diffusion Reinforcement with Forward Process"*,
  Zheng et al. ([arXiv:2509.16117](https://arxiv.org/abs/2509.16117); ICLR 2026 Oral).
  Optimizes the forward process; needs only clean images + rewards.
- **DRPO** — *"Rethinking the Divergence Regularization in LLM RL."* Replaces DPPO's hard
  Binary-TV mask with a smooth advantage-weighted quadratic regularizer (objective Eq. 8,
  gradient weight Table 1). The recipe implements the exact paper method:
  `L_t = −Â·r + |Â|·µ·(r−1)²/(2ε)` with `ε = 12.5`.
- **DPPO** (the hard-mask predecessor) — Qi et al., *"Rethinking the Trust Region in LLM
  Reinforcement Learning"* ([arXiv:2602.04879](https://arxiv.org/abs/2602.04879)).
- **SPO** (the smooth-regularizer ancestor in the lineage) — Xie et al., *"Simple Policy
  Optimization"* ([arXiv:2401.16025](https://arxiv.org/abs/2401.16025)).

## Running a tutorial recipe

The `config.yaml` in each folder is an annotated **extract** for reading. To train, launch
the full canonical recipe:

```bash
# diffusion (flowGRPO / flowDPPO / diffusionNFT)
PRETRAINED_MODEL=stabilityai/stable-diffusion-3.5-medium \
python -m unirl.train_diffusion --config-name=diffusion_rl/sd3_trainside num_devices=8

# LLM (DRPO) — note the train_vlm entrypoint
python -m unirl.utils.prepare_dapo_math --out-dir data/dapo_math
DATA_PATH=data/dapo_math/train.jsonl EVAL_DATA_PATH=data/dapo_math/aime_eval.jsonl \
python -m unirl.train_vlm --config-name=llm_rl/ar_drpo_qwen3_4b_base_dpao_sglang num_devices=64
```

Each folder embeds a raster overview/curve from its `assets/` directory; add figures with
`![caption](assets/your_figure.png)`.
