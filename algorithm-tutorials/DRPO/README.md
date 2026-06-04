# DRPO — Divergence-regularized token-level RL

`ARDRPO` is the repo's **autoregressive (token-level)** RL algorithm for Qwen3-style
token rollouts. It replays the sampled tokens, compares train-side log-probs against
the rollout log-probs, expands sample-level advantages to tokens, and applies one of
three implemented losses. It is the LLM analogue of [flowDPPO](../flowDPPO/) — divergence
control instead of ratio clipping — on discrete tokens rather than a continuous SDE step.

- **Loss:** [`unirl/algorithms/drpo.py`](../../unirl/algorithms/drpo.py) (`ARDRPO`; `_ar_drpo_tv_loss`, `_ar_drpo_kl_loss`, `_ar_pg_tv_penalty_loss`)
- **Recipe (SGLang):** [`recipes/llm_rl/ar_drpo_qwen3_4b_base_dpao_sglang.yaml`](../../recipes/llm_rl/ar_drpo_qwen3_4b_base_dpao_sglang.yaml) — Qwen3-4B-Base on DAPO-Math
- **Config extract:** [`config.yaml`](config.yaml)
- **Checkpoints:** [🤗 zhouzhuoxin/unirl-checkpoint](https://huggingface.co/zhouzhuoxin/unirl-checkpoint/tree/main)
- **Paper:** *"Rethinking the Divergence Regularization in LLM RL."* Related: DPPO — [arXiv:2602.04879](https://arxiv.org/abs/2602.04879).

## Read this first: paper DRPO vs repo variants

The DRPO paper proposes a *smooth* Binary-TV regularizer. The `ARDRPO` class
hosts three variants — the class name is historical and broad, covering DRPO-family and
DPPO-family token trust regions:

| `variant` | What it implements | Shipped recipe? |
|---|---|---|
| `pg_tv_penalty` | Plain REINFORCE + a soft ratio-distance penalty: `−A·logp + penalty_coef·\|ratio−1\|` | **Yes** |
| `tv` | DPPO-style Binary-TV hard mask + truncated importance sampling | No |
| `kl` | DPPO-style Binary-KL hard mask + truncated importance sampling | No |

The canonical Qwen3 recipe uses `pg_tv_penalty`. It does **not** run the exact paper-DRPO
quadratic objective, and it does **not** use the `tv`/`kl` hard masks. Lineage:
**PPO → GRPO → SPO → DPPO → DRPO**.

## Why divergence instead of PPO ratio clipping

LLM RL is almost always off-policy: the rollout engine (SGLang) and the training engine
differ numerically, and one batch of rollouts is split into gradient steps, so the updated
policy `π` is not the behavior policy `µ` that sampled the tokens. PPO/GRPO build the trust
region from the ratio `r_t = π/µ`, but for a long-tailed vocabulary that is a poor proxy
for distributional shift: a rare token gives a huge ratio after a tiny probability change,
while a common token moves a lot of mass at a modest ratio. DPPO instead measures the
sampled token's **absolute probability shift** `|π−µ|` (the "Binary-TV" surrogate).

The reward here is **rule-based**: `MathBoxedRewardScorer`
([`unirl/reward/local/math_boxed.py`](../../unirl/reward/local/math_boxed.py)) checks the
`\boxed{}` answer against the DAPO-Math ground truth — exactly the verifiable reward the
paper trains on, not a learned reward model.

![drpo overview: the prompt to SGLang rollout (behavior policy mu) to group advantage to replay for new logp pi to ratio r and TV shift |pi - mu| to reweighted REINFORCE pipeline (single update), with the centerpiece contrast between DPPO's hard-mask step (full gradient, then a cliff to 0 at the threshold delta) and the paper's smooth, bounded DRPO weight that ramps down and crosses zero into a corrective region past delta.](assets/overview.png)

## Paper DRPO objective

For each sampled token, the ratio and signed Binary-TV shift are:

$$ r_t = \frac{\pi_\theta(y_t|s_t)}{\mu(y_t|s_t)} = \exp(\log\pi_\theta - \log\mu), \qquad \Delta_t = \pi_\theta(y_t|s_t) - \mu(y_t|s_t) $$

The paper keeps DPPO's Binary-TV trust region but replaces the hard mask with a smooth
quadratic regularizer (its Eq. 8):

$$ L_\text{DRPO}(x,\pi) = \mathbb{E}_{y\sim\mu}\!\left[\sum_t r_t \hat A_t - \frac{|\hat A_t|}{2\delta}\,\mu(y_t|s_t)\,(r_t - 1)^2\right] $$

whose gradient induces a continuous, **bounded** per-token weight (its Table 1):

$$ w_t = 1 - \mathrm{sign}\big(\hat A_t (r_t - 1)\big)\,\frac{|\pi_\theta(y_t|s_t) - \mu(y_t|s_t)|}{\delta} \;\in\; \Big[1 - \tfrac1\delta,\; 1 + \tfrac1\delta\Big] $$

For diverging updates the weight decays to 0 at the Binary-TV boundary and becomes
*corrective* beyond it; for converging updates it increases. **This exact smooth-weight
form is not wired as a variant in `drpo.py` today** — the sections below are what the code
actually runs.

## Repo variant: `pg_tv_penalty` (what the recipe runs)

Instead of the hard mask, add a smooth TV penalty to plain REINFORCE (no mask, no
importance weight). Its helper `_ar_pg_tv_penalty_loss`:

```python
log_diff = torch.clamp(new_logp - old_logp, min=-20.0, max=20.0)
ratio = torch.exp(log_diff)
tv_penalty = penalty_coef * torch.abs(ratio - 1.0)
pg_losses = -adv * new_logp + tv_penalty          # L_t = −A·logp + ε·|r−1|
```

This is plain token REINFORCE on `new_logp` plus a soft penalty pulling `ratio` toward 1
(same TV spirit as the paper, but the penalty form, not the exact weight). The recipe also
sets `loss_agg_mode: seq-mean-token-sum-norm`: per-token losses are summed per sequence,
divided by `horizon`, then averaged over sequences.

## Repo variants: `tv` and `kl` (DPPO hard mask)

The `tv`/`kl` variants keep a token unless its sampled-token divergence is over threshold
**in the advantage-improving direction** — an asymmetric, advantage-aware hard mask:

| advantage | keep the gradient iff | mask → 0 when |
|---|---|---|
| `Â_t > 0` | `Δ_t ≤ δ_high` | `π` rose too far (`Δ_t > δ_high`) |
| `Â_t ≤ 0` | `Δ_t ≥ −δ_low` | `π` fell too far (`Δ_t < −δ_low`) |

With the truncated ratio `r̄_t = min(r_t, C)` detached (TIS), the loss is masked REINFORCE,
`L_t = −Â_t·r̄_t·log π_θ·m_t`. Because `r̄_t` is detached, the gradient flows only through
`log π_θ`, giving the *same* policy gradient as PPO's surrogate `r_t·Â_t` (since
`∇(r·Â)=Â·r·∇logπ`) — the TIS clamp caps the importance weight without bending the
direction. Helper `_ar_drpo_tv_loss`:

```python
ratio = torch.exp(new_logp - old_logp)
truncated_ratio = torch.clamp(ratio, max=clip_ratio_c).detach()   # TIS
prob_delta = torch.exp(new_logp) - torch.exp(old_logp)            # π − µ
valid_mask = torch.where(adv > 0,
                         prob_delta <= clip_divergence_high,       # adv>0: block big prob ↑
                         prob_delta >= -clip_divergence_low)       # adv<0: block big prob ↓
pg_losses = -adv * truncated_ratio * new_logp * valid_mask.float()
```

The `kl` variant replaces `Δ_t` with a **Binary-KL** between Bernoulli distributions over
"chosen token vs. the rest" (computed from the chosen token's log-probs only):

$$ D^{\text{BinKL}}_t = \mu_t(\log\mu_t - \log\pi_t) + (1-\mu_t)\log\frac{1-\mu_t}{1-\pi_t} $$

Its mask also "always allows a conservative update": even at high KL, a token is kept if
the ratio moves *opposite* to the advantage (`ratio ≤ 1` for `Â>0`, `ratio ≥ 1` for `Â<0`).

## Math → code map

| Math object | Repo object |
|---|---|
| State `s_t = (x, y_{<t})` | `track.conditions` + the packed prefix in `TextSegment` |
| Sampled token `y_t` | `segment.tokens` |
| Behavior log-prob `log µ(y_t\|s_t)` | `segment.log_probs` (emitted by SGLang) → `old_logp` |
| New log-prob `log π_θ(y_t\|s_t)` | `stage.replay(segment, temperature=sampling_temperature)` → `new_logp` |
| Ratio `r_t` | `torch.exp(new_logp − old_logp)` |
| Chosen-token probs `π_t`, `µ_t` | `torch.exp(new_logp)`, `torch.exp(old_logp)` |
| Sample-level advantage `Â` | `track.advantages` |
| Token-level advantage | `ARGRPO._expand_advantages_to_tokens(advantages, segment.lengths, ...)` |
| Padding/eos mask | `segment.loss_mask` |
| `pg_tv_penalty` / `tv` / `kl` losses | `_ar_pg_tv_penalty_loss` / `_ar_drpo_tv_loss` / `_ar_drpo_kl_loss` |

## From rollout to update

1. `unirl.train_vlm` builds `VLMTrainer` for the text-only Qwen3 recipe.
2. `SGLangLLMRolloutEngine` samples completions and returns an `"ar"` track with packed
   `TextSegment.tokens`, `log_probs`, `lengths`, and masks.
3. `MathBoxedRewardScorer` scores each completion correct/incorrect.
4. `RolloutTrack.compute_advantages(normalize=False, scope="group")` mean-centers rewards
   within each prompt group; the recipe sets `normalize_adv_by_std: false`, so there is **no
   std division**.
5. `TrainStack.train_track` calls `ARDRPO.compute_loss_and_backward`, which replays the
   sampled tokens at `temperature=sampling_temperature`, reads `old_logp = segment.log_probs`,
   expands advantages to tokens, applies the selected variant, applies `segment.loss_mask`,
   reduces, and `backward()`s.

Unlike diffusion GRPO/DPPO, `ARDRPO` does **not** freeze a train-side `old_logp` in
`prepare_segment` — it reuses the rollout log-prob, so the recipe must keep
`stack.num_updates_per_batch: 1` (`TrainStack` raises otherwise:
`supports_multi_update = False`).

## Key knobs ([`config.yaml`](config.yaml))

| Knob | Meaning |
|---|---|
| `variant` | `pg_tv_penalty` (recipe), `tv`, or `kl`. |
| `penalty_coef` | `pg_tv_penalty` coefficient on `\|ratio − 1\|`. Default `0.15`. |
| `clip_divergence` | `tv`/`kl` threshold `δ` (TV: absolute prob shift; KL: nats). Ignored by `pg_tv_penalty`. |
| `clip_divergence_low` / `high` | Direction-specific thresholds for `adv<0` / `adv>0` in the hard-mask variants. |
| `clip_ratio_c` | TIS truncation bound `C` (hard-mask variants). Default `20.0`. Ignored by `pg_tv_penalty`. |
| `sampling_temperature` | **MUST equal `sampling.temperature`** (and the rollout engine's). Replay tempers logits so `π` and `µ` share a distribution (`ratio_mean ≈ 1`). |
| `loss_agg_mode` | `token-mean`, or the recipe's `seq-mean-token-sum-norm`. |
| `horizon` | Fixed normalizer for `seq-mean-token-sum-norm`; recipe `8192`. |
| `normalize_adv_by_std` | Recipe `false` → mean-center only. |

## Debug checklist

| Symptom | First files / variables to check |
|---|---|
| `ratio_mean` far from 1 at first update | `sampling_temperature` vs. `sampling.temperature` vs. rollout `temperature`; SGLang logprob config |
| Large `rollout_replay_logp_absdiff_mean` | train/rollout mismatch, weight sync, tokenization/chat-template mismatch |
| No gradient on many tokens | `segment.loss_mask`; zero advantages from all-correct/all-wrong groups |
| A variant knob seems ignored | check `variant` — `clip_divergence`/`clip_ratio_c` do nothing under `pg_tv_penalty` |
| Completion misses the boxed answer | chat template `/no_think` + `enable_thinking: false` (Qwen3 overruns on `<think>`) |
| SGLang serves stale weights | `LocalLoraWeightSync`, adapter name `default`, rollout wake/sleep logs |

Metric source: `ratio_*`, `valid_fraction`, `clipfrac_lower`, `tv_penalty_mean`, and the
AR-only `rollout_replay_logp_absdiff_mean` are emitted by the variant loss helpers.

## Run it

```bash
# one-time: build the local jsonl from the raw DAPO-Math + AIME datasets
python -m unirl.utils.prepare_dapo_math --out-dir data/dapo_math

DATA_PATH=data/dapo_math/train.jsonl EVAL_DATA_PATH=data/dapo_math/aime_eval.jsonl \
python -m unirl.train_vlm --config-name=llm_rl/ar_drpo_qwen3_4b_base_dpao_sglang num_devices=64
```

The model defaults to `Qwen/Qwen3-4B-Base`; set `QWEN3_PATH` to a local checkpoint dir to
avoid downloading at runtime. (Note the `train_vlm` entrypoint — the AR loss is
modality-agnostic and shares the VLM trainer.)

## vs. the other tutorials

- **[flowDPPO](../flowDPPO/)** is the closest conceptual sibling: both replace ratio
  clipping with divergence-aware control. flowDPPO has *exact* Gaussian KL over latent
  transitions; `ARDRPO` approximates token-distribution shift from chosen-token log-probs.
- **[flowGRPO](../flowGRPO/)** and `ARGRPO` are the ratio-clipped baselines; `ARGRPO`
  ([`unirl/algorithms/ar_grpo.py`](../../unirl/algorithms/ar_grpo.py)) shares
  `_grpo_clip_loss` with diffusion GRPO.
- **[diffusionNFT](../diffusionNFT/)** is not a likelihood-ratio method at all.

## References

- DRPO — *"Rethinking the Divergence Regularization in LLM RL"*: the smooth weight is its
  Table 1, the objective its Eq. 8.
- DPPO (the `tv`/`kl` hard mask): Qi et al., *"Rethinking the Trust Region in LLM
  Reinforcement Learning"* — [arXiv:2602.04879](https://arxiv.org/abs/2602.04879).
- SPO (the smooth-regularizer ancestor): Xie et al., *"Simple Policy Optimization"* —
  [arXiv:2401.16025](https://arxiv.org/abs/2401.16025).
