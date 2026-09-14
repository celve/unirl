# experimental/leo2 — HunyuanVideo 2.0 (Leo2) MoE-A12B t2v FlowGRPO

Text-to-video FlowGRPO on a 75B three-stream MMDiT whose model code lives **outside this
repo** — the `hymm` package inside `hunyuan_multimodal_gen_ar`, reached by `sys.path`. One
FSDP2-sharded copy per DP rank does rollout, log-prob replay and the LoRA update through
the in-process (trainside) engine: no separate inference engine, no weight sync.

Incubating here rather than in `unirl/models/` because the external dependency is neither
installable nor pinnable, its checkpoint is torch-DCP (which core's `load_sharded` cannot
read, so the package carries its own `materialize()`), and it needs a runtime compat shim
against stock hymm. See `models/README.md` for every divergence and its retirement
condition.

**Owner:** Linyu Wu.

## Launch

A Ray cluster must be up (`ray start --head`), and `$HYMM` is a
`hunyuan_multimodal_gen_ar` checkout.

```bash
export LEO2_CKPT_DIR=/path/to/iter_0063300_torch/weights        # torch-DCP, ~150 GB
export LEO2_GENERATION_CONFIG=/path/to/leo2_genconfig_rl.json
export ASSETS_BASE=/path/to/hymm_ar_assets
export PYTHONPATH=$HYMM:$HYMM/deps/hy_parallelism:$HYMM/deps/IndexKits:$PYTHONPATH
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RAY_ADDRESS=auto python -m experimental.leo2.run \
  --config-name=leo2_t2v_trainside num_devices=8
```

hymm must be on `PYTHONPATH` for the **driver and reward actors too**, not only the
workers: rollout Samples carry `Leo2Conditions.hymm` blobs whose pickles reference hymm
classes, and the driver dies unpickling them otherwise. Set `HYMM_REPO_PATH` instead if you
would rather the bundle do the `sys.path` insert itself.

## Layout

| Path | What |
|---|---|
| `run.py` | Hydra entry point; delegates to core's `unirl.train_diffusion.run` — the trainer is unmodified |
| `models/` | the model package: `bundle.py` (hymm bootstrap, meta build, sharded DCP `materialize`), `text_embed.py` (capture-based conditioning), `diffusion.py` (`predict_noise` / `generate` / `replay`), `vae.py`, `pipeline.py`, `conditions.py`, `config.py`, `assets.py`, `compat.py` — mirrors `unirl/models/` (graduates into `unirl/models/leo2/`) |
| `examples/` | flat Hydra config, repo-wide schema — mirrors the top-level `examples/` |
| `datasets/` | the 32 train + 8 held-out prompts this recipe's numbers were measured on |

## The configuration that actually learns

The shipped recipe is a correctness smoke: at 10 steps and eta 0.7 it produces
structureless video while PickScore still reports ~0.73. To train, override:

```bash
RAY_ADDRESS=auto python -m experimental.leo2.run --config-name=leo2_t2v_trainside \
  num_devices=8 \
  sampling.num_inference_steps=30 sampling.eta=0.5 "sampling.sde_indices=[0,1,2,3,4]" \
  sampling.samples_per_prompt=8 sampling.init_same_noise=true \
  stack.num_updates_per_batch=2 backend.optimizer_cfg.learning_rate=2.5e-5 \
  adv_use_global_std=false \
  +reward.backend.config.frame_selection=uniform +reward.backend.config.num_score_frames=4
```

SDE noise only on the five highest-sigma transitions, the rest ODE; one shared `x_T` per
prompt group; PickScore averaged over 4 uniformly spaced frames. ~630 s/step (rollout 400 +
train 225), or ~495 s/step with `algorithm.old_logp_source=rollout` and
`bundle.config.text_encoder_gpu_transient=false`.

**Always eyeball rollout media next to the reward curve.** LoRA lr 1e-4 without KL
reward-hacks into stripe textures by ~rollout 50; 2.5e-5 stays clean, and
`algorithm.beta>0` (KL against the adapter-disabled base) is the intended fix.

## Verification

| Config | Hardware | Head | Status |
| --- | --- | --- | --- |
| `leo2_t2v_trainside` (8 rollouts, shipped defaults) | 8xH20 | pre-relocation (`6b51214a` lineage, eager load) | contributor run — 195–203 s/step, reward 0.727–0.733, `ratio` 1.0000, `gn` 0.0012–0.0026, peak 51.1–52.4 GB/GPU |
| sharded post-wrap DCP load, first-rollout parity against the eager path | 8xH20 | pre-relocation (`83609a1b` lineage) | contributor run — meta build 25.6 s + `to_empty` 7.9 s + read 73.4 s at 17.5 GB/rank, weights-ready ~107 s (was 22 min), first reward 0.7310 through both paths |
| v3 30-step recipe (2 rollouts) | 8xH20 | pre-relocation | contributor run — reward 0.7211, `gn` 0.0003, rollout-vs-replay \|Δlogp\| 8.6e-6 |
| FA3 varlen shim (both signatures, tuple return, drift fails closed, idempotency) | CPU, faked `sys.modules` | current head | PASS — 9/9 assertions |
| lint guards + ruff + `compileall` | CPU | current head | PASS |
| **anything on this branch's own head, on GPU** | 8xH20 | current head | **pending** — the relocation and the compat shim have never run against real weights |

The last row is the gate that matters. The first three were measured before this package was
relocated and before the FA3 shim existed; the numbers are the targets a run on this head
must reproduce, not evidence about this head.
