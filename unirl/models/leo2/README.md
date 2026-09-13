# Leo2 (HunyuanVideo 2.0, MoE-A12B) — t2v trainside FlowGRPO

> **Where it fits:** one model package under [`unirl/models/`](../README.md), driving
> text-to-video FlowGRPO through the in-process rollout engine
> (`examples/diffusion/leo2/leo2_t2v_trainside.yaml`). One FSDP2-sharded copy per DP rank
> does rollout, log-prob replay and the LoRA update — no separate inference engine, no
> weight sync.

## What it is

A 75B three-stream MMDiT (`LeoLayer` / `LeoDualLayer` / `LeoTripleLayer`). **The model
implementation lives outside this repo** — the `hymm` package inside
`hunyuan_multimodal_gen_ar` — and is reached by `sys.path`, not vendored. This package is
only the UniRL adapters: bundle, pipeline, conditioning / diffusion / decode stages.

Two consequences shape everything here:

- **Conditioning is captured, not reimplemented.** `text_embed.py` swaps
  `model._diffusion_pipeline` for a recorder, calls hymm's own `generate_video`, and
  catches the kwargs at the diffusion-pipeline call boundary. Whatever the verified
  inference path feeds its denoising loop is exactly what the RL stage gets, so drift in
  hymm's template/tokenization/media-layout logic cannot silently change conditioning. It
  swaps the **private** attribute because `build_diffusion_pipeline()` guards on
  `_diffusion_pipeline is None` and the public name may be read-only.
- **Everything downstream is batch-1.** The captured hymm `model_kwargs` carries no batch
  axis, so `Leo2Conditions.hymm` is a per-sample list of opaque dicts and the recipe pins
  `forward_batch_size: 1` / `micro_batch_size: 1`.

**Sign convention — do not copy H3's minus sign.** Leo2's `FlowMatchDiscreteScheduler`
Euler step is `x_next = x + v * (sigma_next - sigma)`, which is exactly `FlowSDEStrategy`'s
eta=0 mean, so `diffusion_prediction` feeds `strategy.denoise` **un-negated**.

## Precision & loading

- **`uniform_bf16: true` casts the whole DiT, including the fp32 MoE router.** FSDP2
  asserts one original dtype per shard group and hymm stores `gate.wg` in fp32 inside
  every block, so a mixed block is rejected. The trade-off is a bf16 router; it is a
  deliberate smoke-recipe choice, not an oversight.
- **The router then needs `_patch_router_dtype`.** hymm computes the router in an
  autocast-disabled fp32 region (`hidden.float() @ gate.wg`); with `wg` now bf16 that
  matmul raises a dtype mismatch, so `wg.forward` is rebound to cast the weight to the
  input dtype at call time. Safe under FSDP — the weight is unsharded by then.
- **The checkpoint is torch-DCP, not safetensors** (`.metadata` + `FileSystemReader`), so
  `unirl/train/backend/sharded_load.py`'s `load_sharded` cannot read it. Checkpoint keys
  carry a `model.` prefix, handled by nesting the request dict one level.
- **Only the block classes are sharded and moved.** `FSDPBackend` wraps
  `block_class_names`; the patch/time embeds, projectors, final layers and RoPE buffers
  are left behind, so `_move_non_block_to_device` places them or the first forward hits a
  cpu/cuda mismatch. `hunyuan_image3` sidesteps this with an eager whole-transformer
  `.to(device)`, which is not possible at 150 GB.
- **`meta_init_transformer: true` is the measured path, and this package supplies
  `materialize()` rather than `_transformer_weights_path`.** `load_sharded` reads only
  `*.safetensors`, so the DCP read is ours. The order inside `materialize` is load-bearing:

      to_empty(device) -> restore_init_state -> apply_deferred_ops -> _dcp_load_sharded

  `apply_deferred_ops` (the LoRA A/B reset) must run **before** the load, because the load
  *skips* every `lora_A` / `lora_B` / `lora_embedding_` key and would otherwise leave
  `to_empty` garbage in them. The backend calls `apply_deferred_ops` again afterwards; that
  second call is a no-op because the queue is cleared.
- **The load iterates the live state dict, not the checkpoint.** peft exposes a base weight
  as `<module>.base_layer.<leaf>` while the checkpoint says `<leaf>`, so each live key is
  mapped back and a reverse map re-keys the write-back. Getting this wrong is not
  hypothetical: an earlier version loaded 1176 of 2130 tensors and left the rest garbage,
  which is why the load raises when either direction has more than a quarter as many
  unmatched keys as matched ones.
- **The bf16 cast under meta-init re-declares rather than copies.** `model.to(dtype)` would
  be a 150 GB host copy per rank of values that are garbage until the load fills them, so
  `_cast_params_dtype_no_copy` swaps in `torch.empty_like` tensors instead.
- **Aux models live in `model_dict`, a plain dict.** hymm keeps the text encoder and VAE
  there specifically so they are not registered submodules — FSDP, `to_empty` and the
  optimizer never see them.

## Environment

Nothing in this package has a site-specific default; a run supplies these.

| Variable | Recipe key | What |
|---|---|---|
| `LEO2_CKPT_DIR` | `ckpt_path` | torch-DCP weights directory (required) |
| `LEO2_GENERATION_CONFIG` | `generation_config_path` | hymm generation-config JSON (required) |
| `ASSETS_BASE` or `LEO2_ASSETS_BASE` | `assets_base` | hymm asset root (text encoder, VAE); required |
| `HYMM_REPO_PATH` | `hymm_repo_path` | only needed when `hymm` is not already importable |
| `LEO2_HYMM_CONFIG` | `config_yaml` | defaults to `LEO2_CONFIG_RELPATH` under the repo |
| `LEO2_PROMPT_FILE` / `LEO2_EVAL_PROMPT_FILE` | `data_path` / `eval_data_path` | default to `datasets/leo2_t2v/` |
| `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` | — | set to `1` by the bundle |

The validated launcher puts the hymm repo and its `deps/hy_parallelism` + `deps/IndexKits`
on `PYTHONPATH` for the **driver and reward actors too**, not just the workers: rollout
Samples carry `Leo2Conditions.hymm` blobs whose pickles reference hymm classes, and the
driver dies unpickling them otherwise.

**Two debug variables worth knowing:**

- `LEO2_DUMP_VIDEOS=<dir>` — dump a first/middle/last-frame contact sheet of decoded
  video 0, per rank, for the first few decodes.
- `LEO2_DEBUG_HYMM_SAMPLE=1` — additionally sample the same prompt through **hymm's own**
  `generate_video`, on the same FSDP-wrapped weights and the same VAE, and dump its frames
  alongside. If hymm's output is clean and the rollout's is not, the bug is in the UniRL
  stepping/conditioning path; if both are bad, it is in weights or wrapping. Drive it with
  `num_rollouts=1 sampling.eta=0.02` (≈ODE, so the two should agree closely).

## Gotchas

- **`root_wrap: true` is mandatory.** FSDP2 never reshards a root unit after forward, so
  without a root wrap every block becomes its own root and 48 unsharded layers accumulate
  into a 92.9 GB OOM.
- **`ensure_hy_parallel_state()` must run in the process that does the forward, and it is
  collective.** `hy_parallelism.parallel_states.get_parallel_state()` does not fail when
  uninitialised — it *constructs a fresh* `ParallelDims` → `build_mesh` →
  `init_device_mesh` on every call, leaking NCCL process groups and watchdog threads until
  `pthread_create` fails with "Resource temporarily unavailable" (>10k PGs within one
  forward). Because mesh construction is collective, every rank must reach it in the same
  order; the first `predict_noise()` of a DP rollout satisfies that, and the bundle
  re-checks there because `torch.distributed` may not be up at build time. This is a
  *different* object from `hymm.core.parallel_states.ParallelState`, which the bundle also
  sets (single-process, EP=1: UniRL's FSDP hosts the 64 experts locally).
- **`predict_noise` pins `model.eval()`.** `LeoModel.forward` is a predictor only when
  `not self.training`; in train mode it takes the SFT-loss path and returns `None`.
- **Never "clean" the hymm blob.** `prepare_inputs_for_generation` hard-dereferences
  `visual_mask`, `text_mask`, `timesteps_index` and the `*_token_indices` keys. A `None`
  value is fine; a *missing* key raises.
- **Geometry snaps to the training bucket.** hymm rounds duration to the ≥49-frame, 4k+1
  grid while `Leo2Pipeline.latent_shape` computes the exact value, so a 21-frame probe
  makes the capture and the latent disagree — it is an invalid smoke configuration.
- **Watch the media, not just the reward.** FlowSDE at 10 steps with eta 0.7 and noise on
  the last transition produced structureless video while PickScore still read ~0.73. LoRA
  lr 1e-4 without KL reward-hacks into stripe textures by ~rollout 50; 2.5e-5 stays clean.
- **`_report_foreign_types` exists for a reason.** `model_kwargs` entries whose class is
  neither builtin nor torch drag hymm onto the driver's and reward actors' `sys.path` when
  pickled; the one-time diagnostic lists what a pure-tensor blob would have to encode.

## Upstream patches

Each row is a divergence from stock hymm that this package installs at runtime, with the
condition that retires it.

| Patch | Where it installs | Delete it when |
|---|---|---|
| `gate.wg.forward` follows the input dtype | `bundle._patch_router_dtype`, under `uniform_bf16` | FSDP2 stops requiring one original dtype per shard group, or hymm stops computing the MoE router in an autocast-disabled fp32 region |
| whole-DiT bf16 cast (fp32 router lost) | `config.uniform_bf16` | same trigger as above |
| `ensure_hy_parallel_state()` | `bundle`, re-checked in `diffusion.predict_noise` | `hy_parallelism.parallel_states.get_parallel_state()` stops constructing a fresh `ParallelDims` when uninitialised |
| `sys.path` bootstrap of the hymm repo | `bundle._resolve_hymm_paths` | hymm ships as an installable distribution |
| assert `capture_init_state` is empty | `bundle.from_config`, under meta-init | core `restore_init_state` resolves post-wrap owners and raises on misses instead of skipping them — the AC/peft-unwrapping version on `personal/zuhaoding/leo2.0/dev` is the ready patch |
| `_dcp_load_sharded` re-implements core's `.base_layer.` remap and coverage check | `bundle._dcp_load_sharded` | `sharded_load` grows a DCP reader, at which point this package can stash `_transformer_weights_path` like every other bundle |
