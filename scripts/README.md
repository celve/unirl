# Training Scripts

These shell scripts are the primary researcher entrypoints in this repo.

Use `scripts/*.sh` as the maintained experiment templates.
If you prefer editing grouped YAML directly, use `scripts/example_*.yaml` as
auxiliary examples. A local `configs/recipes/` directory may exist in some
working trees, but it is gitignored and should not be treated as the public
repo interface.
Public config tests cover the committed YAMLs in this directory.

All scripts now resolve paths relative to repository root:

- toy data: `data/samples/prompts_toy.json`
- OCR toy data: `data/samples/ocr_prompts_toy.json`
- toy video prompts: `data/samples/video_prompts_toy.txt`
- local model mount: `models/local/`
- outputs: `outputs/`

## Quick sanity run

```bash
DATA_PATH=data/samples/ocr_prompts_toy.json \
  bash scripts/train_dancegrpo_sd3_train_actor_sampling.sh \
  --rollout.control.num-rollout 1 \
  --rollout.artifacts.save-steps 1000
```

Precision config:

- Use `precision.training.*` for training model load precision, FSDP param precision, and train-side autocast.
- Use `precision.rollout.*` for sampler/replay autocast plus trajectory/logprob storage precision.
- In dedicated SGLang rollout, prompt encoder precision also follows `precision.rollout.autocast_precision`.
- Keep `rollout.topology.service_transport_dtype` under `rollout.topology`; it is a transport setting, not part of `precision.*`.

Terminology:

- `rollout_id` is the outer rollout-train loop step.
- For experiment tracking it is close to a framework-level global step.
- It is not guaranteed to equal raw optimizer step count when gradient accumulation or inner epochs are used.

Eval data:

- `data_path` is the training prompt source.
- `eval_data_path` is optional; when unset, periodic eval reads a deterministic, unshuffled view of `data_path`.
- For a real validation split, set `eval_data_path` explicitly.
- Prompt datasets should provide text via `prompt` or `caption`; legacy embedding fields are no longer supported.
- Prompt datasets should ideally provide `prompt_id`; when omitted, the prompt loaders now synthesize deterministic IDs.

Group-reassembly rollout buffer:

- When `rollout.buffer.reassemble_by_group=true`,
  `rollout.buffer.group_size` must be set explicitly.
- This mode decomposes incoming rollout batches to sample locators and
  reassembles outgoing training batches by `group_id`.
- This mode is incompatible with sample-dropping built-in buffer filters
  (`rollout.buffer.drop_invalid=true` or reward-range filtering), because the
  producer contract requires complete groups.

## Typical usage

```bash
# SD3 / DanceGRPO
bash scripts/train_dancegrpo_sd3_train_actor_sampling.sh

# SD3 / DanceGRPO (SGLang, separate mode)
bash scripts/train_dancegrpo_sd3_sglang_separate.sh

# FLUX / MixGRPO
bash scripts/train_mixgrpo_flux_train_actor_sampling.sh

# FLUX / MixGRPO (SGLang, separate mode)
bash scripts/train_mixgrpo_flux_sglang_separate.sh

# Hunyuan / DanceGRPO
bash scripts/train_dancegrpo_hunyuan_train_actor_sampling.sh

# Hunyuan / DanceGRPO (SGLang, separate mode)
bash scripts/train_dancegrpo_hunyuan_sglang_separate.sh
```

## Auxiliary YAML examples

These committed YAMLs are small, reviewable snapshots mirrored from selected
local recipes and include the nested `precision.training` / `precision.rollout`
schema:

- `scripts/example_flux_dancegrpo_direct.yaml`
- `scripts/example_flux_dancegrpo_sglang_separate.yaml`
- `scripts/example_hunyuan_dancegrpo_direct.yaml`

Use them like this:

```bash
python -m diffusionrl.train --config scripts/example_flux_dancegrpo_direct.yaml
python -m diffusionrl.train_async --config scripts/example_flux_dancegrpo_sglang_separate.yaml
```

## Plugin demo

```bash
# End-to-end plugin wiring example (algorithm/reward)
bash scripts/train_plugin_demo.sh --rollout.control.num-rollout 1
```

## Engine note

For training-actor direct sampling, set `rollout.topology.mode=direct_sampling`
and `sync.protocol=disabled`.
Leave `rollout.topology.service_engine` unset in direct mode.
For SGLang, use `rollout.topology.mode=separate` or `rollout.topology.mode=colocate`
with `rollout.topology.service_engine=sglang`, and set an explicit dedicated-rollout
weight-sync mode such as `tensor_payload`, `nccl_broadcast`, or
`checkpoint_path`.
Dedicated rollout modes also require `rollout.topology.service_num_gpus` to be set explicitly.

### SGLang remote scheduler mode (TCP, non-HTTP data plane)

Use `rollout.topology.sglang_local_mode=false` and pass explicit scheduler endpoint(s)
through `rollout.topology.sglang_kwargs`:

```yaml
rollout:
  topology:
    mode: separate
    service_engine: sglang
    service_num_gpus: 4
    engine_tp_size: 4
    service_transport_dtype: bf16
    service_transport_drop_decoded_videos: true
    service_transport_log_payload_bytes: true
    sglang_local_mode: false
    sglang_kwargs:
      remote_scheduler_endpoints:
        - tcp://10.0.0.11:35555
        - tcp://10.0.0.12:35555
```

Notes:
- When `remote_scheduler_endpoints` is set, rollout actors map by rank (`rank % len(endpoints)`).
- Rollout weight updates are deduplicated per logical scheduler endpoint (avoids repeated updates to the same scheduler).
- SGLang rollout now encodes prompts inside `generate()` unconditionally so sampler outputs satisfy the rollout embedding contract without a driver-side fallback path.

## Local model setup

Models are loaded from `models/local/` (symlinks to `shared_models/`).
If local paths don't exist, scripts automatically fall back to HuggingFace
downloads (configured in `diffusionrl/config/arguments.py`).

Override default paths by environment variables per script:

```bash
DATA_PATH=/path/to/prompts.json PRETRAINED_MODEL=/path/to/model bash scripts/train_dancegrpo_sd3_train_actor_sampling.sh
```

All commands invoke canonical package entrypoint:

```bash
python -m diffusionrl.train ...
```
