# Training Scripts

These shell scripts are the primary researcher entrypoints in this repo.

Use `scripts/*.sh` as the maintained experiment templates.
A local `configs/recipes/` directory may exist in some working trees, but it
is gitignored and should not be treated as the public repo interface.

All scripts now resolve paths relative to repository root:

- toy data: `data/samples/prompts_toy.json`
- OCR toy data: `data/samples/ocr_prompts_toy.json`
- toy video prompts: `data/samples/video_prompts_toy.txt`
- local model mount: `models/local/`
- outputs: `outputs/`

## Quick sanity run

```bash
DATA_PATH=data/samples/ocr_prompts_toy.json \
  bash scripts/train_flowgrpo_sd3_train_actor_sampling.sh \
  --rollout.num-rollout 1 \
  --rollout.save-steps 1000
```

Precision config:

- Use `precision.training.*` for training model load precision, FSDP param precision, and train-side autocast.
- Use `precision.rollout.*` for sampler/replay autocast plus trajectory/logprob storage precision.
- In dedicated SGLang rollout, prompt encoder precision also follows `precision.rollout.autocast_precision`.

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

## Typical usage

```bash
# SD3 / FlowGRPO (training-actor direct sampling)
bash scripts/train_flowgrpo_sd3_train_actor_sampling.sh

# SD3 / MixGRPO (SGLang, separate mode)
bash scripts/train_mixgrpo_sd3_sglang_separate.sh

# SD3 / NFT (SGLang, separate mode)
bash scripts/train_nft_sd3_sglang_separate.sh
```

## Engine note

For training-actor direct sampling, set `rollout.mode=direct_sampling`
and `sync.protocol=disabled`.
Leave `rollout.rollout_engine` unset in direct mode.
For SGLang, use `rollout.mode=separate` or `rollout.mode=colocate`
with `rollout.rollout_engine=sglang`, and set an explicit dedicated-rollout
weight-sync mode such as `tensor_payload`, `nccl_broadcast`, or
`checkpoint_path`.
Dedicated rollout modes also require `rollout.num_gpus_per_actor` to be set explicitly.

### SGLang remote scheduler mode (TCP, non-HTTP data plane)

Use `rollout.sglang_local_mode=false` and pass explicit scheduler endpoint(s)
through `rollout.sglang_kwargs`:

```yaml
rollout:
  mode: separate
  rollout_engine: sglang
  num_gpus_per_actor: 4
  tp_size: 4
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
DATA_PATH=/path/to/prompts.json PRETRAINED_MODEL=/path/to/model bash scripts/train_flowgrpo_sd3_train_actor_sampling.sh
```

All commands invoke canonical package entrypoint:

```bash
python -m diffusionrl.train ...
```
