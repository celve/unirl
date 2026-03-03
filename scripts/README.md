# Training Scripts

These shell scripts are reproducible experiment templates.

All scripts now resolve paths relative to repository root:

- toy data: `data/samples/prompts_toy.json`
- toy video prompts: `data/samples/video_prompts_toy.txt`
- local model mount: `models/local/`
- outputs: `outputs/`

## Quick sanity run

```bash
bash scripts/train_dancegrpo_sd3_train_actor_sampling.sh --rollout.num-rollout 1 --rollout.save-steps 1000
```

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

## Plugin demo

```bash
# End-to-end plugin wiring example (algorithm/loss/reward/rollout pipeline)
bash scripts/train_plugin_demo.sh --rollout.num-rollout 1
```

## Engine note

`sampling.training_actor_direct_sampling=true` currently only supports `--sampling.sampler-engine-type fsdp`.
For SGLang, use dedicated rollout actors (`--ray.colocate-rollout-training true/false`)
with scripts named `*_sglang_colocate.sh` or `*_sglang_separate.sh`.

### SGLang remote scheduler mode (TCP, non-HTTP data plane)

Use `local_mode=false` and pass explicit scheduler endpoint(s) in `engine_kwargs`:

```bash
--sampling.engine-kwargs '{
  "local_mode": false,
  "remote_scheduler_endpoints": [
    "tcp://10.0.0.11:35555",
    "tcp://10.0.0.12:35555"
  ],
  "num_gpus": 4,
  "tp_size": 4,
  "rollout_transport_dtype": "bf16",
  "rollout_transport_drop_decoded_videos": true,
  "rollout_transport_log_payload_bytes": true
}'
```

Notes:
- When `remote_scheduler_endpoints` is set, rollout actors map by rank (`rank % len(endpoints)`).
- Rollout weight updates are deduplicated per logical scheduler endpoint (avoids repeated updates to the same scheduler).
- For SGLang rollout, `encode_prompt_in_generate` now defaults to `false` (can be re-enabled in `engine_kwargs`).

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
