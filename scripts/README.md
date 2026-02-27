# Training Scripts

These shell scripts are reproducible experiment templates.

All scripts now resolve paths relative to repository root:

- toy data: `data/samples/prompts_toy.json`
- toy video prompts: `data/samples/video_prompts_toy.txt`
- local model mount: `models/local/`
- outputs: `outputs/`

## Quick sanity run

```bash
bash scripts/train_dancegrpo_sd3_train_actor_sampling.sh --num-rollout 1 --save-steps 1000
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

## Engine note

`training_actor_direct_sampling=true` currently only supports `--sampler-engine-type fsdp`.
For SGLang, use dedicated rollout actors (`--colocate-rollout-training true/false`)
with scripts named `*_sglang_colocate.sh` or `*_sglang_separate.sh`.

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
