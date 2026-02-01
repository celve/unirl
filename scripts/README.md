# Training Scripts

These shell scripts are reproducible experiment templates.

All scripts now resolve paths relative to repository root:

- toy data: `data/samples/prompts_toy.json`
- toy video prompts: `data/samples/video_prompts_toy.txt`
- local model mount: `models/local/`
- outputs: `outputs/`

## Quick sanity run

```bash
bash scripts/test_ray_single_gpu.sh separate --num-rollout 1
```

## Typical usage

```bash
# SD3 / GRPO (separate mode)
bash scripts/train_dancegrpo_sd3_separate.sh

# FLUX / MixGRPO (separate mode)
bash scripts/train_mixgrpo_flux_separate.sh

# Hunyuan (FSDP separate mode)
bash scripts/train_dancegrpo_hunyuan_separate.sh
```

## Local model setup

Models are loaded from `models/local/` (symlinks to `shared_models/`).
If local paths don't exist, scripts automatically fall back to HuggingFace
downloads (configured in `diffusionrl/config/arguments.py`).

Override default paths by environment variables per script:

```bash
DATA_PATH=/path/to/prompts.json PRETRAINED_MODEL=/path/to/model bash scripts/train_dancegrpo_sd3_separate.sh
```

All commands invoke canonical package entrypoint:

```bash
python -m diffusionrl.train ...
```
