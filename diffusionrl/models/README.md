# Model Code Packages

This directory contains model implementation packages. It is separate from the
repository-root `models/` directory, which is only for local checkpoint and
reward-model symlinks.

## Package Layout

Current model packages:

- `sd3/`
- `qwen_image/`
- `wan21/`
- `wan22/`
- `hunyuan_video15/`
- `hunyuan_image3/`

Shared protocols and small data abstractions live under `types/`.

Most model packages follow this shape:

| File | Purpose |
|---|---|
| `config.py` | Hydra model config and checkpoint/precision/LoRA fields |
| `bundle.py` | materializes model stages and exposes named trainable stages |
| `diffusion.py` | diffusion stage implementation used by rollout/replay |
| `pipeline.py` | generation pipeline adapter |
| `conditions.py` | typed condition construction |
| `text_embed.py` / `vae.py` / vision helpers | model-specific encoding and decoding helpers |

## Contracts

Model packages bridge config, rollout, and training:

- `cfg.model._target_` points to the selected pipeline factory such as
  `SD3Pipeline.from_config` or `HunyuanImage3Pipeline.from_meta_config`;
- the pipeline exposes `pipeline.bundle`, and the bundle either loads weights
  eagerly or materializes them later on train actors;
- train-side policy composition wraps one stage named by `cfg.training.policy_source`;
- rollout engines call model-specific pipelines or backend adapters;
- model configs may provide default LoRA target modules when YAML omits them.

Keep model-specific logic inside its package. Cross-model typed contracts should
go under `diffusionrl/types` or `diffusionrl/models/types`.

## Adding a Model

1. Add a package under `diffusionrl/models/<model_name>/`.
2. Register its config with `@register_config(group="model", name="<model_name>", target=...)`.
3. Implement a bundle that exposes the stages needed by training and rollout.
4. Add condition, text/vision, diffusion, and VAE helpers as needed.
5. Add at least one experiment YAML under `conf/experiment/`.
6. Document required external checkpoints in the experiment YAML or launcher env docs.
