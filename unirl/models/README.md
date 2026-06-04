# Model Code Packages

This directory contains model implementation packages. It is separate from the
repository-root `models/` directory, which is only for local checkpoint and
reward-model symlinks.

## Package Layout

Current model packages:

- Diffusion: `sd3/`, `qwen_image/`, `wan21/`, `wan22/`, `hunyuan_video/`, `hunyuan_video15/`, `flux2_klein/`
- Autoregressive: `qwen3/`, `qwen_vl/`
- Mixed AR + diffusion: `hunyuan_image3/`
- Prompt-enhancer (composes other models): `pe/`

A few packages register under a different config name — e.g. `qwen3` registers
as `qwen3_v2` and `flux2_klein` as `flux2_klein_v2`.

Shared protocols and small data abstractions live under `types/`.

Most model packages follow this shape:

| File | Purpose |
|---|---|
| `config.py` | Hydra model config and checkpoint / precision / LoRA fields |
| `bundle.py` | materializes model stages and exposes named trainable stages |
| `diffusion.py` | diffusion stage used by rollout / replay (diffusion & mixed models) |
| `ar.py` | autoregressive stage (`ARStage` / `ARStep`) for AR & mixed models |
| `pipeline.py` | generation pipeline adapter |
| `conditions.py` | typed condition construction (a `Batch` subclass) |
| `text_embed.py` / `vae.py` / vision helpers | model-specific encoding and decoding helpers |

A package ships `diffusion.py`, `ar.py`, or both (HunyuanImage3 is mixed).

## Contracts

Model packages bridge config, rollout, and training:

- `cfg.model._target_` points to the selected pipeline factory such as
  `SD3Pipeline.from_config` or `HunyuanImage3Pipeline.from_config`;
- the pipeline exposes `pipeline.bundle`, and the bundle either loads weights
  eagerly or materializes them later on train workers;
- training wraps the model's trainable stage (resolved via the bundle's
  `trainable_module()`);
- rollout engines call model-specific pipelines or backend adapters;
- `config.lora_target_modules` defaults to `None`; LoRA targets come from the
  recipe YAML.

Keep model-specific logic inside its package. Cross-model typed contracts should
go under `unirl/types` or `unirl/models/types`.

## Adding a Model

1. Add a package under `unirl/models/<model_name>/`.
2. Define its config as a plain `@dataclass` (recipes reference it by `_target_`).
3. Implement a bundle that exposes the stages needed by training and rollout.
4. Add condition, text / vision, diffusion and/or AR, and VAE helpers as needed.
5. Add at least one recipe under `examples/<domain>/<model>/`.
6. Document required external checkpoints in the recipe YAML or launcher env docs.

See `.claude/skills/development/add-model-bundle/SKILL.md` for the full
bundle / pipeline / stage / conditions contract.
