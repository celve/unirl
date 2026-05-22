# Model Artifact Layout

This folder contains Python model implementation packages such as `sd3/`,
`wan21/`, `wan22/`, `qwen_image/`, `hunyuan_video15/`, and
`hunyuan_image3/`.

## Directories

- `local/` — symlinked pretrained models (not committed)
- `local/reward_models/` — symlinked reward models (not committed)

## HuggingFace Fallback

Experiment YAMLs and model configs provide HuggingFace ID fallbacks through
Hydra env interpolation, for example
`${oc.env:PRETRAINED_MODEL,stabilityai/stable-diffusion-3.5-medium}`.

Do not commit large model weights (`.bin`, `.safetensors`, `.pt`, etc.).
