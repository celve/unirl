# Model Artifact Layout

This folder also contains Python model implementation code (`flux.py`, `sd3.py`, ...).

## Directories

- `local/` — symlinked pretrained models (not committed)
- `local/reward_models/` — symlinked reward models (not committed)

## HuggingFace Fallback

If local model paths don't exist, training scripts automatically fall back to
HuggingFace model IDs (configured in `diffusionrl/config/arguments.py`).

Do not commit large model weights (`.bin`, `.safetensors`, `.pt`, etc.).
