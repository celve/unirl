# ImageReward Installation

ImageReward is a BLIP-based (~300M) human preference reward model from THUDM. It outputs a single scalar score reflecting overall human preference (text-image alignment, aesthetics, composition, etc.).

- Paper: https://arxiv.org/abs/2304.05977
- Repo: https://github.com/THUDM/ImageReward

## 1. Install image-reward package

```bash
pip install image-reward
```

### Missing dependency: OpenAI CLIP

`image-reward` internally imports `clip` (OpenAI's CLIP package) but does **not** declare it in its dependencies. You must install it manually:

```bash
pip install git+https://github.com/openai/CLIP.git
```

> **Note**: Unlike HPSv3, image-reward does NOT pin a specific transformers version, so no version conflicts expected.

## 2. Model weights

On first call to `RM.load("ImageReward-v1.0")`, the library auto-downloads weights from HuggingFace to `~/.cache/`. For offline environments, download manually:

```bash
huggingface-cli download THUDM/ImageReward --local-dir /path/to/image_reward_ckpt
```

Expected files:

```
image_reward_ckpt/
├── ImageReward.pt       (~1.1 GB)
└── med_config.json      (BLIP medical config)
```

Current DiffusionRL integration follows the same style as other built-in image
scorers: it calls `RM.load("ImageReward-v1.0")` and relies on the library's
default cache/download behavior. It does not currently expose a dedicated
DiffusionRL CLI flag for custom ImageReward checkpoint paths.

Equivalent raw library call:

```python
import ImageReward as RM
model = RM.load("/path/to/image_reward_ckpt/ImageReward.pt",
                med_config="/path/to/image_reward_ckpt/med_config.json")
```

## 3. GPU memory

ImageReward is a ~300M model, about **0.8 GB** in fp16. Negligible compared to HPSv3's ~14 GB.

| Model | Params | GPU Memory |
|-------|--------|-----------|
| ImageReward | ~300M | ~0.8 GB |
| HPSv3 | ~7B | ~14 GB |

## 4. Known issues

### Missing `clip` module

`image-reward` imports `clip` (from `ImageReward/models/AestheticScore.py`) but does not list it as a dependency. Without it you get:

```
ModuleNotFoundError: No module named 'clip'
```

Fix: `pip install git+https://github.com/openai/CLIP.git`

### timm downgrade

`image-reward` pins `timm==0.6.13` (its BLIP backbone uses old timm APIs). Installing it will downgrade timm from 1.x to 0.6.x:

```
- timm==1.0.16
+ timm==0.6.13
```

Currently no component in DiffusionRL imports timm, so this has **no impact**. If a future dependency requires timm 1.x, you may need to install image-reward with `--no-deps` and patch compatibility manually.

## 5. Quick verification

```bash
python -c "
import ImageReward as RM
from PIL import Image

model = RM.load('ImageReward-v1.0', device='cuda')
img = Image.new('RGB', (512, 512), color=(135, 206, 235))
ranking, rewards = model.inference_rank('a beautiful blue sky', [img])
print(f'Score: {rewards[0]:.4f}')  # expected: positive value
print('OK')
"
```
