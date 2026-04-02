# ImageReward Installation

ImageReward is a BLIP-based (~300M) human preference reward model from THUDM. It outputs a single scalar score reflecting overall human preference (text-image alignment, aesthetics, composition, etc.).

- Paper: https://arxiv.org/abs/2304.05977
- Repo: https://github.com/THUDM/ImageReward

## 1. Install

```bash
pip install image-reward transformers==4.45.2 huggingface_hub==0.36.2 datasets==4.8.3 tokenizers==0.20.3
pip install git+https://github.com/openai/CLIP.git
```

> `image-reward` depends on OpenAI CLIP but does not declare it — the second line is required.
>
> Pinning `transformers`, `huggingface_hub`, `datasets`, `tokenizers` avoids cascading incompatibilities between `image-reward`, `transformers`, `datasets`, and `pyarrow`.

## 2. Model weights

On first call to `RM.load("ImageReward-v1.0")`, weights are auto-downloaded from HuggingFace to `~/.cache/`. For offline environments:

```bash
huggingface-cli download THUDM/ImageReward --local-dir /path/to/image_reward_ckpt
```

Expected files:

```
image_reward_ckpt/
├── ImageReward.pt       (~1.1 GB)
└── med_config.json      (BLIP medical config)
```

To load from a custom path:

```python
import ImageReward as RM
model = RM.load("/path/to/image_reward_ckpt/ImageReward.pt",
                med_config="/path/to/image_reward_ckpt/med_config.json")
```

## 3. GPU memory

| Model | Params | GPU Memory |
|-------|--------|-----------|
| ImageReward | ~300M | ~0.8 GB |
| HPSv3 | ~7B | ~14 GB |

## 4. Troubleshooting

> If you hit any of the issues below, the easiest fix is to re-run the pinned install command from [Section 1](#1-install).


### `timm` gets downgraded to 0.6.13

This is expected — `image-reward` pins `timm==0.6.13`. Currently no DiffusionRL component depends on timm, so this is safe to ignore.

## 5. Quick verification

```bash
python -c "
import ImageReward as RM
from PIL import Image

model = RM.load('ImageReward-v1.0', device='cpu')
img = Image.new('RGB', (512, 512), color=(135, 206, 235))
score = model.score('a beautiful blue sky', img)
print(f'Score: {score:.4f}')
print('OK')
"
```
