# Data Layout

## Directories

- `samples/` — committed toy data for smoke tests
  - `prompts_toy.json` — image prompts (SD3/FLUX scripts)
  - `video_prompts_toy.txt` — video prompts (HunyuanVideo scripts)
- `datasets/` — symlinks to real training datasets (not committed)

## Using real datasets

Override `DATA_PATH` to point to a real dataset:

```bash
DATA_PATH=data/datasets/hpdv2/train.json bash scripts/train_dancegrpo_sd3_separate.sh
```
