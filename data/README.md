# Data Layout

## Directories

- `samples/` — committed toy data for smoke tests
  - `prompts_toy.json` — generic image prompts (alignment / non-OCR smoke)
  - `ocr_prompts_toy.json` — quoted text prompts for OCR-reward smoke tests
  - `video_prompts_toy.txt` — video prompts (HunyuanVideo scripts)
- `datasets/` — symlinks to real training datasets (not committed)

## Using real datasets

Override `DATA_PATH` to point to a real dataset (e.g. `DATA_PATH=data/datasets/hpdv2/train.json`) when invoking a `scripts/` wrapper.
