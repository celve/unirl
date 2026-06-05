# Installation

Install the core training, inference, and evaluation dependencies:

```bash
pip install -e ".[train,infer,eval]" --no-build-isolation
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

For development tools (lint and tests):

```bash
pip install -e ".[train,infer,eval,dev]" --no-build-isolation
```

## Environment

Example configs read cluster-local paths, checkpoints, data, and W&B settings from
environment variables via `${oc.env:...}`. Common variables:

| Variable | Purpose |
|---|---|
| `PRETRAINED_MODEL` | Base model checkpoint path |
| `DATA_PATH` | Training data / prompt-list path |
| `EVAL_DATA_PATH` | Evaluation data path |
| `REPORT_TO_WANDB` | Enable W&B logging (`true` / `false`) |
| `WANDB_PROJECT` | W&B project name |
| `WANDB_ENTITY` | W&B entity / team |

Sample prompt lists are committed under `datasets/`.

Once installed, see the [launch guide](examples/README.md#running-a-recipe) to run an experiment.
