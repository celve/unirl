# Loss Minimal Template

This file keeps the historical name, but the current mainline no longer supports
external `--loss-path` plugins. Loss logic now lives inside the algorithm module.

This page shows the current algorithm-owned loss pattern.

## Why This Matters

In the default training pipeline, gradients are computed by the algorithm's
private loss helper (`_loss_cls`) together with
`algorithm.compute_loss_and_backward()`. If you want to change the objective,
define the loss helper inside your algorithm module and wire it through the
algorithm class.

## Minimal Loss Helper Skeleton (GRPO-like)

```python
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from diffusionrl.algorithms.base import BaseAlgorithm
from diffusionrl.types import PromptEmbeddings, TimestepData


class _MyLoss:
    def __init__(self, algorithm: "MyAlgorithm") -> None:
        self.algorithm = algorithm

    def compute_loss(
        self,
        model: nn.Module,
        timestep_data: TimestepData,
        advantages: torch.Tensor,
        embeddings: PromptEmbeddings,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Replace with your objective.
        loss = timestep_data.latents.float().sum() * 0.0
        return loss, {"placeholder": True}


class MyAlgorithm(BaseAlgorithm):
    _loss_cls = _MyLoss
```

## Forward-Batch Variant (NFT-like)

```python
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from diffusionrl.types import ForwardTrainingBatch


class _MyForwardLoss:
    def __init__(self, algorithm: "MyAlgorithm") -> None:
        self.algorithm = algorithm

    def compute_loss(
        self,
        model: nn.Module,
        batch: ForwardTrainingBatch,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Replace with your objective.
        loss = batch.clean_latents.float().sum() * 0.0
        return loss, {"placeholder": True}
```

## Use in Training

```bash
python -m diffusionrl.train \
  --algorithm.algorithm-path my_project.algorithms.my_algo.MyAlgorithm \
  --algorithm.algorithm-kwargs '{"my_coef": 0.3}'
```

Use `algorithm.algorithm_kwargs` to feed custom knobs into `MyAlgorithm.from_config()`.
For a full working example, copy `diffusionrl_plugins/algorithms/minimal_algorithm.py`
instead of creating a standalone loss plugin.

## Example Source

- `diffusionrl_plugins/algorithms/minimal_algorithm.py`
