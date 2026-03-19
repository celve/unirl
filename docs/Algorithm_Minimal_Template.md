# Algorithm Minimal Template

This page describes the minimal algorithm plugin contract in the current
diffusionrl pipeline.

## What Algorithm Plugins Actually Do

In the current mainline, `algorithm` owns both rollout contracts and the training
objective:

- `from_config(config)` is the runtime construction entrypoint
- `get_sampling_requirements()` declares rollout contract requirements
- `compute_advantages_with_components()` / `compute_advantages()` own reward shaping
- `assemble_training_batch()` converts rollout outputs into typed training batches
- `compute_loss_and_backward()` owns gradient computation
- optional `resolve_rollout_sde_indices()` / `get_filtered_training_indices()` refine timestep behavior

`from_args(args)` may still exist as a local convenience wrapper, but rollout and
training actors instantiate algorithms through `from_config(config)`.

## Single Import Rule

Use `diffusionrl.types` as the data-contract entrypoint in plugin code.

## Minimal Skeleton (Rollout-Centric)

```python
from typing import Any, Dict, Optional, Set, Tuple

import torch
import torch.nn as nn

from diffusionrl.algorithms.base import BaseAlgorithm, SamplingRequirements
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
        del model, advantages, embeddings, kwargs
        loss = timestep_data.latents.float().sum() * 0.0
        return loss, {"placeholder": True}


class MyAlgorithm(BaseAlgorithm):
    _loss_cls = _MyLoss

    def __init__(
        self,
        *,
        sde_ratio: float = 1.0,
        train_only_sde_steps: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.sde_ratio = float(sde_ratio)
        self.train_only_sde_steps = bool(train_only_sde_steps)
        self._current_sde_indices: Optional[Set[int]] = None

    @classmethod
    def from_config(cls, config: dict) -> "MyAlgorithm":
        extra = dict(config.get("algorithm_kwargs") or {})
        return cls(
            sde_ratio=float(extra.get("sde_ratio", config.get("sde_ratio", 1.0))),
            train_only_sde_steps=bool(extra.get("train_only_sde_steps", False)),
        )

    def get_sampling_requirements(self) -> SamplingRequirements:
        return SamplingRequirements(
            requires_trajectory=True,
            requires_log_prob=True,
            requires_embeddings=True,
            extras={"sde_ratio": self.sde_ratio},
        )
```

## If You Need a Custom Objective

Define a private loss helper inside the algorithm module and bind it via
`_loss_cls`, as shown above. There is no standalone `--loss-path` / `--loss-type`
extension point in the current algorithm-centric runtime.

## Use in Training

```bash
python -m diffusionrl.train \
  --algorithm-path my_project.algorithms.my_algo.MyAlgorithm
```

## Example Source

- `diffusionrl_plugins/algorithms/minimal_algorithm.py`
- `docs/Loss_Minimal_Template.md`
