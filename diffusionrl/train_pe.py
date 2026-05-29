#!/usr/bin/env python
"""diffusionRL v2 PE (Prompt Enhancement) joint training entry point.

Thin Hydra wrapper around :class:`diffusionrl.trainer.pe.PETrainer`. The
trainer owns the placement scope, wires the two sibling stacks (diffusion +
ar), composes the PEPipeline, and runs the ``train_step → train`` loop; this
module just maps the loaded Hydra config blocks to constructor kwargs.

Pairs with ``conf_v2/pe_trainside.yaml``.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from diffusionrl.trainer.pe import PETrainer


@hydra.main(version_base=None, config_path="../conf_v2", config_name="pe_trainside")
def main(cfg: DictConfig) -> None:
    trainer = PETrainer(
        num_devices=cfg.num_devices,
        batch_size=cfg.batch_size,
        diffusion_cfg=cfg.diffusion,
        ar_cfg=cfg.ar,
        rollout_cfg=cfg.rollout,
        reward_cfg=cfg.reward,
        data_source_cfg=cfg.data_source,
        sampling_cfg=cfg.sampling,
    )
    trainer.train(num_rollouts=int(cfg.get("num_rollouts", 100)))


if __name__ == "__main__":
    main()
