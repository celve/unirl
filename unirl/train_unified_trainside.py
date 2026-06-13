#!/usr/bin/env python
"""UniRL v2 unified-backbone trainside joint training entry point.

Thin Hydra wrapper around :class:`unirl.trainer.unified_trainside.UnifiedTrainsideTrainer`.
The trainer owns the placement scope, wires the single shared backbone + two
algorithms + ``UnifiedModelTrainStack``, builds the trainside composed rollout,
and runs the ``train_step → train`` loop; this module maps Hydra config blocks
to constructor kwargs.

Pairs with ``examples/unified_model/bagel_unified_trainside.yaml``.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from unirl.trainer.unified_trainside import UnifiedTrainsideTrainer


@hydra.main(version_base=None, config_path="../examples", config_name="unified_model/bagel_unified_trainside")
def main(cfg: DictConfig) -> None:
    trainer = UnifiedTrainsideTrainer(
        cfg=cfg,
        batch_size=cfg.batch_size,
        bundle_cfg=cfg.bundle,
        pipeline_cfg=cfg.pipeline,
        backend_cfg=cfg.backend,
        ar_algorithm_cfg=cfg.algorithm.ar,
        image_algorithm_cfg=cfg.algorithm.image,
        stack_cfg=cfg.stack,
        rollout_cfg=cfg.rollout,
        reward_cfg=cfg.reward,
        data_source_cfg=cfg.data_source,
        sampling_cfg=cfg.sampling,
        logging_cfg=cfg.get("logging"),
    )
    trainer.train(
        num_rollouts=int(cfg.get("num_rollouts", 100)),
        weight_sync_interval=int(cfg.get("weight_sync_interval", 1)),
    )


if __name__ == "__main__":
    main()
