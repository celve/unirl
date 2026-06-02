#!/usr/bin/env python
"""UniRL v2 HunyuanImage3 training entry point (Hydra-native).

Thin wrapper around :class:`unirl.trainer.hi3.HI3Trainer`. The trainer
owns the placement scope, sibling Remote wiring, and the ``train_step → train``
loop; this module just maps the loaded Hydra config blocks to constructor
kwargs.

Pairs with ``recipes/unified_model_rl/hi3_vllmomni.yaml``::

    python -m unirl.train_hi3 --config-name unified_model_rl/hi3_vllmomni
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from unirl.trainer.hi3 import HI3Trainer


@hydra.main(version_base=None, config_path="../recipes", config_name="unified_model_rl/hi3_vllmomni")
def main(cfg: DictConfig) -> None:
    trainer = HI3Trainer(
        num_devices=cfg.num_devices,
        batch_size=cfg.batch_size,
        bundle_cfg=cfg.bundle,
        pipeline_cfg=cfg.pipeline,
        backend_cfg=cfg.backend,
        ar_rollout_cfg=cfg.ar_rollout,
        dit_rollout_cfg=cfg.dit_rollout,
        reward_cfg=cfg.reward,
        ar_algorithm_cfg=cfg.algorithm.ar,
        image_algorithm_cfg=cfg.algorithm.image,
        stack_cfg=cfg.stack,
        data_source_cfg=cfg.data_source,
        sampling_cfg=cfg.sampling,
        sync_cfg=cfg.get("sync"),
        dump_dir=cfg.get("dump_dir"),
        logging_cfg=cfg.get("logging"),
        enable_fsdp_offload=cfg.get("enable_fsdp_offload", True),
    )
    trainer.train(
        num_rollouts=int(cfg.get("num_rollouts", 100)),
        weight_sync_interval=int(cfg.get("weight_sync_interval", 1)),
    )


if __name__ == "__main__":
    main()
