#!/usr/bin/env python
"""UniRL agentic image joint training entry point (LIN-577).

Thin Hydra wrapper around :class:`unirl.trainer.agentic_image.AgenticImageTrainer`.
The trainer owns the placement scope and wires the two sibling stacks (diffusion +
ar) exactly as PE does; this module maps Hydra config blocks to constructor kwargs.

Pairs with ``examples/agentic_image/agentic_image_flux_klein_sglang.yaml``.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from unirl.trainer.agentic_image import AgenticImageTrainer


@hydra.main(version_base=None, config_path="../examples", config_name="agentic_image/agentic_image_flux_klein_sglang")
def main(cfg: DictConfig) -> None:
    trainer = AgenticImageTrainer(
        cfg=cfg,
        batch_size=cfg.batch_size,
        diffusion_cfg=cfg.diffusion,
        ar_cfg=cfg.ar,
        rollout_cfg=cfg.rollout,
        reward_cfg=cfg.reward,
        data_source_cfg=cfg.data_source,
        sampling_cfg=cfg.sampling,
        sync_cfg=cfg.get("sync"),
        logging_cfg=cfg.get("logging"),
        enable_fsdp_offload=cfg.get("enable_fsdp_offload", False),
        freeze_llm=cfg.get("freeze_llm", False),
        eval_interval=int(cfg.get("eval_interval", 0)),
        stop=cfg.get("stop"),
    )
    trainer.train(
        num_rollouts=int(cfg.get("num_rollouts", 100)),
        weight_sync_interval=int(cfg.get("weight_sync_interval", 1)),
        save_interval=int(cfg.get("save_interval", 0)),
        save_dir=cfg.get("save_dir"),
        load_dir=cfg.get("load_dir"),
        save_mode=str(cfg.get("save_mode", "auto")),
    )


if __name__ == "__main__":
    main()
