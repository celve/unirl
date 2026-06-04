#!/usr/bin/env python
"""UniRL v2 VLM/AR training entry point (Hydra-native).

Thin wrapper around :class:`unirl.trainer.vlm.VLMTrainer` — the AR path's
sibling of ``train_diffusion.py`` (which drives the diffusion ``DiffusionTrainer``).
Kept separate so the VLM path never routes through diffusion / SDE code.

Launch (per node, SPMD; rank 0 owns the driver):
  QWEN_VL_PATH=/path/to/Qwen2.5-VL-7B-Instruct DATA_PATH=/path/to/train.jsonl \
  python -m unirl.train_vlm --config-name=vlm/qwen_vl/argrpo_qwen_vl_geo3k_mc_4x8

This AR entrypoint also serves pure-LLM recipes under ``examples/llm/``
(e.g. ``--config-name=llm/qwen3/ar_drpo_qwen3_4b_base_dpao_sglang``).
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from unirl.trainer.vlm import VLMTrainer


@hydra.main(version_base=None, config_path="../examples", config_name="vlm/qwen_vl/argrpo_qwen_vl_geo3k_mc_4x8")
def main(cfg: DictConfig) -> None:
    trainer = VLMTrainer(
        cfg=cfg,
        batch_size=cfg.batch_size,
        bundle_cfg=cfg.bundle,
        pipeline_cfg=cfg.pipeline,
        backend_cfg=cfg.backend,
        rollout_cfg=cfg.rollout,
        reward_cfg=cfg.reward,
        algorithm_cfg=cfg.algorithm,
        stack_cfg=cfg.stack,
        data_source_cfg=cfg.data_source,
        sampling_cfg=cfg.sampling,
        sync_cfg=cfg.get("sync"),
        logging_cfg=cfg.get("logging"),
        adv_normalization_scope=cfg.get("adv_normalization_scope", "group"),
        normalize_adv_by_std=bool(cfg.get("normalize_adv_by_std", True)),
    )
    trainer.train(
        num_rollouts=int(cfg.get("num_rollouts", 100)),
        weight_sync_interval=int(cfg.get("weight_sync_interval", 1)),
    )


if __name__ == "__main__":
    main()
