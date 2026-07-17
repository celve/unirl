#!/usr/bin/env python
"""Fully asynchronous, single-rollout agentic training with SAO.

The train slab hosts two independent FSDP models: an actor optimized with
Direct Importance Sampling and a token-value critic optimized twice before
each actor step.  A resident agentic rollout engine occupies the other slab;
only actor weights are synchronized to it.

Mechanical 4B smoke::

  QWEN3_INSTRUCT_PATH=/path/to/Qwen3-4B-Instruct \
  DATA_PATH=/path/to/train.jsonl \
  python -m unirl.train_sao_deep_research \
    --config-name=deep_research/deep_research_sao_4b_smoke

Paper-scale compose-only blueprint (grouped TP rollout and TP-aware actor
publication are not implemented by the current agentic engine)::

  QWEN3_INSTRUCT_PATH=/path/to/Qwen3-30B-A3B-Thinking-2507 \
  VALUE_MODEL_PATH=/path/to/qwen3-30b-token-value \
  DATA_PATH=/path/to/tir.jsonl \
  python -m unirl.train_sao_deep_research \
    --config-name=deep_research/deep_research_sao_30b_math
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from unirl.trainer.sao_async import AsyncSAOTrainer


@hydra.main(
    version_base=None,
    config_path="../examples",
    config_name="deep_research/deep_research_sao_4b_smoke",
)
def main(cfg: DictConfig) -> None:
    trainer = AsyncSAOTrainer(
        cfg=cfg,
        batch_size=cfg.batch_size,
        actor_cfg=cfg.actor,
        critic_cfg=cfg.critic,
        rollout_cfg=cfg.rollout,
        reward_cfg=cfg.reward,
        stack_cfg=cfg.stack,
        data_source_cfg=cfg.data_source,
        sampling_cfg=cfg.sampling,
        sync_cfg=cfg.get("sync"),
        logging_cfg=cfg.get("logging"),
        stop=cfg.get("stop"),
        train_fraction=cfg.get("train_fraction", 0.5),
        oversample_batch_size=cfg.get("oversample_batch_size"),
        max_oldest_version_lag=cfg.get("max_oldest_version_lag"),
    )
    trainer.train(
        num_rollouts=cfg.get("num_rollouts", 100),
        weight_sync_interval=cfg.get("weight_sync_interval", 1),
        save_interval=cfg.get("save_interval", 0),
        save_dir=cfg.get("save_dir"),
        load_dir=cfg.get("load_dir"),
        save_mode=cfg.get("save_mode", "full"),
    )


if __name__ == "__main__":
    main()
