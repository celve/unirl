#!/usr/bin/env python
"""AReaL-compatible deep-research barrier training entry point.

Launch (per node, SPMD; rank 0 owns the driver):
  QWEN3_PATH=/path/to/Qwen3-1.7B DATA_PATH=data/asearcher/train.jsonl \
  DEEP_RESEARCH_RUN_DATE=YYYY-MM-DD \
  SERPER_KEY_ID=... JINA_API_KEYS=... JUDGE_URL=... JUDGE_MODEL=... \
  python -m unirl.train_areal --config-name=deep_research/deep_research_search_judge
"""

from __future__ import annotations

import datetime

import hydra
from omegaconf import DictConfig

from unirl.trainer.areal import ARealTrainer
from unirl.utils.graceful_shutdown import GracefulShutdown
from unirl.utils.misc import set_seed


@hydra.main(version_base=None, config_path="../examples", config_name="deep_research/deep_research_search_judge")
def main(cfg: DictConfig) -> None:
    if cfg.get("seed") is not None:
        set_seed(int(cfg.seed))
    if cfg.get("run_date") is not None:
        run_date = str(cfg.run_date)
        try:
            parsed_date = datetime.date.fromisoformat(run_date)
        except ValueError as exc:
            raise ValueError(f"run_date must use YYYY-MM-DD format; got {cfg.run_date!r}") from exc
        if parsed_date.isoformat() != run_date:
            raise ValueError(f"run_date must use YYYY-MM-DD format; got {cfg.run_date!r}")

    trainer = None

    def teardown() -> None:
        if trainer is not None:
            trainer.shutdown()

    with GracefulShutdown(teardown, name="areal-train") as guard:
        trainer = ARealTrainer(
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
            sync_cfg=cfg.sync,
            logging_cfg=cfg.get("logging"),
            stop=cfg.get("stop"),
            per_worker_inflight=cfg.get("per_worker_inflight", 8),
        )
        guard.claim_signals()
        trainer.train(
            num_rollouts=cfg.get("num_rollouts", 100),
            save_interval=cfg.get("save_interval", 0),
            save_dir=cfg.get("save_dir"),
            load_dir=cfg.get("load_dir"),
            save_mode=cfg.get("save_mode", "auto"),
        )


if __name__ == "__main__":
    main()
