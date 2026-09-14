#!/usr/bin/env python
"""Hydra entry point for the experimental.leo2 t2v FlowGRPO recipe."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from unirl.train_diffusion import run


@hydra.main(version_base=None, config_path="examples", config_name="leo2_t2v_trainside")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
