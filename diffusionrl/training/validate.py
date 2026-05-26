"""Cfg-shape preflight validators for the training stack.

Pure-config functions — no Ray, no torch.distributed, no GPU.
"""

from __future__ import annotations

from omegaconf import DictConfig


def _validate_cfg_for_train(cfg: DictConfig) -> None:
    """Fail-fast preflight: every leaf the train actor reads must be present.

    Requires the multi-track shape ``cfg.training.tracks``.
    """
    tracks = cfg.training.get("tracks") if cfg.get("training") is not None else None
    if tracks is None or len(tracks) == 0:
        raise ValueError("cfg.training.tracks must be a non-empty mapping of track-name -> TrainingTrackConfig.")

    algorithms = cfg.algorithm.get("algorithms") if cfg.get("algorithm") else None
    if algorithms is None or len(algorithms) == 0:
        raise ValueError("cfg.algorithm.algorithms must be a non-empty track-keyed dict of StageAlgorithm presets.")

    track_keys = set(tracks.keys())
    alg_keys = set(algorithms.keys())
    if alg_keys != track_keys:
        raise ValueError(
            f"cfg.algorithm.algorithms keys {sorted(alg_keys)} must match cfg.training.tracks "
            f"keys {sorted(track_keys)} (every track needs exactly one algorithm; "
            "the algorithm slot key IS the track name)."
        )

    for name, alg_node in algorithms.items():
        if alg_node.get("_target_") is None:
            raise ValueError(
                f"cfg.algorithm.algorithms.{name} must carry _target_ (use a registered StageAlgorithm preset)"
            )

    for name, track_cfg in tracks.items():
        if track_cfg.get("model") is None or track_cfg.model.get("_target_") is None:
            raise ValueError(f"cfg.training.tracks.{name}.model must carry _target_")
        if not track_cfg.get("source_stage_attr"):
            raise ValueError(f"cfg.training.tracks.{name}.source_stage_attr must be set")
        if track_cfg.get("optimizer") is None:
            raise ValueError(f"cfg.training.tracks.{name}.optimizer must be set")
        if track_cfg.get("lr_scheduler") is None:
            raise ValueError(f"cfg.training.tracks.{name}.lr_scheduler must be set")


__all__ = ["_validate_cfg_for_train"]
