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

    # Default: a track named X hosts one algorithm at slot X. A track may
    # opt into hosting multiple algorithms via ``algorithm_keys``
    # (HI3 shared-backbone case: training track "image" hosts both
    # ``algorithms.image`` and ``algorithms.ar``). Every algorithm slot
    # must be claimed by exactly one track; every claimed slot must exist.
    alg_keys = set(algorithms.keys())

    claimed_by_track: dict = {}  # alg_key -> track_name
    for track_name, track_cfg in tracks.items():
        claimed = track_cfg.get("algorithm_keys") if hasattr(track_cfg, "get") else None
        keys_for_track = [str(k) for k in claimed] if claimed else [track_name]
        for k in keys_for_track:
            if k in claimed_by_track:
                raise ValueError(
                    f"cfg.algorithm.algorithms slot {k!r} is claimed by both "
                    f"track {claimed_by_track[k]!r} and track {track_name!r}; "
                    f"each algorithm may be hosted by exactly one training track."
                )
            claimed_by_track[k] = track_name

    missing = sorted(set(claimed_by_track) - alg_keys)
    if missing:
        raise ValueError(
            f"cfg.training.tracks reference algorithm slots {missing} that "
            f"have no entry in cfg.algorithm.algorithms (have {sorted(alg_keys)})."
        )
    orphaned = sorted(alg_keys - set(claimed_by_track))
    if orphaned:
        raise ValueError(
            f"cfg.algorithm.algorithms slots {orphaned} are not hosted by any "
            f"training track. List them in some track's algorithm_keys, or "
            f"create matching training tracks. Hosted slots: "
            f"{sorted(claimed_by_track)}."
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
