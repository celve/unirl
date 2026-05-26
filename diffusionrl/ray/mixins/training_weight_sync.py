"""Training-side weight synchronization mixin (weight sender/publisher)."""

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch.nn as nn

from diffusionrl.distributed.weight_sync.track_spec import TrackSyncSpec
from diffusionrl.distributed.weight_sync.weight_sync import WeightSync

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from diffusionrl.ray.placement import PlacementConfig

logger = logging.getLogger(__name__)


class TrainingWeightSyncMixin:
    """Mixin providing weight-sync sender methods for training actors.

    Delegates to :class:`WeightSync` which manages per-track metadata
    and drives the shared :class:`UpdateWeight` handler.

    Host class must provide these instance attributes:
        model: nn.Module                     -- primary track model (backward compat)
        rank: int                            -- this actor's rank
        _use_lora: bool                      -- primary track LoRA flag (backward compat)
        tracks: Dict[str, TrainTrack]        -- all training tracks
        _track_use_lora: Dict[str, bool]     -- per-track LoRA flags
    """

    model: nn.Module
    rank: int
    _use_lora: bool

    _weight_sync: Optional[WeightSync]

    def _init_weight_sync_state(self) -> None:
        self._weight_sync = None

    def setup_weight_sync(
        self,
        *,
        sync_cfg: "DictConfig",
        placement_cfg: "PlacementConfig",
        rollout_runtime: Any,
        param_name_prefix: str = "",
        packed_modules: dict | None = None,
        track_sync_specs: Dict[str, dict] | None = None,
    ) -> None:
        """Configure multi-track weight sync.

        When ``track_sync_specs`` is provided, registers one
        :class:`TrackSyncSpec` per track. Otherwise falls back to
        single-track mode using ``self.model`` / ``self._use_lora``
        for backward compatibility.
        """
        from diffusionrl.config.instantiate import build

        handler = build(
            sync_cfg,
            model=self.model,
            rollout_runtime=rollout_runtime,
            placement_cfg=placement_cfg,
            param_name_prefix=str(param_name_prefix or ""),
        )
        handler._packed_modules = dict(packed_modules or {})
        handler.connect_rollout_engines()

        ws = WeightSync(handler)

        if track_sync_specs is not None:
            for name in track_sync_specs:
                track = self.tracks[name]
                spec_cfg = track_sync_specs[name]
                ws.register_track(
                    name,
                    TrackSyncSpec(
                        model=track.stage.trainable_module(),
                        use_lora=self._track_use_lora[name],
                        param_name_prefix=spec_cfg.get("param_name_prefix", ""),
                        packed_modules=spec_cfg.get("packed_modules", {}),
                    ),
                )
        else:
            ws.register_track(
                "__primary__",
                TrackSyncSpec(
                    model=self.model,
                    use_lora=self._use_lora,
                    param_name_prefix=param_name_prefix,
                    packed_modules=dict(packed_modules or {}),
                ),
            )

        self._weight_sync = ws
        logger.info(
            "Rank %s: weight sync configured (tracks=%s, handler=%s)",
            self.rank,
            ws.track_names,
            sync_cfg.get("_target_"),
        )

    def sync_weights_to_rollout(self) -> None:
        if self._weight_sync is None:
            raise RuntimeError("Weight sync not configured. Call setup_weight_sync() first.")
        self._weight_sync.sync_all()

    def teardown_weight_sync(self) -> None:
        self._weight_sync = None
        logger.info("Rank %s: weight sync torn down", self.rank)
