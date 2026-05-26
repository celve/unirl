"""Multi-track weight sync orchestrator.

Holds per-track :class:`TrackSyncSpec` metadata and a single
:class:`UpdateWeight` handler. Iterates tracks, extracts tensors
(LoRA or full-FT), and calls the handler per-track. When multiple
tracks are registered, tensor keys are prefixed with the track name
so the rollout-side :class:`ComposedRolloutEngine` can demux.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch
import torch.nn as nn

from diffusionrl.distributed.weight_sync.base import UpdateWeight
from diffusionrl.distributed.weight_sync.track_spec import TrackSyncSpec

logger = logging.getLogger(__name__)


def _resolve_peft_config_obj(
    model: nn.Module,
    adapter_name: str = "default",
) -> Any:
    """Walk model wrap layers and return the per-adapter PEFT config object."""
    cur: Any = model
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        pc = getattr(cur, "peft_config", None)
        if isinstance(pc, dict) and adapter_name in pc:
            return pc[adapter_name]
        cur = getattr(cur, "module", None) or getattr(cur, "base_model", None)
    return None


def _peft_config_dict(model: nn.Module, adapter_name: str = "default") -> dict:
    """Return a JSON/Ray-safe PEFT config dict for one adapter."""
    peft_cfg_obj = _resolve_peft_config_obj(model, adapter_name)
    if peft_cfg_obj is None:
        raise RuntimeError(f"WeightSync._peft_config_dict: model has no peft_config[{adapter_name!r}] entry.")

    if hasattr(peft_cfg_obj, "to_dict"):
        peft_dict = peft_cfg_obj.to_dict()
    else:
        peft_dict = dict(peft_cfg_obj)

    tm = peft_dict.get("target_modules")
    if isinstance(tm, str):
        pass
    elif isinstance(tm, (list, tuple, set, frozenset)):
        peft_dict["target_modules"] = sorted(tm) if isinstance(tm, (set, frozenset)) else list(tm)
    elif tm is None:
        pass
    else:
        raise RuntimeError(
            f"_peft_config_dict: peft_config['target_modules'] has "
            f"unsupported type {type(tm).__name__}; expected str / list / "
            f"set / tuple."
        )

    for required in ("r", "lora_alpha", "target_modules"):
        if peft_dict.get(required) in (None, "", [], ()):
            raise RuntimeError(
                f"_peft_config_dict: peft_config[{required!r}] is "
                f"missing or empty (got {peft_dict.get(required)!r}); "
                f"rollout LoRA receive will reject this."
            )

    return peft_dict


class WeightSync:
    """Multi-track weight sync orchestrator.

    Holds per-track metadata and a shared handler. Extracts tensors
    per-track, optionally prefixes keys with the track name, and
    calls the handler.
    """

    def __init__(self, handler: UpdateWeight) -> None:
        self._handler = handler
        self._tracks: Dict[str, TrackSyncSpec] = {}

    def register_track(self, name: str, spec: TrackSyncSpec) -> None:
        self._tracks[name] = spec
        logger.info(
            "WeightSync: registered track %r (use_lora=%s, prefix=%r)",
            name,
            spec.use_lora,
            spec.param_name_prefix,
        )

    @property
    def track_names(self) -> list[str]:
        return list(self._tracks.keys())

    def sync_all(self) -> None:
        """Sync all registered tracks sequentially, then barrier."""
        for name in self._tracks:
            self.sync_track(name)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def sync_track(self, name: str) -> None:
        """Sync one track's weights to rollout."""
        spec = self._tracks[name]
        multi_track = len(self._tracks) > 1
        track_prefix = name if multi_track else ""

        if spec.use_lora:
            peft_cfg = _peft_config_dict(spec.model)
            if not spec.base_sync_done:
                self._handler.update_weights(
                    model=spec.model,
                    peft_config=None,
                    base_sync_done=False,
                    param_name_prefix=spec.param_name_prefix,
                    packed_modules=spec.packed_modules,
                    track_prefix=track_prefix,
                )
                spec.base_sync_done = True

            self._handler.update_weights(
                model=spec.model,
                peft_config=peft_cfg,
                base_sync_done=True,
                param_name_prefix=spec.param_name_prefix,
                packed_modules=spec.packed_modules,
                track_prefix=track_prefix,
            )
        else:
            self._handler.update_weights(
                model=spec.model,
                param_name_prefix=spec.param_name_prefix,
                packed_modules=spec.packed_modules,
                track_prefix=track_prefix,
            )


__all__ = ["WeightSync"]
