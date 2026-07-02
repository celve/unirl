"""Fail-loud guards for the Megatron backend.

The dominant risk across the milestone plan is **silent corruption** in a
half-implemented parallelism axis — training or sampling on wrong weights with no
crash. Every not-yet-supported axis raises here, mirroring the existing
``dispatch.py`` ``pp_size>1`` guard, so an unsupported config fails loud instead.

Pure-python (no torch / mcore imports) so it runs in Hydra compose / config
linting on a machine without CUDA.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unirl.train.configs import MegatronConfig


def assert_supported_topology(cfg: "MegatronConfig", milestone: str = "M0") -> None:
    """Reject parallelism topologies a milestone does not yet implement.

    M0 supports DP only (``tp=pp=vpp=ep=cp=1``, plus the existing Ulysses SP,
    which rides the DP mesh). ``tp>1`` needs the vocab-parallel loss all-reduce +
    TP weight resharding (M1); ``pp/ep/vpp>1`` need the PP/EP walk + the
    ``(dp,tp,pp,ep)`` dispatch mesh (M2). Raising here is the whole point — a
    silently-partial axis would ship fragment weights / a wrong loss.
    """
    if milestone != "M0":
        raise ValueError(f"assert_supported_topology: unknown milestone {milestone!r}")
    if cfg.tp_size > 1 or cfg.pp_size > 1 or cfg.ep_size > 1 or cfg.cp_size > 1 or (cfg.vpp_size or 1) > 1:
        raise NotImplementedError(
            "MegatronBackend M0 supports DP only "
            f"(got tp={cfg.tp_size}, pp={cfg.pp_size}, vpp={cfg.vpp_size}, "
            f"ep={cfg.ep_size}, cp={cfg.cp_size}; all must be 1). "
            "tp>1 needs the vocab-parallel loss + TP reshard (M1); "
            "pp/ep/vpp>1 need the PP/EP walk + dispatch mesh (M2)."
        )


def assert_supported_save_mode(mode: str) -> None:
    """LoRA/adapter checkpointing has no native mcore ``dist_checkpointing``
    equivalent in M0, so ``mode="adapter"`` raises rather than writing an
    unloadable / silently-empty adapter checkpoint. ``resolve_save_mode("auto")``
    must never pick ``adapter`` on the mcore path.
    """
    if mode == "adapter":
        raise NotImplementedError(
            "MegatronBackend M0 does not support mode='adapter' (no native mcore "
            "adapter dist_checkpointing). Full-finetune only in v1."
        )
