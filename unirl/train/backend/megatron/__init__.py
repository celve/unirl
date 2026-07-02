"""Megatron-Core training backend (M0: DP only).

Sibling of :class:`~unirl.train.backend.fsdp.FSDPBackend` and
:class:`~unirl.train.backend.veomni.VeOmniBackend`, but — unlike those — it is a
direct :class:`~unirl.distributed.group.remote.Remote` subclass, **not** a
:class:`~unirl.train.backend.base_backend.BaseFSDP2Backend` child: the base's
checkpoint / optimizer-step / on-offload members all assume FSDP2 DTensors, which
do not describe mcore's ``parallel_state``-managed params. The genuinely-shared
~40-line surface (EMA swap, eval-swap, the save/load visibility handshake) is
copied, not inherited.

Like ``veomni``, the re-export is lazy (PEP 562): ``backend.py`` imports Megatron
at module level, but recipe tooling (Hydra ``_target_`` compose checks, config
linting) must import this package on torch-less / mcore-less machines.
``unirl.train.backend.megatron.MegatronBackend`` still resolves as a Hydra
``_target_`` — ``hydra.utils.get_method``/``get_class`` trigger the lazy load.
"""

from typing import Any

__all__ = ["MegatronBackend"]


def __getattr__(name: str) -> Any:
    if name == "MegatronBackend":
        from unirl.train.backend.megatron.backend import MegatronBackend

        return MegatronBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
