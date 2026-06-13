"""FSDP training backend.

``backend.py`` holds :class:`FSDPBackend` (the training-state Remote);
``wrap.py`` the FSDP2 model wrapping; ``state.py`` the sharded-state helpers
(state-dict gather/load, grad clipping, onload/offload).

The re-export below keeps ``unirl.train.backend.fsdp.FSDPBackend`` importable —
the dotted ``_target_`` path used across the example recipes — even though the
class now lives in the ``backend`` submodule.
"""

from unirl.train.backend.fsdp.backend import FSDPBackend

__all__ = ["FSDPBackend"]
