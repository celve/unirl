"""Driver-side rollout managers (LIN-693). See ``unirl/rollout/README.md``.

The layer between the trainer and the rollout engines: it owns admission,
acceptance and disposal over time, and holds no model. Named *manager* rather than
*scheduler* because this tree already calls three unrelated things a scheduler —
the LR scheduler, the diffusion noise scheduler, and SGLang's own subprocesses.
"""

from unirl.rollout.manager.agentic import AgenticManager, Carried
from unirl.rollout.manager.batch import BatchManager, InflightPool
from unirl.rollout.manager.buffers import PendingGroups, VersionedBuffer, root_of
from unirl.rollout.manager.protocol import RolloutManager

__all__ = [
    "AgenticManager",
    "BatchManager",
    "Carried",
    "InflightPool",
    "PendingGroups",
    "RolloutManager",
    "VersionedBuffer",
    "root_of",
]
