"""Base class for Ray actor groups.

Holds the actor handle list and exposes scatter/gather dispatch. Subclasses
(``TrainActorGroup``, ``RolloutActorGroup``) add role-specific spawn
(``bootstrap``) plus typed control-plane methods.

Broadcast and rank-0 dispatch are intentionally *not* methods here —
they're one-line list comprehensions at the call site, which preserves
static typing on the actor method name (e.g. ``a.update_weights.remote()``
vs. the stringly-typed ``self.call_all("update_weights")``).
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

import ray
from ray.actor import ActorHandle

logger = logging.getLogger(__name__)


class ActorGroup:
    """Holds a list of Ray actor handles and dispatches scatter/gather calls."""

    def __init__(self, actors: Sequence[ActorHandle]) -> None:
        self._actors: List[ActorHandle] = list(actors)
        self.num_actors = len(self._actors)

    def get_actors(self) -> List[ActorHandle]:
        """Return a defensive copy of the actor handles."""
        return list(self._actors)

    def dispose(self) -> None:
        """Kill all managed actors. Safe to call multiple times."""
        for actor in self._actors:
            try:
                ray.kill(actor)
            except Exception:
                pass

    def scatter_gather_async(
        self,
        method: str,
        shards: Sequence[Optional[Any]],
    ) -> List[ray.ObjectRef]:
        """Dispatch ``method(shard)`` per actor; ``None`` shards are skipped.

        Length of ``shards`` must equal ``num_actors``. Returns one
        ``ObjectRef`` per actor with a non-``None`` shard, in actor-rank
        order (so skipped actors shrink the returned list).
        """
        if len(shards) != len(self._actors):
            raise ValueError(f"shards length {len(shards)} does not match actor count {len(self._actors)}")
        refs: List[ray.ObjectRef] = []
        for actor, shard in zip(self._actors, shards):
            if shard is None:
                continue
            refs.append(getattr(actor, method).remote(shard))
        return refs

    def scatter_gather(
        self,
        method: str,
        shards: Sequence[Optional[Any]],
    ) -> List[Any]:
        """Synchronous variant of :meth:`scatter_gather_async`."""
        return ray.get(self.scatter_gather_async(method, shards))


__all__ = ["ActorGroup"]
