"""Actor-group classes used by ``diffusionrl.train``.

- :class:`ActorGroup` is the shared base: it holds the handle list and
  exposes ``scatter_gather`` / ``scatter_gather_async`` dispatch.
- :class:`TrainActorGroup` wraps the distributed training actors.
- :class:`RolloutActorGroup` wraps the distributed rollout actors.

Broadcast and rank-0 dispatch are intentionally left as inline list
comprehensions at the subclass call sites — that keeps static typing on
``actor.method_name`` and avoids the stringly-typed ``call_all("foo")``
shape.
"""

from .base import ActorGroup
from .rollout import RolloutActorGroup
from .train import TrainActorGroup

__all__ = ["ActorGroup", "RolloutActorGroup", "TrainActorGroup"]
