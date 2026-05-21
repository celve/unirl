"""Actor-group classes.

- :class:`ActorGroup` is the shared base: it holds the handle list and
  exposes ``scatter_gather`` / ``scatter_gather_async`` dispatch.

Concrete actor groups live in ``rollout.py`` / ``train.py`` and are
imported directly by ``diffusionrl.train`` rather than re-exported here.
"""

from .base import ActorGroup

__all__ = ["ActorGroup"]
