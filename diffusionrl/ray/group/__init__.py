"""Actor-group classes.

- :class:`ActorGroup` is the shared base: it holds the handle list and
  exposes ``scatter_gather`` / ``scatter_gather_async`` dispatch.

NEW-stack actor groups live in ``new_rollout.py`` / ``new_train.py``
and are imported directly by ``train_new.py`` rather than re-exported
here.
"""

from .base import ActorGroup

__all__ = ["ActorGroup"]
