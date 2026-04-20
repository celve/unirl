"""Actor-group classes used by ``diffusionrl.train``.

- ``TrainActorGroup`` wraps the distributed training actors.
- ``RolloutActorGroup`` wraps the distributed rollout actors.

Both classes fold what used to be a data-plane ``ActorGroup`` +
control-plane ``GroupRuntime`` pair into a single class per side.
"""

from .rollout import RolloutActorGroup
from .train import TrainActorGroup

__all__ = ["RolloutActorGroup", "TrainActorGroup"]
