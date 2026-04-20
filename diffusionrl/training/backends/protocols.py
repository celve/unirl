"""Duck-typed protocols for optimizer and LR scheduler objects.

VeOmni's ``build_optimizer`` may return a ``MultiOptimizer`` whose ``__init__``
skips ``super().__init__()`` — so it isn't a true ``torch.optim.Optimizer``
even though it subclasses it nominally. VeOmni's ``build_lr_scheduler`` may
return a ``MultiLRScheduler`` that subclasses ``dict``, not
``torch.optim.lr_scheduler.LRScheduler``. Annotating backend hooks and
factory returns with the torch base classes would be a false contract.

These protocols capture the method surface the diffusionrl training code
actually uses, verified by grep across ``ray/train_actor.py``,
``training/stack.py``, and ``training/train_executor.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class OptimizerProtocol(Protocol):
    state: Dict[Any, Dict[str, Any]]

    def step(self) -> None: ...
    def zero_grad(self) -> None: ...
    def state_dict(self) -> Dict[str, Any]: ...
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None: ...


@runtime_checkable
class LRSchedulerProtocol(Protocol):
    def step(self) -> None: ...
    def state_dict(self) -> Dict[str, Any]: ...
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None: ...
    def get_last_lr(self) -> List[float]: ...


__all__ = ["OptimizerProtocol", "LRSchedulerProtocol"]
