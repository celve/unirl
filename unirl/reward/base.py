"""Base abstractions for reward backends.

A reward is one backend — either a local in-process scorer (CPU check or
small-GPU model) or the remote RewardService HTTP client. Both implement
:class:`RewardBackend`; :class:`unirl.reward.service.RewardService`
holds exactly one of them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from unirl.types.reward import RewardRequest, RewardResponse


class RewardBackend(ABC):
    """Turn a :class:`RewardRequest` into a :class:`RewardResponse`.

    Implemented by local scorers (:class:`LocalRewardBackend`,
    :class:`VideoRewardScorer`) and the remote client
    (:class:`RemoteRewardBackend`).
    """

    input_kind = "image"

    def __init__(
        self,
        model_name: str = "",
        batch_size: int = 8,
        timeout: float = 60.0,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.timeout = timeout

    def get_model_name(self) -> str:
        """Name of the reward model/component this backend serves."""
        return self.model_name

    @property
    def preferred_input_kind(self) -> str:
        """The decoded media kind this backend consumes (image/video/text)."""
        return str(getattr(self, "input_kind", "image") or "image").strip().lower()

    @abstractmethod
    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        """Score the request."""

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this backend is ready to score."""

    def offload(self) -> None:
        """Optional lifecycle hook: release device memory."""

    def onload(self) -> None:
        """Optional lifecycle hook: reacquire device memory."""

    def dispose(self) -> None:
        """Optional lifecycle hook: terminal cleanup."""


class BaseRewardComponentSpec(ABC):
    """Marker base for every reward backend spec.

    Each backend registers a concrete ``<Name>Spec`` dataclass via
    ``@register_config(group="reward/component", name="<short>", target=...)``
    and inherits from this base. Kept as a plain ``ABC`` (not a ``@dataclass``)
    so that ``register_config``'s auto-promotion processes each subclass's own
    field annotations without ``is_dataclass(parent)`` short-circuiting.

    The Spec is pure data; ``target`` points at the backend constructor, which
    :func:`unirl.config.instantiate.build` invokes per component.
    """


__all__ = [
    "BaseRewardComponentSpec",
    "RewardBackend",
]
