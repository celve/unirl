"""Typed Policy interface + default delegation base.

Policies are *trainable-module facades* — they wrap a stage (or another
policy) and expose the surface a training algorithm needs: ``replay`` for
forward, ``parameters`` / ``state_dict`` / ``load_state_dict`` for
optimizer + checkpoint plumbing, ``train`` / ``eval`` for mode toggles,
plus ``post_materialize_init`` for any (re-)initialization that has to
happen *after* the bundle has materialized weights from the HF checkpoint.

Policies compose by constructor argument:

::

    lora = LoRAPolicy(lora_cfg, pipe.diffusion)   # peft inject on a Stage
    fsdp = FSDPPolicy(fsdp_cfg, lora)              # FSDP wraps the LoRA-modified tree

Both peft injection and FSDP ``fully_shard`` mutate the underlying
``nn.Module`` in place, so ``lora.model is fsdp.model`` after the stack
is built — the difference between two stacked policies is *which surface
they expose*, not *which model they own*.

The Protocol declares the typed interface; ``PolicyBase`` provides
default implementations for the methods that are pure delegation across
most concrete policies, so subclasses override only what's specific to
their behavior.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    Iterator,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

from torch import nn
from torch.nn.parameter import Parameter


@runtime_checkable
class Policy(Protocol):
    """Typed interface for trainable-module facades.

    Any concrete Policy must expose these methods. Optional FSDP- or
    LoRA-specific methods (``clip_grad_norm``, ``offload``,
    ``lora_state_dict``, ``disable_adapter``, ``is_materialized``) live
    on the concrete class and are accessed via concrete-type knowledge
    or ``hasattr`` at call sites.
    """

    @property
    def model(self) -> nn.Module: ...

    def trainable_module(self) -> nn.Module:
        """Return the module a *downstream* Policy or stage would wrap.

        Always ``self.model``. Implemented as a method (not just a
        property) so policies can serve as the ``source`` argument for
        the next policy uniformly with stages.
        """

    def replay(self, *args, **kwargs): ...

    def parameters(self) -> Iterator[Parameter]: ...
    def named_parameters(self) -> Iterator[Tuple[str, Parameter]]: ...

    def state_dict(self) -> Dict[str, Any]: ...
    def load_state_dict(self, sd: Dict[str, Any]) -> None: ...

    def train(self, mode: bool = True) -> None: ...
    def eval(self) -> None: ...

    def post_materialize_init(self) -> None:
        """Run after ``bundle.materialize`` finishes loading HF weights.

        Each Policy that needs to (re-)initialize state owned by it
        overrides and calls ``super().post_materialize_init()`` first
        (so inner policies' init runs before the outer's). The default
        delegates inward through ``self.source`` and stops at a Stage.
        """


class PolicyBase:
    """Default implementations for Policies that compose by holding a
    ``source`` reference.

    Subclasses set ``self.source`` (Stage or Policy) and ``self.model``
    (the wrapped module, typically ``source.trainable_module()``) in
    ``__init__``. Overrides are needed only for methods with
    policy-specific semantics. Most methods here are either:

    * Pure delegation to ``self.source`` (``replay``, ``state_dict``,
      ``load_state_dict``).
    * Direct ops on the in-place-mutated module shared with source
      (``train``, ``eval``, ``parameters``, ``named_parameters``,
      ``trainable_module``).
    * A chain-walking hook (``post_materialize_init``).

    The innermost Policy in a stack — typically ``FSDPPolicy``, whose
    source is a Stage — must override ``state_dict`` /
    ``load_state_dict`` since stages don't implement those. Outer
    Policies (e.g. ``LoRAPolicy``) inherit the delegation defaults.
    """

    # Attribute hints for subclasses; not enforced at runtime.
    source: Union[Any, "Policy"]
    model: nn.Module

    # Pure structural — same for every Policy.
    def trainable_module(self) -> nn.Module:
        return self.model

    # Forward-path delegation — replay always goes to the underlying stage.
    def replay(self, *args, **kwargs):
        return self.source.replay(*args, **kwargs)

    # Mode toggles operate on the in-place-mutated module shared with source.
    def train(self, mode: bool = True) -> None:
        self.model.train(mode)

    def eval(self) -> None:
        self.model.eval()

    @property
    def training(self) -> bool:
        return self.model.training

    # Default param iteration: all params of the wrapped model. Policies
    # that filter (LoRAPolicy → trainable-only) override.
    def parameters(self) -> Iterator[Parameter]:
        return self.model.parameters()

    def named_parameters(self, *args, **kwargs) -> Iterator[Tuple[str, Parameter]]:
        return self.model.named_parameters(*args, **kwargs)

    # Default state-dict semantics: delegate inward. The innermost Policy
    # must override (e.g. FSDPPolicy does the DCP gather).
    def state_dict(self) -> Dict[str, Any]:
        return self.source.state_dict()

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.source.load_state_dict(sd)

    # Post-materialize hook: chain inward. Policies that need to (re-)
    # init state (LoRA: reset adapters; future EMA: snapshot) override
    # and call ``super().post_materialize_init()`` first.
    def post_materialize_init(self) -> None:
        fn = getattr(self.source, "post_materialize_init", None)
        if callable(fn):
            fn()


def walk_source_chain(policy: Any) -> Iterator[Any]:
    """Yield ``policy``, then ``policy.source``, then ``policy.source.source``,
    until the chain reaches a leaf (typically a Stage with no ``source``).

    Useful for tests and introspection (e.g. "find the FSDPPolicy in this
    stack"). Stops cleanly if any link lacks a ``source`` attribute.
    """
    seen: set[int] = set()
    cur = policy
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        cur = getattr(cur, "source", None)


def compose_policy(source: Any, configs: Sequence[Any]) -> Any:
    """Build a Policy stack by wrapping ``source`` with each config in order.

    ``configs`` is an ordered iterable of policy *config instances* (each
    registered via :func:`diffusionrl.config.registration.register_config`
    with a ``target=`` pointing at a Policy class with the standard
    ``__init__(config, source)`` signature). Wrapping is inside-out: the
    *first* config in ``configs`` becomes the *innermost* Policy (closest
    to ``source``); the *last* becomes the outermost (the handle returned).

    An empty ``configs`` returns ``source`` unchanged — useful so call
    sites don't need to special-case the no-policy path.

    Example::

        compose_policy(stage, [lora_cfg, fsdp_cfg, ema_cfg])
        # equivalent to
        # EMAPolicy(ema_cfg,
        #   FSDPPolicy(fsdp_cfg,
        #     LoRAPolicy(lora_cfg, stage)))

    Raises
    ------
    ValueError
        If a config has no ``_target_`` attribute (i.e. wasn't registered
        with ``register_config(..., target=...)``).
    """
    import hydra.utils

    current: Any = source
    for cfg in configs:
        target_path = getattr(cfg, "_target_", None)
        if not target_path:
            raise ValueError(
                f"compose_policy: config {type(cfg).__name__!r} has no "
                "_target_ — was it created via "
                "@register_config(..., target='diffusionrl....')?"
            )
        cls = hydra.utils.get_method(target_path)
        current = cls(cfg, current)
    return current


__all__ = ["Policy", "PolicyBase", "walk_source_chain", "compose_policy"]
