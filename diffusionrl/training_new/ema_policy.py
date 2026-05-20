"""EMAPolicy — exponential moving average shadow as a stackable :class:`Policy`.

Wraps :class:`diffusionrl.utils.ema.EMAModuleWrapper` (DTensor-safe) so EMA
composes with the rest of the Policy stack uniformly. Stack ordering: this
should be the **outermost** policy, so the optimizer's ``policy.parameters()``
sees the underlying source's params (FSDPPolicy / LoRAPolicy DTensors), and
the EMA shadow is updated *after* ``optimizer.step()`` via :meth:`step`::

    pipe = HunyuanImage3Pipeline.from_meta_config(cfg)
    policy = compose_policy(pipe.diffusion, [lora_cfg, fsdp_cfg, ema_cfg])
    pipe.bundle.materialize(device=cuda, with_aux=("vae",))
    policy.post_materialize_init()  # walks chain → EMA snapshots shadow

    # Training loop:
    optimizer.step()
    policy.step(optimization_step)        # update EMA shadow

    # Eval / sampling with EMA weights:
    with policy.use_ema_parameters():
        outputs = policy.replay(...)

The shadow is built lazily in :meth:`post_materialize_init` because, at
``__init__`` time, the wrapped source still has meta tensors —
``EMAModuleWrapper`` would clone meta tensors and the snapshot would be
junk. Running after the inner chain finishes its own (re-)init means the
shadow reflects fully materialized + LoRA-reset weights.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional, Union

import torch

from diffusionrl.config.registration import register_config
from diffusionrl.training_new.policy import Policy, PolicyBase
from diffusionrl.utils.ema import EMAModuleWrapper

logger = logging.getLogger(__name__)


@register_config(
    group="training_new/policy",
    name="ema",
    target="diffusionrl.training_new.ema_policy.EMAPolicy",
)
@dataclass
class EMAPolicyConfig:
    """Construction args for :class:`EMAPolicy`.

    The decay schedule mirrors :class:`EMAModuleWrapper` defaults
    (``min((1+step)/(10+step), decay)`` — warmup that asymptotes to
    ``decay`` after a few hundred steps).
    """

    name: ClassVar[str] = "ema"

    # Target EMA decay rate. Effective decay during warmup is
    # ``min((1+step)/(10+step), decay)``.
    decay: float = 0.9999
    # Update EMA every N optimizer steps. ``policy.step()`` is called
    # once per optimizer step; the wrapper short-circuits when the
    # interval condition isn't met.
    update_step_interval: int = 1
    # Storage device for EMA shadow tensors. ``None`` keeps the
    # same device as the source params (recommended for FSDP/DTensor —
    # the clone stays a DTensor on the same shard). Set to ``"cpu"`` to
    # spill the shadow to host memory and free GPU memory.
    shadow_device: Optional[str] = None


class EMAPolicy(PolicyBase):
    """EMA shadow-parameters facade as a stackable :class:`Policy`.

    Inherits :class:`PolicyBase` defaults for ``trainable_module``,
    ``replay``, ``parameters``, ``named_parameters``, ``state_dict``,
    ``load_state_dict``, ``train``, ``eval``. The shadow tensors live
    on ``self.ema`` and are intentionally *not* exposed via
    ``state_dict`` — checkpoint code should call
    :meth:`ema_state_dict` and :meth:`load_ema_state_dict` separately
    so the EMA shadow can be persisted alongside, but distinctly from,
    the model's own state.

    Overrides:

    - ``post_materialize_init`` — chain inward first, then snapshot
      the source's trainable params into a fresh
      :class:`EMAModuleWrapper`.

    Adds:

    - ``step(optimization_step)`` — update shadow after
      ``optimizer.step()``. Call once per step; the wrapper handles
      ``update_step_interval`` and the warmup decay schedule.
    - ``use_ema_parameters()`` — context manager that swaps the EMA
      shadow into the model's params for the duration of a forward,
      and restores the originals on exit. Used for eval / sampling
      against the smoothed weights.
    - ``ema_state_dict`` / ``load_ema_state_dict`` — shadow-only state
      for checkpointing.
    """

    def __init__(
        self,
        config: EMAPolicyConfig,
        source: Union[Any, Policy],
    ) -> None:
        if not hasattr(source, "trainable_module"):
            raise TypeError(
                f"EMAPolicy: source {type(source).__name__} has no "
                "trainable_module() method. Source must be a Stage or "
                "another Policy that exposes the wrap target."
            )
        self.config = config
        self.source = source
        self.model = source.trainable_module()
        self.ema: Optional[EMAModuleWrapper] = None  # built post-materialize
        self._eval_swap_token: Any = None  # cm token for apply/restore

    # ------------------------------------------------------------------
    # Post-materialize: snapshot source params into a fresh EMA shadow
    # ------------------------------------------------------------------

    def post_materialize_init(self) -> None:
        # Inner first — base + LoRA materialize/reset before we snapshot,
        # so the shadow reflects the actual starting weights.
        super().post_materialize_init()

        # Filter to ``requires_grad=True`` so the shadow tracks only
        # *trainable* params:
        #   - With LoRA below us, peft sets base params to
        #     requires_grad=False; only adapter params remain — shadow is
        #     ~tens of MiB.
        #   - Without LoRA, every nn.Parameter starts with
        #     requires_grad=True, so the shadow covers the full base —
        #     ~size of the model, FSDP-sharded across ranks.
        # Going through ``self.source.parameters()`` (typically FSDPPolicy)
        # would *include* frozen base params under a LoRA stack, blowing
        # up shadow memory by ~20 GiB/rank on HI3-scale models.
        params = self._trainable_params()
        device = torch.device(self.config.shadow_device) if self.config.shadow_device is not None else None
        self.ema = EMAModuleWrapper(
            parameters=params,
            decay=float(self.config.decay),
            update_step_interval=int(self.config.update_step_interval),
            device=device,
        )
        if _current_rank() == 0:
            logger.info(
                "EMAPolicy: snapshot %d trainable shadow tensors "
                "(decay=%.6f, update_step_interval=%d, shadow_device=%s)",
                len(params),
                float(self.config.decay),
                int(self.config.update_step_interval),
                str(self.config.shadow_device),
            )

    def _trainable_params(self) -> list:
        """The list of params the shadow should track / step / swap.

        Filters to ``requires_grad=True`` so frozen base params (e.g. when
        peft has set them False under a LoRA stack) are excluded. The
        ordering matches ``self.source.parameters()`` so the
        :class:`EMAModuleWrapper`'s positional mapping stays consistent
        across :meth:`post_materialize_init`, :meth:`step`, and
        :meth:`use_ema_parameters`.
        """
        return [p for p in self.source.parameters() if p.requires_grad]

    # ------------------------------------------------------------------
    # EMA-specific surface (off-Protocol)
    # ------------------------------------------------------------------

    def step(self, optimization_step: Optional[int] = None) -> None:
        """Update the EMA shadow toward the current source params.

        Call once per optimizer step (after ``optimizer.step()``). The
        underlying ``EMAModuleWrapper`` handles the warmup decay schedule
        and the ``update_step_interval`` short-circuit.
        """
        if self.ema is None:
            raise RuntimeError(
                "EMAPolicy.step: shadow not initialized — call "
                "policy.post_materialize_init() after bundle.materialize()."
            )
        self.ema.step(self._trainable_params(), optimization_step)

    @contextmanager
    def use_ema_parameters(self):
        """Context manager — temporarily swap EMA shadow into the model.

        The originals are saved on entry and restored on exit, so any
        forward pass inside the ``with`` block sees the smoothed weights.
        """
        if self.ema is None:
            raise RuntimeError(
                "EMAPolicy.use_ema_parameters: shadow not initialized — "
                "call policy.post_materialize_init() after "
                "bundle.materialize()."
            )
        with self.ema.use_ema_parameters(self._trainable_params()):
            yield

    def apply_ema_to_model(self) -> None:
        """RPC-friendly counterpart of :meth:`use_ema_parameters`.

        Saves the current source params and swaps the EMA shadow into the
        model. Must be paired with :meth:`restore_from_ema`. Used by
        actor-style call sites where the swap and restore are issued as
        separate RPC calls (a context manager would not survive across
        ``actor.method.remote(...)`` boundaries).
        """
        if self.ema is None:
            raise RuntimeError(
                "EMAPolicy.apply_ema_to_model: shadow not initialized — call policy.post_materialize_init() first."
            )
        if self._eval_swap_token is not None:
            raise RuntimeError(
                "EMAPolicy.apply_ema_to_model: EMA already applied; call restore_from_ema() before applying again."
            )
        cm = self.use_ema_parameters()
        cm.__enter__()
        self._eval_swap_token = cm

    def restore_from_ema(self) -> None:
        """Restore source params after :meth:`apply_ema_to_model`.

        No-op if EMA was never applied — safe to call defensively.
        """
        if self._eval_swap_token is None:
            return
        token = self._eval_swap_token
        self._eval_swap_token = None
        token.__exit__(None, None, None)

    def ema_state_dict(self) -> Dict[str, Any]:
        """Return the EMA shadow state for checkpointing.

        Returns ``{}`` if the shadow hasn't been initialized yet
        (post_materialize_init wasn't called).
        """
        if self.ema is None:
            return {}
        return self.ema.state_dict()

    def load_ema_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore the EMA shadow from a previously saved state_dict.

        The shadow must already be initialized (post_materialize_init
        called) so the loaded tensors land in the expected device/dtype
        layout.
        """
        if self.ema is None:
            raise RuntimeError(
                "EMAPolicy.load_ema_state_dict: shadow not initialized — "
                "call policy.post_materialize_init() first so the load "
                "target exists."
            )
        self.ema.load_state_dict(state_dict)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _current_rank() -> int:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


__all__ = ["EMAPolicyConfig", "EMAPolicy"]
