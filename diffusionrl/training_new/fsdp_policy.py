"""FSDPPolicy — stage-scoped FSDP wrap + lifecycle facade.

A concrete :class:`Policy` whose source is a trainable Stage
(``DiffusionStage`` / ``ARStage``). FSDPPolicy ``fully_shard``s the
trainable surface and provides the DTensor-aware ``state_dict`` /
``load_state_dict`` implementations that outer policies (e.g.
:class:`LoRAPolicy`) delegate to. Inherits :class:`PolicyBase` for
``replay`` / ``train`` / ``eval`` / ``parameters`` /
``named_parameters`` / ``trainable_module`` / ``post_materialize_init``
defaults.

Block-class discovery (which submodules to ``fully_shard`` individually):

  1. **HF auto-discovery**: walk ``type(trainable_root).__mro__`` looking for
     ``_no_split_modules`` (HF ``PreTrainedModel`` convention).
  2. **Stage-class fallback**: ``getattr(type(source), "_no_split_modules", ())``.
     For models without HF heritage (e.g. SD3 from diffusers), the stage author
     declares the class names model-side as a class attribute.
  3. **Empty fallback**: log a warning and fall through to root-only wrap.

LoRA-related methods (``lora_state_dict``, ``disable_adapter``) live on
:class:`LoRAPolicy` — compose by stacking that policy on top of FSDPPolicy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional, Tuple, Union

import torch
from torch import nn

from diffusionrl.config.registration import register_config
from diffusionrl.config.validation import validate_precision_type
from diffusionrl.training_new.policy import Policy, PolicyBase
from diffusionrl.utils.dtypes import parse_torch_dtype

logger = logging.getLogger(__name__)


@register_config(
    group="training_new/policy",
    name="fsdp",
    target="diffusionrl.training_new.fsdp_policy.FSDPPolicy",
)
@dataclass
class FSDPPolicyConfig:
    """Config for the FSDPPolicy.

    Same field shape as the legacy ``FSDPBackendConfig`` so production
    recipes can swap targets without changing the field set.
    """

    name: ClassVar[str] = "fsdp"

    cpu_offload: bool = False
    param_dtype: str = "bf16"
    mixed_precision: bool = True
    fsdp_mode: str = "full"  # "full" | "hybrid" (HSDP)
    reshard_after_forward: bool = True
    activation_checkpointing: bool = False

    def __post_init__(self) -> None:
        self.param_dtype = validate_precision_type(self.param_dtype, field="training_new.policy.param_dtype")


class FSDPPolicy(PolicyBase):
    """Stage-scoped FSDP wrap + lifecycle facade. Inherits
    :class:`PolicyBase`; provides FSDP-specific overrides for
    ``state_dict`` / ``load_state_dict`` (DCP) and the FSDP-only surface
    (``clip_grad_norm``, ``offload`` / ``onload``, ``is_materialized``,
    ``device_mesh``).
    """

    def __init__(
        self,
        config: FSDPPolicyConfig,
        source: Union[Any, Policy],
        *,
        topology: Optional[Any] = None,
    ) -> None:
        # ``topology`` is accepted for call-site symmetry; FSDP2 derives
        # world/DP sizes from ``torch.distributed.get_world_size()`` at wrap.
        del topology
        self.config = config
        # ``source`` is either a trainable Stage (innermost case — typical)
        # or another Policy that was applied first (e.g. a LoRAPolicy that
        # injected adapters before FSDP shards them with the base).
        self.source = source

        # The module FSDP wraps. For HI3 this is ``transformer.model``
        # (the bare decoder); for SD3 it's the ``transformer`` itself.
        # When ``source`` is a Policy, ``trainable_module()`` returns the
        # already-transformed module (e.g. with peft adapters injected).
        if not hasattr(source, "trainable_module"):
            raise TypeError(
                f"FSDPPolicy: source {type(source).__name__} has no "
                "trainable_module() method. Source must be a Stage or a "
                "Policy that exposes the wrap target."
            )
        self.model: nn.Module = source.trainable_module()

        self.last_grad_norm: Optional[torch.Tensor] = None
        self._is_offloaded = False
        self._device_mesh: Optional[Any] = None

        # If the trainable_root has any non-meta parameter, remember the
        # device so ``onload`` can restore. For meta-init this stays the
        # config-recommended cuda device picked at wrap time; the actual
        # cuda assignment happens later via ``bundle.materialize(device)``.
        self._device = _infer_device(self.model)

        self._wrap_model()

    # ------------------------------------------------------------------
    # FSDP2 wrap (incl. block-class discovery + meta-aware dtype cast)
    # ------------------------------------------------------------------

    def _wrap_model(self) -> None:
        from torch.distributed.fsdp import (
            CPUOffloadPolicy,
            MixedPrecisionPolicy,
            fully_shard,
        )

        target_dtype = parse_torch_dtype(self.config.param_dtype, field_name="training.policy.param_dtype")

        fsdp_kwargs: Dict[str, Any] = {
            "reshard_after_forward": bool(self.config.reshard_after_forward),
        }
        if self.config.mixed_precision:
            fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
                param_dtype=target_dtype,
                reduce_dtype=torch.float32,
            )
        if self.config.cpu_offload:
            fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

        mesh = self._create_device_mesh()
        if mesh is not None:
            fsdp_kwargs["mesh"] = mesh
            self._device_mesh = mesh

        block_class_names = self._discover_block_classes()
        block_instances = self._enumerate_block_instances(block_class_names)

        # Pre-wrap dtype homogenization. FSDP2 requires uniform *original*
        # parameter dtype within each shard group regardless of mp_policy
        # (asserted in ``_init_mp_dtypes`` on first forward). HI3 decoder
        # layers carry bf16 attention/MLP weights plus fp32 LayerNorm
        # weights, so cast all floating-point params to ``target_dtype``
        # in-place before wrapping. Meta-init params are also cast: the
        # meta tensor's dtype is metadata that ``to(dtype)`` rewrites
        # without allocating storage, and that metadata is what FSDP
        # records at wrap time and re-checks at first forward. Norm
        # precision drop is acceptable here — matches the prior LIN-243
        # smoke (commits c78e818, aa35008).
        casts = 0
        for layer in block_instances:
            for p in layer.parameters(recurse=True):
                if p.dtype.is_floating_point and p.dtype != target_dtype:
                    p.data = p.data.to(target_dtype)
                    casts += 1

        for layer in block_instances:
            fully_shard(layer, **fsdp_kwargs)

        # Activation checkpointing on the same units FSDP shards. Applied
        # AFTER ``fully_shard`` via direct ``layer.forward`` monkey-patch
        # rather than ``apply_activation_checkpointing`` (which replaces
        # the module in the parent ModuleList). The monkey-patch path is
        # the one experimentally validated at scripts/profile_hi3_ckpt_per_layer.py
        # (peak 92 GB OOM → 45 GB on HI3 1024x1024). The ``apply_activation_checkpointing``
        # path tripped a ``CheckpointError: Recomputed values ... different
        # metadata`` (bf16 saved vs fp32 recomputed) — autocast context
        # wasn't preserved across recompute when kwargs were forwarded
        # through ``checkpoint(... **kwargs)``. Closing over kwargs (so
        # ``ckpt.checkpoint`` only sees positional args) avoids that.
        if self.config.activation_checkpointing:
            from torch.utils import checkpoint as _ckpt

            def _make_ckpt_forward(orig_fwd: Any) -> Any:
                def wrapped(*args: Any, **kwargs: Any) -> Any:
                    def fn(*a: Any) -> Any:
                        return orig_fwd(*a, **kwargs)

                    return _ckpt.checkpoint(fn, *args, use_reentrant=False)

                return wrapped

            for layer in block_instances:
                layer.forward = _make_ckpt_forward(layer.forward)
        # No root ``fully_shard(self.model)``. The HF wrapper (the parent
        # of trainable_root in HI3) calls into leaf children of
        # trainable_root directly — e.g. ``transformer.forward`` does
        # ``self.model.wte(input_ids)`` — and a root wrap would convert
        # those leaves' params to DTensor, breaking the cross-FSDP-boundary
        # call (plain ``input_ids`` × DTensor ``weight`` mixes types in
        # ``F.embedding``). Block-only wrap keeps non-block leaves
        # (``wte``, root-level norms, etc.) as plain tensors that work in
        # both FSDP and non-FSDP code paths. The block instances still get
        # fully sharded; only the root residual stays unwrapped.

        rank = self._current_rank()
        if rank == 0:
            logger.info(
                "FSDPPolicy: wrapped %d backbone block(s) of class %r "
                "(%s, cpu_offload=%s, mixed_precision=%s, reshard_after_forward=%s, "
                "activation_checkpointing=%s, dtype_casts=%d)",
                len(block_instances),
                tuple(block_class_names),
                "HSDP" if mesh is not None else "FSDP2",
                self.config.cpu_offload,
                self.config.mixed_precision,
                fsdp_kwargs["reshard_after_forward"],
                self.config.activation_checkpointing,
                casts,
            )

    # ------------------------------------------------------------------
    # Block-class discovery (HF auto → stage-class fallback)
    # ------------------------------------------------------------------

    def _discover_block_classes(self) -> Tuple[str, ...]:
        """Resolve the class-name tuple to FSDP-shard individually.

        Stages stay free of FSDP-shape knowledge; FSDPPolicy reads the
        architectural fact from where it actually lives. When ``source``
        is a Policy (e.g. LoRAPolicy), we walk inward through the source
        chain to find the original Stage class for the fallback.
        """
        # 1. HF convention: walk the trainable_root's class MRO.
        for cls in type(self.model).__mro__:
            attr = getattr(cls, "_no_split_modules", None)
            if attr:
                return tuple(str(n) for n in attr)
        # 2. Stage-class fallback (model-side declaration). When source
        #    is a Policy, walk inward to the underlying Stage.
        leaf_source = self.source
        while hasattr(leaf_source, "source"):
            leaf_source = leaf_source.source
        attr = getattr(type(leaf_source), "_no_split_modules", None)
        if attr:
            return tuple(str(n) for n in attr)
        # 3. Empty — root-only wrap.
        if self._current_rank() == 0:
            logger.warning(
                "FSDPPolicy: no block classes discovered for trainable_root "
                "%r (source %r). Falling back to root-only wrap; per-block "
                "all-gather granularity will be lost. Declare "
                "_no_split_modules on the stage class to fix.",
                type(self.model).__name__,
                type(leaf_source).__name__,
            )
        return ()

    def _enumerate_block_instances(self, class_names: Tuple[str, ...]) -> Tuple[nn.Module, ...]:
        """Walk ``self.model`` and return all submodule instances whose
        ``type(m).__name__`` is in ``class_names``.

        Class-name matching (rather than ``isinstance``) avoids the
        trust_remote_code synthetic-module problem where the production
        package's class object differs from the dynamically-loaded one.
        """
        if not class_names:
            return ()
        names = set(class_names)
        return tuple(m for _, m in self.model.named_modules() if type(m).__name__ in names)

    # ------------------------------------------------------------------
    # State dict (FSDP-aware gather/broadcast via DCP)
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Return a full state dict on rank 0 (empty dict elsewhere).

        Used for checkpoint export and weight sync to inference engines.
        """
        from torch.distributed.checkpoint.state_dict import get_model_state_dict

        options = self._build_state_dict_options(full_state_dict=True, cpu_offload=True)
        try:
            full = dict(get_model_state_dict(self.model, options=options))
        except TypeError:
            full = dict(get_model_state_dict(self.model))

        if self._current_rank() != 0:
            return {}
        return self._to_cpu_state_dict(full)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load a full state dict, broadcasting from rank 0 across ranks.

        On rank 0, ``state_dict`` carries the full keys; on other ranks it
        is typically an empty dict. The broadcast happens inside
        ``set_model_state_dict`` and lands in each rank's DTensor shard.
        Used for both initial weight load (after meta + materialize) and
        runtime checkpoint restore.
        """
        from torch.distributed.checkpoint.state_dict import set_model_state_dict

        options = self._build_state_dict_options(
            full_state_dict=True,
            broadcast_from_rank0=True,
            cpu_offload=False,
        )
        try:
            set_model_state_dict(self.model, state_dict, options=options)
        except TypeError:
            set_model_state_dict(self.model, state_dict)

    @staticmethod
    def _build_state_dict_options(**kwargs: Any) -> Any:
        """Build ``StateDictOptions`` tolerating cross-version kwarg drift."""
        from torch.distributed.checkpoint.state_dict import StateDictOptions

        candidates = [
            dict(kwargs),
            {k: v for k, v in kwargs.items() if k != "broadcast_from_rank0"},
            {k: v for k, v in kwargs.items() if k in {"full_state_dict", "cpu_offload"}},
            {},
        ]
        for candidate in candidates:
            try:
                return StateDictOptions(**candidate)
            except TypeError:
                continue
        return StateDictOptions()

    # ------------------------------------------------------------------
    # Gradient clip (FSDP-aware; falls back to explicit global-norm path
    # under cpu_offload where DTensor CPU collectives don't work)
    # ------------------------------------------------------------------

    def clip_grad_norm(self, max_grad_norm: float) -> torch.Tensor:
        grad_norm = self._do_clip_grad_norm(max_grad_norm)
        self.last_grad_norm = grad_norm
        return grad_norm

    def _do_clip_grad_norm(self, max_grad_norm: float) -> torch.Tensor:
        if self.config.cpu_offload:
            return self._global_clip_for_sharded_grads(max_grad_norm)
        try:
            clip_fn = getattr(self.model, "clip_grad_norm_", None)
            if callable(clip_fn):
                grad_norm = clip_fn(max_grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=max_grad_norm,
                )
            return self._maybe_dtensor_to_tensor(grad_norm)
        except RuntimeError as exc:
            # Two FSDP corner cases that the standard
            # ``torch.nn.utils.clip_grad_norm_`` path can't handle:
            # - ``No backend type associated with device type cpu`` — CPU
            #   DTensor collectives missing; happens under cpu_offload.
            # - ``mixed torch.Tensor and DTensor`` — per-block ``fully_shard``
            #   (no root wrap) leaves non-block params as regular Tensors
            #   while block params are DTensors; the foreach norm op rejects
            #   the mixed input. The explicit global-norm path handles both.
            msg = str(exc)
            fallback_triggers = (
                "No backend type associated with device type cpu",
                "mixed torch.Tensor and DTensor",
            )
            if not any(t in msg for t in fallback_triggers):
                raise
            logger.warning(
                "FSDPPolicy: grad clipping hit %r; falling back to explicit global-norm clipping path.",
                msg.splitlines()[0] if msg else "<no message>",
            )
            return self._global_clip_for_sharded_grads(max_grad_norm)

    def _global_clip_for_sharded_grads(self, max_grad_norm: float) -> torch.Tensor:
        import torch.distributed as dist

        grads: list[torch.Tensor] = []
        local_sq_sum = 0.0
        for param in self.model.parameters():
            grad = getattr(param, "grad", None)
            if grad is None:
                continue
            local_grad = grad
            if hasattr(local_grad, "to_local") and callable(getattr(local_grad, "to_local")):
                try:
                    local_grad = local_grad.to_local()
                except Exception:
                    pass
            if not isinstance(local_grad, torch.Tensor):
                continue
            local_sq_sum += float(torch.sum(local_grad.detach().float() ** 2).item())
            grads.append(grad)

        if not grads:
            return torch.tensor(0.0)

        reduce_device = torch.device("cpu")
        if torch.cuda.is_available():
            try:
                reduce_device = torch.device(f"cuda:{torch.cuda.current_device()}")
            except Exception:
                reduce_device = torch.device("cuda")

        total_sq = torch.tensor(local_sq_sum, device=reduce_device, dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(total_sq, op=dist.ReduceOp.SUM)
        global_norm = float(torch.sqrt(total_sq).item())
        clip_coef = float(max_grad_norm) / (global_norm + 1e-6)
        if clip_coef < 1.0:
            for grad in grads:
                grad.mul_(clip_coef)
        return torch.tensor(global_norm, device=reduce_device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Offload / onload (params + grads on the FSDP-wrapped trainable_root)
    # ------------------------------------------------------------------

    def offload(self) -> None:
        """Move FSDP-wrapped params + grads to CPU. Idempotent.

        Uses ``Module._apply`` (via ``self.model.cpu()``) so DTensor
        wrappers reconstruct on a CPU device-mesh and ``state_dict`` /
        ``redistribute`` continue to work post-offload. Per-param
        ``param.data = ...`` mutation crashes torch >= 2.9 inside
        ``_correct_storage_aliasing`` because the wrapper's mesh stays on
        cuda while its storage moves to cpu.
        """
        if self._is_offloaded:
            return
        self.model.cpu()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        self._is_offloaded = True
        logger.debug("FSDPPolicy: offloaded params/grads to CPU")

    def onload(self) -> None:
        """Move FSDP-wrapped params + grads back to the recorded device."""
        if not self._is_offloaded:
            return
        self.model.to(self._device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._is_offloaded = False
        logger.debug("FSDPPolicy: onloaded params/grads to %s", self._device)

    # ``replay``, ``parameters``, ``named_parameters``, ``train``,
    # ``eval``, ``training``, ``trainable_module``, ``post_materialize_init``
    # are inherited from :class:`PolicyBase`. ``disable_adapter`` and
    # ``lora_state_dict`` live on :class:`LoRAPolicy`.

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def device_mesh(self) -> Optional[Any]:
        return self._device_mesh

    @property
    def is_offloaded(self) -> bool:
        return self._is_offloaded

    @property
    def is_materialized(self) -> bool:
        """``True`` if no parameter on the wrapped trainable_root is on meta.

        After ``__init__`` on a meta-init bundle this is ``False``;
        ``materialize(device)`` flips it to ``True``.
        """
        for p in self.model.parameters():
            if p.is_meta:
                return False
        return True

    # ------------------------------------------------------------------
    # HSDP mesh
    # ------------------------------------------------------------------

    def _create_device_mesh(self) -> Optional[Any]:
        """Build a 2D ``(dp_replicate, dp_shard)`` mesh for HSDP.

        Shards within groups of 8 GPUs (one node), replicates across
        nodes. Returns ``None`` for ``full`` mode, or for world_size <= 8
        (no benefit), or world_size not a multiple of 8.
        """
        fsdp_mode = str(self.config.fsdp_mode).strip().lower()
        if fsdp_mode != "hybrid":
            return None

        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return None

        world_size = dist.get_world_size()
        shard_size = 8
        if world_size <= shard_size:
            logger.info(
                "FSDPPolicy: hybrid requested but world_size=%d <= %d; falling back to pure FSDP.",
                world_size,
                shard_size,
            )
            return None
        if world_size % shard_size != 0:
            logger.warning(
                "FSDPPolicy: hybrid requested but world_size=%d is not a multiple of %d; falling back to pure FSDP.",
                world_size,
                shard_size,
            )
            return None

        from torch.distributed.device_mesh import init_device_mesh

        replicate_size = world_size // shard_size
        mesh = init_device_mesh(
            "cuda",
            (replicate_size, shard_size),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
        logger.info(
            "FSDPPolicy: HSDP mesh dp_replicate=%d × dp_shard=%d",
            replicate_size,
            shard_size,
        )
        return mesh

    # ------------------------------------------------------------------
    # Helpers (ported from FSDPBackend)
    # ------------------------------------------------------------------

    @staticmethod
    def _maybe_dtensor_to_tensor(value: Any) -> Any:
        if hasattr(value, "full_tensor") and callable(getattr(value, "full_tensor")):
            try:
                return value.full_tensor()
            except Exception:
                return value
        return value

    @staticmethod
    def _to_cpu_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        converted: Dict[str, Any] = {}
        for key, value in state_dict.items():
            tensor_or_obj = FSDPPolicy._maybe_dtensor_to_tensor(value)
            if isinstance(tensor_or_obj, torch.Tensor):
                converted[key] = tensor_or_obj.detach().cpu()
            else:
                converted[key] = tensor_or_obj
        return converted

    @staticmethod
    def _current_rank() -> int:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
        return 0


# ----------------------------------------------------------------------
# Module-level helpers (reused by LoRAPolicy)
# ----------------------------------------------------------------------


def _infer_device(model: nn.Module) -> torch.device:
    """First non-meta parameter's device, else current cuda, else cpu."""
    for param in model.parameters():
        if param.is_meta:
            continue
        return param.device
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def _filter_lora_state(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Substring-match fallback when peft isn't available or attached."""
    return {k: v for k, v in state_dict.items() if "lora" in str(k).lower()}


def _extract_peft_lora_state(model: nn.Module) -> Dict[str, Any]:
    """peft-aware adapter-only state extractor. Returns ``{}`` when peft
    is missing or the model has no peft adapters registered.
    """
    try:
        from peft.utils import get_peft_model_state_dict
    except Exception:
        return {}

    base_model = model.module if hasattr(model, "module") else model
    adapter_names: list[str] = []
    if hasattr(base_model, "peft_config"):
        adapter_names = list(base_model.peft_config.keys())
    if not adapter_names:
        adapter_names = [getattr(base_model, "active_adapter", "default")]

    lora_state: Dict[str, Any] = {}
    for adapter_name in adapter_names:
        lora_state.update(get_peft_model_state_dict(base_model, adapter_name=adapter_name))
    return lora_state


__all__ = [
    "FSDPPolicyConfig",
    "FSDPPolicy",
    "_filter_lora_state",
    "_extract_peft_lora_state",
]
