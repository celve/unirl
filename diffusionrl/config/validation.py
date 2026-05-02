"""Shared validation helpers for registered component configs.

Two flavors of validator live here:

- **Per-field helpers** (e.g. :func:`validate_precision_type`) are called from
  individual ``__post_init__`` bodies so every dataclass that owns the same
  kind of field validates it the same way.
- **Cross-component validators** (``validate_weight_sync_contract``,
  ``validate_offload_contract``, ...) take the full ``cfg`` and enforce
  rules that span multiple resolved sections. They are invoked from
  ``train.py`` after ``validate(cfg)`` has materialized every registered
  leaf.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

import torch
from omegaconf import DictConfig

from diffusionrl.config.require import require
from diffusionrl.utils.dtypes import parse_torch_dtype

logger = logging.getLogger(__name__)


class PrecisionName(str, Enum):
    """Canonical precision aliases accepted by config fields."""

    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"


_CANONICAL_BY_DTYPE = {
    torch.bfloat16: PrecisionName.BF16,
    torch.float16: PrecisionName.FP16,
    torch.float32: PrecisionName.FP32,
}


def validate_precision_type(value: Any, *, field: str) -> str:
    """Return the canonical precision alias (``bf16``/``fp16``/``fp32``).

    Delegates alias expansion to ``parse_torch_dtype`` so all precision fields
    accept the same inputs (``bf16``/``bfloat16``, ``fp16``/``float16``/``half``,
    ``fp32``/``float32``/``float``) and raise the same ``ValueError`` on unknown
    names. Caller supplies ``field`` for error-message attribution.
    """
    dtype = parse_torch_dtype(value, field_name=field)
    return _CANONICAL_BY_DTYPE[dtype].value


_SGLANG_ENGINE_TARGET_SUFFIX = "SGLangRolloutEngine"
_FSDP_ENGINE_TARGET_SUFFIX = "FSDPSamplingEngine"
_TENSOR_SYNC_SUFFIXES = frozenset({"UpdateWeightFromTensor", "UpdateWeightFromDistributed"})


def is_direct_sampling(cfg: DictConfig) -> bool:
    """Training-actor-sampling mode is derived from the selected engine.

    ``rollout/engine: fsdp`` wires ``_target_=FSDPSamplingEngine``, which is
    exactly the case where the training actor samples directly on its own
    GPU (no dedicated rollout service).
    """
    target = str(cfg.rollout.engine.get("_target_") or "")
    return target.endswith(_FSDP_ENGINE_TARGET_SUFFIX)


def validate_dynamic_dotpaths(cfg: DictConfig) -> None:
    """Fail-fast import of every dynamic dotpath the driver will later resolve."""
    from diffusionrl.utils import load_function

    dotpath = str(cfg.run.data_source_dotpath or "").strip()
    require(
        bool(dotpath), f"cfg.run.data_source_dotpath must be a non-empty dotpath; got {cfg.run.data_source_dotpath!r}"
    )
    try:
        load_function(dotpath)
    except Exception as exc:
        raise ValueError(f"cfg.run.data_source_dotpath={dotpath!r} failed to import: {exc}") from exc


def validate_training_actor_sampling_mode(cfg: DictConfig) -> None:
    """Direct-sampling mode requires a backend that supports it."""
    if not is_direct_sampling(cfg):
        return
    from diffusionrl.ray.group.train import _backend_name_from_cfg
    from diffusionrl.training.types import resolve_train_backend_capabilities

    backend_name = _backend_name_from_cfg(cfg)
    caps = resolve_train_backend_capabilities(backend_name)
    require(
        caps.supports_training_actor_sampling,
        f"direct_sampling mode (rollout/engine=fsdp) requires a backend with supports_training_actor_sampling=True; got backend={backend_name!r}",
    )


def validate_training_batch_geometry(cfg: DictConfig) -> None:
    """Cross-section: training plan's global batch size must divide by DP sizes.

    ``dp_size`` is ``Optional[int]`` on ``TrainTopology``; ``None`` means
    "derive from ``dist.get_world_size()`` at runtime" and is not checkable
    at cfg time.
    """
    global_batch = int(cfg.training.plan.global_batch_size)
    raw_dp_size = cfg.training.topology.dp_size
    dp_replicate_size = int(cfg.training.topology.dp_replicate_size)
    if raw_dp_size is not None:
        dp_size = int(raw_dp_size)
        require(
            dp_size <= 0 or global_batch % dp_size == 0,
            f"cfg.training.plan.global_batch_size ({global_batch}) must be divisible by cfg.training.topology.dp_size ({dp_size})",
        )
    require(
        dp_replicate_size <= 0 or global_batch % dp_replicate_size == 0,
        f"cfg.training.plan.global_batch_size ({global_batch}) must be divisible by cfg.training.topology.dp_replicate_size ({dp_replicate_size})",
    )


def validate_sampling_chunk_geometry(cfg: DictConfig) -> None:
    """Cross-section: SGLang chunk size must keep every chunk K-aligned.

    SGLang's ``_deexpand_prompts`` (``samplers/sglang/engine.py``) folds
    K-major prompts back into ``unique_prompts + num_outputs_per_prompt=K``
    to unlock the K-aware fast path (shared prompt encoding, KV-cache reuse).
    The fold succeeds only when every chunk emitted by
    ``chunked_engine_generate`` is K-aligned; misaligned chunks silently
    collapse to ``K=1`` and the fast path is lost without any error surface.

    Skip when chunking can't fire (``fwd is None``), the engine isn't
    SGLang, ``K <= 1``, or ``fwd >= prompts_per_rollout * K`` (a conservative
    single-actor upper bound — the chunker short-circuits to one call so
    fold sees the full K-major layout). Otherwise require ``fwd >= K`` and
    ``fwd % K == 0``. FSDP paths treat the expanded list as a flat batch
    and have no K-aware fast path to lose.
    """
    fwd = cfg.rollout.plan.forward_batch_size
    if fwd is None:
        return

    engine_target = str(cfg.rollout.engine.get("_target_") or "")
    if not engine_target.endswith(_SGLANG_ENGINE_TARGET_SUFFIX):
        return

    samples_per_prompt = int(cfg.algorithm.samples_per_prompt)
    if samples_per_prompt <= 1:
        return

    prompts_per_rollout = int(cfg.algorithm.prompts_per_rollout)
    fwd = int(fwd)
    if fwd >= prompts_per_rollout * samples_per_prompt:
        return  # chunked_engine_generate fast-paths to one engine.generate call

    fix_hint = (
        f"set cfg.rollout.plan.forward_batch_size to a multiple of "
        f"cfg.algorithm.samples_per_prompt ({samples_per_prompt}) that is >= K, "
        "or leave it unset."
    )
    require(
        fwd >= samples_per_prompt,
        f"sglang rollout engine: forward_batch_size ({fwd}) < samples_per_prompt "
        f"({samples_per_prompt}). Sub-K chunks split K-groups across SGLang requests "
        "and the client silently falls back to K=1, losing the K-aware fast path; " + fix_hint,
    )
    require(
        fwd % samples_per_prompt == 0,
        f"sglang rollout engine: forward_batch_size ({fwd}) is not a multiple of "
        f"samples_per_prompt ({samples_per_prompt}). Misaligned chunks cross K-group "
        "boundaries and the SGLang client silently falls back to K=1, losing the "
        "K-aware fast path; " + fix_hint,
    )


def validate_weight_sync_contract(cfg: DictConfig) -> None:
    """Weight-sync section presence + variant must match rollout engine."""
    has_sync = cfg.get("sync") is not None
    is_direct = is_direct_sampling(cfg)
    require(
        not is_direct or not has_sync,
        f"direct_sampling mode (rollout/engine=fsdp) forbids a sync section; got cfg.sync={cfg.get('sync')!r}",
    )
    require(
        is_direct or has_sync,
        "dedicated-rollout mode (rollout/engine=sglang) requires a sync variant; got no sync section",
    )
    if has_sync:
        sync_target = str(cfg.sync.get("_target_") or "")
        sync_name = sync_target.rsplit(".", 1)[-1]
        if sync_name in _TENSOR_SYNC_SUFFIXES:
            engine_target = str(cfg.rollout.engine.get("_target_") or "")
            require(
                engine_target.endswith(_SGLANG_ENGINE_TARGET_SUFFIX),
                f"sync={sync_name} requires the sglang rollout engine; got rollout.engine._target_={engine_target!r}",
            )


def validate_rollout_layout(cfg: DictConfig) -> None:
    """Multi-GPU colocated rollout requires the sglang engine."""
    num_gpus_per_actor = int(cfg.placement.num_rollout_gpus_per_actor)
    if bool(cfg.placement.colocate) and num_gpus_per_actor > 1:
        engine_target = str(cfg.rollout.engine.get("_target_") or "")
        require(
            engine_target.endswith(_SGLANG_ENGINE_TARGET_SUFFIX),
            f"multi-GPU colocated rollout (num_rollout_gpus_per_actor={num_gpus_per_actor}, colocate=True) requires the sglang engine; got rollout.engine._target_={engine_target!r}",
        )


def validate_offload_contract(cfg: DictConfig) -> None:
    """Direct-sampling mode forbids GPU offloading (there is no paired rollout actor)."""
    if not is_direct_sampling(cfg):
        return
    require(
        not bool(cfg.training.execution.offload_train),
        "direct_sampling mode (rollout/engine=fsdp) is incompatible with cfg.training.execution.offload_train=True",
    )
    require(
        not bool(cfg.training.execution.offload_rollout),
        "direct_sampling mode (rollout/engine=fsdp) is incompatible with cfg.training.execution.offload_rollout=True",
    )


def validate_lora_target_modules(cfg: DictConfig) -> None:
    """Materialize ``cfg.model.lora_target_modules`` from the bundle's class default.

    When LoRA is requested but no explicit target list was supplied, resolve the
    model class via ``cfg.model._target_`` and call its
    ``default_lora_target_modules()`` classmethod. Mutates ``cfg.model`` in
    place (``ModelBundleConfig`` is registered ``mutable=True``) so PEFT (training
    side) and SGLang ``ServerArgs.lora_target_modules`` (rollout side) see the
    same list. Without this materializer, PEFT injects LoRA into a model-class
    default subset while SGLang receives ``None`` and wraps every linear layer,
    producing a wall of "LoRA adapter None does not contain the weights for layer ..."
    warnings and silently disabling LoRA on unmatched layers.

    Priority: explicit ``cfg.model.lora_target_modules`` > model class default
    > ``None`` (warn).
    """
    if not bool(cfg.model.get("use_lora", False)):
        return
    if cfg.model.get("lora_target_modules") is not None:
        return

    target_dotpath = str(cfg.model.get("_target_") or "")
    if not target_dotpath:
        return

    try:
        from diffusionrl.utils.misc import load_function

        model_cls = load_function(target_dotpath)
    except (ImportError, AttributeError, KeyError, ValueError) as exc:
        logger.debug(
            "Could not resolve model class %r for LoRA target lookup: %s",
            target_dotpath,
            exc,
        )
        return

    fn = getattr(model_cls, "default_lora_target_modules", None)
    if not callable(fn):
        return

    try:
        resolved = fn()
    except (TypeError, NotImplementedError) as exc:
        logger.warning(
            "Model class %s.default_lora_target_modules() raised %s; "
            "falling back to None (SGLang will wrap every linear layer).",
            model_cls.__name__,
            exc,
        )
        return

    if resolved is None:
        logger.warning(
            "%s.default_lora_target_modules() returned None and no explicit "
            "cfg.model.lora_target_modules was provided; SGLang will wrap every "
            "linear layer and silently disable LoRA on unmatched ones.",
            model_cls.__name__,
        )
        return
    if not isinstance(resolved, (list, tuple)) or not resolved:
        logger.warning(
            "%s.default_lora_target_modules() returned %r; expected a non-empty list. Falling back to None.",
            model_cls.__name__,
            resolved,
        )
        return

    materialised = [str(item) for item in resolved]
    cfg.model.lora_target_modules = materialised
    logger.info(
        "LoRA target modules materialised from %s.default_lora_target_modules(): %s",
        model_cls.__name__,
        materialised,
    )


__all__ = [
    "PrecisionName",
    "is_direct_sampling",
    "validate_dynamic_dotpaths",
    "validate_lora_target_modules",
    "validate_offload_contract",
    "validate_precision_type",
    "validate_rollout_layout",
    "validate_sampling_chunk_geometry",
    "validate_training_actor_sampling_mode",
    "validate_training_batch_geometry",
    "validate_weight_sync_contract",
]
