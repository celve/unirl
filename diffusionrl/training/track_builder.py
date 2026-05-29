"""Non-Ray helpers for multi-track ``TrainActor`` construction.

The TrainActor's per-track init body lives here so it can be unit-tested
without importing Ray (the actor class lives in ``ray/train_actor.py``).

Public surface:

- :func:`build_training_tracks` — iterates ``cfg.training.tracks.<name>``
  and builds per-track ``TrainTrack`` artefacts (stage, EMA handle,
  optimizer, scheduler, algorithm).  Structural mutations (LoRA, NFT,
  mirror shadows) are applied via inject functions; FSDP wrapping via
  ``fsdp_wrap``; deferred init drained via ``apply_deferred_ops``.
- :func:`_build_algorithm_for_track` — instantiates one track's
  :class:`StageAlgorithm` from its cfg node.
- :func:`_resolve_primary_track` — picks the weight-sync-target track.
- :func:`_detect_lora_on_model` — runtime PEFT-adapter detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import hydra.utils
import torch
from omegaconf import DictConfig, OmegaConf

from diffusionrl.training.ema import EMA, make_decay_fn
from diffusionrl.training.fsdp_utils import trainable_params
from diffusionrl.training.inject import (
    apply_deferred_ops,
    fsdp_wrap,
    inject_lora,
    inject_mirror,
    inject_nft,
)
from diffusionrl.training.shadow import Shadow
from diffusionrl.training.train_track import TrainTrack
from diffusionrl.types.sampling import get_diffusion_params

logger = logging.getLogger(__name__)


def _iter_optimizer_param_states(optimizer: object) -> object:
    """Yield per-parameter optimizer state dicts (AdamW-style)."""
    yield from optimizer.state.values()


def _detect_lora_on_model(model: object) -> bool:
    """Return True iff ``model`` has at least one PEFT adapter registered."""
    candidates = []
    seen: set[int] = set()
    cur = model
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        candidates.append(cur)
        cur = getattr(cur, "module", None) or getattr(cur, "base_model", None)
    for c in candidates:
        pc = getattr(c, "peft_config", None)
        if isinstance(pc, dict) and len(pc) > 0:
            return True
    return False


def _resolve_ar_sampling_temperature(cfg: DictConfig) -> Optional[float]:
    """Single source of truth for the AR rollout sampling temperature.

    Reads from ``cfg.rollout.engine.ar.temperature`` (composed PE engine)
    or ``cfg.rollout.engine.temperature`` (direct AR engine). Returns
    ``None`` when no AR engine is present so the algorithm falls back to
    its own default.
    """
    if cfg.get("rollout") is None:
        return None
    rollout_eng = cfg.rollout.get("engine") if hasattr(cfg.rollout, "get") else None
    if rollout_eng is None:
        return None
    ar_node = rollout_eng.get("ar") if hasattr(rollout_eng, "get") else None
    candidate = ar_node.get("temperature") if ar_node is not None else None
    if candidate is None:
        candidate = rollout_eng.get("temperature") if hasattr(rollout_eng, "get") else None
    return float(candidate) if candidate is not None else None


def _build_algorithm_for_track(
    alg_node: object,
    *,
    track_name: str,
    pipeline: object,
    stage: object,
    ema: Optional[EMA],
    sampling_params: Optional[Any] = None,
    ar_sampling_temperature: Optional[float] = None,
) -> object:
    """Instantiate one track's :class:`StageAlgorithm` from its cfg node.

    ``DiffusionNFT`` requires an ``EMA`` handle (raised cleanly here if
    absent).
    """
    from omegaconf import OmegaConf as _OmegaConf

    _CONTROL_KEYS = ("stage_attr", "params_attr", "conditions_cls")
    stage_attr = alg_node.get("stage_attr", track_name)
    resolved_stage = getattr(pipeline, stage_attr)
    params_attr = alg_node.get("params_attr", f"{track_name}_params")
    params_obj = getattr(pipeline, params_attr, None)
    conditions_cls_path = alg_node.get("conditions_cls")
    conditions_cls = hydra.utils.get_class(str(conditions_cls_path)) if conditions_cls_path else None
    clean_node = _OmegaConf.create({k: v for k, v in alg_node.items() if k not in _CONTROL_KEYS})
    inject_kwargs: Dict[str, object] = {"stage": resolved_stage}
    if params_obj is not None:
        inject_kwargs["params"] = params_obj
    if conditions_cls is not None:
        inject_kwargs["conditions_cls"] = conditions_cls

    target = str(alg_node.get("_target_") or "")
    if (
        "params" not in inject_kwargs
        and target.endswith((".DiffusionGRPO", ".DiffusionDPPO", ".DiffusionNFT"))
        and alg_node.get("params") is None
        and sampling_params is not None
    ):
        inject_kwargs["params"] = sampling_params

    if target.endswith(".ARGRPO") and ar_sampling_temperature is not None:
        inject_kwargs["sampling_temperature"] = ar_sampling_temperature

    if target.endswith(".DiffusionNFT"):
        if ema is None:
            raise RuntimeError(
                f"TrainActor: algorithms.{track_name}._target_={target!r} "
                f"requires ema_lora to be configured on the track. "
                f"Set training.tracks.{track_name}.ema_lora with the "
                f"appropriate NFT EMA settings."
            )
        inject_kwargs["nft_lora_policy"] = ema

    return hydra.utils.instantiate(clean_node, _convert_="object", **inject_kwargs)


@dataclass
class _PerTrackArtifacts:
    """Per-track training artefacts built at actor init.

    ``tracks`` holds :class:`TrainTrack` instances (stage + ema +
    optimizer + scheduler + algorithm).  ``pipelines``, ``bundles``,
    and ``use_lora`` are kept as side dicts for the actor's non-training
    needs (inference, export, model access).
    """

    tracks: Dict[str, TrainTrack] = field(default_factory=dict)
    pipelines: Dict[str, object] = field(default_factory=dict)
    bundles: Dict[str, object] = field(default_factory=dict)
    use_lora: Dict[str, bool] = field(default_factory=dict)


def build_training_tracks(
    cfg: DictConfig,
    *,
    device: torch.device,
    rank: int,
) -> _PerTrackArtifacts:
    """Build per-track ``TrainTrack`` artefacts.

    For each entry in ``cfg.training.tracks.<name>``:

    1. Build pipeline + bundle from ``track_cfg.model``.
    2. Apply structural mutations via inject functions (LoRA / NFT /
       mirror) based on named optional fields on the track config.
    3. FSDP wrap via ``fsdp_wrap``.
    4. Materialize bundle weights.
    5. Drain deferred ops (LoRA reset, shadow sync).
    6. Build EMA handle from Shadow (if any inject returned one).
    7. Per-track LoRA detection.
    8. Per-track algorithm.
    9. Per-track optimizer + scheduler.
    10. Per-track ``micro_batch_size``.
    """
    from diffusionrl.config.instantiate import build, materialize
    from diffusionrl.training.factories import build_lr_scheduler, build_optimizer

    artefacts = _PerTrackArtifacts()
    materialize_cfg = cfg.training.get("materialize")
    with_aux: tuple = ()
    if materialize_cfg is not None:
        with_aux = tuple(materialize_cfg.get("with_aux", []) or ())

    shared_micro_batch_size = int(cfg.training.plan.micro_batch_size)
    diffusion_params = get_diffusion_params(cfg.sampling) if cfg.get("sampling") is not None else None
    sde_strategy_cfg = diffusion_params.sde_strategy if diffusion_params is not None else None
    ar_sampling_temperature = _resolve_ar_sampling_temperature(cfg)

    for track_name, track_cfg in cfg.training.tracks.items():
        # 1. Pipeline.
        deps: Dict[str, object] = {}
        if sde_strategy_cfg is not None:
            target_path = OmegaConf.select(track_cfg.model, "_target_") or ""
            if target_path:
                import inspect

                target_callable = hydra.utils.get_method(target_path)
                sig = inspect.signature(target_callable)
                if "strategy" in sig.parameters:
                    deps["strategy"] = build(sde_strategy_cfg)
        pipeline = build(track_cfg.model, **deps)
        bundle = pipeline.bundle

        # 2. Structural mutations via inject functions.
        source_slot = str(track_cfg.source_stage_attr)
        source_stage = getattr(pipeline, source_slot)
        model = source_stage.trainable_module()
        shadow: Optional[Shadow] = None

        lora_cfg = getattr(track_cfg, "lora", None)
        ema_lora_cfg = getattr(track_cfg, "ema_lora", None)
        ema_cfg = getattr(track_cfg, "ema", None)

        if lora_cfg is not None and ema_lora_cfg is not None:
            raise ValueError(
                f"Track {track_name!r}: lora and ema_lora are mutually "
                f"exclusive (both inject LoRA adapters). Use ema_lora for "
                f"NFT-style adapter EMA, or lora for plain LoRA."
            )

        if ema_lora_cfg is not None:
            if isinstance(ema_lora_cfg, DictConfig):
                ema_lora_cfg = materialize(ema_lora_cfg)
            shadow = inject_nft(
                model,
                rank=ema_lora_cfg.rank,
                alpha=ema_lora_cfg.alpha,
                target_modules=tuple(ema_lora_cfg.target_modules),
                default=ema_lora_cfg.default_adapter,
                shadow=ema_lora_cfg.shadow_adapter,
                dropout=ema_lora_cfg.dropout,
                bias=ema_lora_cfg.bias,
                task_type=ema_lora_cfg.task_type,
            )
        elif lora_cfg is not None:
            if isinstance(lora_cfg, DictConfig):
                lora_cfg = materialize(lora_cfg)
            inject_lora(
                model,
                rank=lora_cfg.rank,
                alpha=lora_cfg.alpha,
                target_modules=tuple(lora_cfg.target_modules),
                dropout=lora_cfg.dropout,
                bias=lora_cfg.bias,
                task_type=lora_cfg.task_type,
            )

        if ema_cfg is not None:
            if isinstance(ema_cfg, DictConfig):
                ema_cfg = materialize(ema_cfg)
            shadow = inject_mirror(model, prefix=ema_cfg.shadow_prefix)

        # 3. FSDP wrap.
        fsdp_cfg = getattr(track_cfg, "fsdp", None)
        if fsdp_cfg is not None:
            if isinstance(fsdp_cfg, DictConfig):
                fsdp_cfg = materialize(fsdp_cfg)
            fsdp_wrap(
                model,
                source_stage,
                param_dtype=fsdp_cfg.param_dtype,
                cpu_offload=fsdp_cfg.cpu_offload,
                mixed_precision=fsdp_cfg.mixed_precision,
                fsdp_mode=fsdp_cfg.fsdp_mode,
                reshard_after_forward=fsdp_cfg.reshard_after_forward,
                activation_checkpointing=fsdp_cfg.activation_checkpointing,
                use_torch_compile=fsdp_cfg.use_torch_compile,
            )

        # 4. Materialize weights.
        bundle_materialize = getattr(bundle, "materialize", None)
        if callable(bundle_materialize):
            bundle_materialize(device=device, with_aux=with_aux)
        elif with_aux:
            logger.info(
                "Rank %s: track %s bundle %s loads eagerly; ignoring with_aux=%s",
                rank,
                track_name,
                type(bundle).__name__,
                with_aux,
            )

        # 5. Drain deferred ops (LoRA reset, shadow sync).
        apply_deferred_ops(model)

        # 6. Build EMA handle.
        ema: Optional[EMA] = None
        if shadow is not None:
            active_ema_cfg = ema_lora_cfg or ema_cfg
            ema = EMA(
                shadow=shadow,
                decay_fn=make_decay_fn(active_ema_cfg),
                timing=active_ema_cfg.timing,
            )

        # 7. Per-track LoRA detection.
        use_lora = _detect_lora_on_model(model)
        if use_lora:
            logger.info(
                "Rank %s: LoRA detected on track %r (peft_config keys=%s)",
                rank,
                track_name,
                list(getattr(model, "peft_config", None) or {}),
            )

        # 8. Per-track algorithms.
        # ``algorithm_keys`` lets one track host multiple algorithms (HI3
        # shared-backbone). Absent / empty → fall back to [track_name]
        # (preserves single-algo PE / SD3 behavior).
        configured_keys = getattr(track_cfg, "algorithm_keys", None)
        if configured_keys:
            algorithm_keys: List[str] = [str(k) for k in configured_keys]
        else:
            algorithm_keys = [track_name]
        algorithms: Dict[str, object] = {}
        for alg_key in algorithm_keys:
            if alg_key not in cfg.algorithm.algorithms:
                raise ValueError(
                    f"TrainActor: cfg.algorithm.algorithms must define an entry for "
                    f"algorithm key {alg_key!r} (referenced by track {track_name!r}); "
                    f"got keys {sorted(cfg.algorithm.algorithms.keys())}."
                )
            algorithms[alg_key] = _build_algorithm_for_track(
                cfg.algorithm.algorithms[alg_key],
                track_name=alg_key,
                pipeline=pipeline,
                stage=source_stage,
                ema=ema,
                sampling_params=diffusion_params,
                ar_sampling_temperature=ar_sampling_temperature,
            )

        # 9. Per-track optimizer + scheduler.
        optimizer = build_optimizer(
            materialize(track_cfg.optimizer),
            params=list(trainable_params(model)),
            backend=None,
            actor=None,
        )
        scheduler = build_lr_scheduler(
            materialize(track_cfg.lr_scheduler),
            optimizer=optimizer,
            backend=None,
            actor=None,
        )

        # 10. Per-track micro_batch_size.
        override = track_cfg.get("plan_overrides")
        if override is not None and override.get("micro_batch_size") is not None:
            micro_bs = int(override.micro_batch_size)
        else:
            micro_bs = shared_micro_batch_size

        artefacts.tracks[track_name] = TrainTrack(
            stage=source_stage,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            algorithms=algorithms,
            micro_batch_size=micro_bs,
        )
        artefacts.pipelines[track_name] = pipeline
        artefacts.bundles[track_name] = bundle
        artefacts.use_lora[track_name] = use_lora

    return artefacts


def _resolve_primary_track(cfg: DictConfig) -> str:
    """Return ``cfg.training.primary_track`` or fall back to the first track key."""
    explicit = cfg.training.get("primary_track")
    if explicit:
        explicit_str = str(explicit)
        if explicit_str not in cfg.training.tracks:
            raise ValueError(
                f"cfg.training.primary_track={explicit_str!r} is not a key in "
                f"cfg.training.tracks (have {sorted(cfg.training.tracks.keys())})."
            )
        return explicit_str
    return next(iter(cfg.training.tracks.keys()))


__all__ = [
    "_PerTrackArtifacts",
    "_build_algorithm_for_track",
    "_detect_lora_on_model",
    "_iter_optimizer_param_states",
    "_resolve_primary_track",
    "build_training_tracks",
]
