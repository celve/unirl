"""Cmdline adapters for model bundle runtime config."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from diffusionrl.cmdline.construction import build_component_init_payload_from_args
from diffusionrl.cmdline.registry import register_cmdline_config_parser
from diffusionrl.config import ModelSpec
from diffusionrl.construction import ComponentInitPayload
from diffusionrl.models.config import ModelBundleConfig
from diffusionrl.models.flux import FluxModelBundleConfig
from diffusionrl.models.registry import (
    MODEL_BUNDLE_COMPONENT_FAMILY,
    resolve_model_class,
)
from diffusionrl.utils.dtypes import parse_torch_dtype

logger = logging.getLogger(__name__)


def _resolve_lora_target_modules(
    args: Any,
    *,
    model_spec: ModelSpec,
) -> Optional[List[str]]:
    """Resolve LoRA target modules with CLI > model-class default > None.

    Single source of truth for LoRA target selection shared by PEFT (training
    side) and SGLang ``ServerArgs.lora_target_modules`` (rollout side).  When
    the CLI does not pass ``--training.lora-target-modules`` and the model
    class declares a ``default_lora_target_modules()`` class method, the
    declared list is materialised into ``ModelBundleConfig`` so downstream
    consumers (``EngineConfig`` → SGLang) see a non-None value.
    """
    cli = args.training.lora_target_modules
    if cli:
        return list(cli)

    # Only bother probing the model class when LoRA is actually used.
    if not bool(args.training.use_lora):
        return None

    try:
        model_cls = resolve_model_class(model_spec.model_dotpath)
    except Exception as exc:  # pragma: no cover - diagnostic path
        logger.debug(
            "Could not resolve model class %r for LoRA target lookup: %s",
            model_spec.model_dotpath,
            exc,
        )
        return None

    fn = getattr(model_cls, "default_lora_target_modules", None)
    if not callable(fn):
        return None

    try:
        resolved = fn()
    except Exception as exc:  # pragma: no cover - diagnostic path
        logger.warning(
            "Model class %s.default_lora_target_modules() raised %s; "
            "falling back to None (SGLang will wrap every linear layer).",
            model_cls.__name__,
            exc,
        )
        return None

    if resolved is None:
        return None
    if not isinstance(resolved, (list, tuple)) or not resolved:
        logger.warning(
            "%s.default_lora_target_modules() returned %r; expected a non-empty "
            "list. Falling back to None.",
            model_cls.__name__,
            resolved,
        )
        return None

    materialised = [str(item) for item in resolved]
    logger.info(
        "LoRA target modules materialised from %s.default_lora_target_modules(): %s",
        model_cls.__name__,
        materialised,
    )
    return materialised


@register_cmdline_config_parser(ModelBundleConfig)
def build_model_bundle_config_from_args(
    args: Any,
    *,
    model_spec: ModelSpec,
) -> ModelBundleConfig:
    """Build ModelBundleConfig from framework args."""
    return ModelBundleConfig(
        pretrained_model_ckpt_path=args.model.pretrained_model_ckpt_path,
        vae_ckpt_path=args.model.vae_ckpt_path,
        text_encoder_ckpt_path=args.model.text_encoder_ckpt_path,
        use_lora=bool(args.training.use_lora),
        lora_rank=int(args.training.lora_rank),
        lora_alpha=int(args.training.lora_alpha),
        lora_target_modules=_resolve_lora_target_modules(args, model_spec=model_spec),
        use_gradient_checkpointing=bool(args.training.use_gradient_checkpointing),
        model_precision=parse_torch_dtype(
            args.precision.model_precision,
            field_name="precision.model_precision",
        ),
    )


@register_cmdline_config_parser(FluxModelBundleConfig)
def build_flux_model_bundle_config_from_args(
    args: Any,
    *,
    model_spec: ModelSpec,
) -> FluxModelBundleConfig:
    sde_type = str(args.sampling.sde_type or "").strip().lower()
    valid_sde_types = ("dance", "flow", "dpm2", "")
    if sde_type not in valid_sde_types:
        raise ValueError(
            f"Unknown sampling.sde_type={args.sampling.sde_type!r} for model_type='flux'. "
            f"Valid options: {', '.join(t for t in valid_sde_types if t)}."
        )
    return FluxModelBundleConfig(
        **build_model_bundle_config_from_args(args, model_spec=model_spec).__dict__
    )


def build_model_bundle_init_payload_from_args(
    args: Any,
    *,
    model_spec: ModelSpec,
) -> ComponentInitPayload:
    return build_component_init_payload_from_args(
        component_family=MODEL_BUNDLE_COMPONENT_FAMILY,
        identifier=model_spec.model_dotpath,
        args=args,
        parser_kwargs={"model_spec": model_spec},
    )


__all__ = [
    "build_model_bundle_init_payload_from_args",
]
