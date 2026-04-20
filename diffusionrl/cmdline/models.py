"""Cmdline adapters for model bundle runtime config."""

from __future__ import annotations

from typing import Any

from diffusionrl.cmdline.construction import build_component_init_payload_from_args
from diffusionrl.cmdline.registry import register_cmdline_config_parser
from diffusionrl.config import ModelSpec
from diffusionrl.construction import ComponentInitPayload
from diffusionrl.models.config import ModelBundleConfig
from diffusionrl.models.flux import FluxModelBundleConfig
from diffusionrl.models.registry import MODEL_BUNDLE_COMPONENT_FAMILY
from diffusionrl.utils.dtypes import parse_torch_dtype


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
        lora_target_modules=(list(args.training.lora_target_modules) if args.training.lora_target_modules else None),
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
    return FluxModelBundleConfig(**build_model_bundle_config_from_args(args, model_spec=model_spec).__dict__)


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
