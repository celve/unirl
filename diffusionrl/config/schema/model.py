"""Model-domain config builders for actor initialization."""

from __future__ import annotations

from typing import Any, Dict, Optional


def build_model_config(args) -> Dict[str, Any]:
    """Build model config consumed by training and inference actors."""
    return {
        "model_path": args.model_path,
        "pretrained_model_saved_path": args.pretrained_model_saved_path,
        "use_lora": bool(getattr(args, "use_lora", False)),
        "lora_rank": int(getattr(args, "lora_rank", 16)),
        "lora_alpha": int(getattr(args, "lora_alpha", 16)),
        "lora_target_modules": get_lora_target_modules(args),
        "use_gradient_checkpointing": bool(
            getattr(args, "use_gradient_checkpointing", False)
        ),
    }


def get_lora_target_modules(args) -> Optional[Any]:
    """Get LoRA target modules from args if provided."""
    return getattr(args, "lora_target_modules", None)
