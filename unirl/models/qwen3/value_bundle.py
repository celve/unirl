"""Bundle and checkpoint contract for the independent Qwen3 value critic."""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Tuple

import torch

from unirl.models.types.bundle import Bundle
from unirl.models.types.meta_init import capture_init_state, finalize_meta_init, stamp_init_state_restore
from unirl.utils.dtypes import parse_torch_dtype

from .config import Qwen3ValueConfig
from .value import (
    Qwen3TokenValueModel,
    configure_value_trainable_parameters,
    trainable_value_parameter_summary,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Qwen3ValueCheckpointManifest:
    """Safetensors shards and keys discovered in a local value artifact."""

    path: str
    shards: Tuple[str, ...]
    keys: FrozenSet[str]
    key_to_shard: Dict[str, str]


def inspect_value_checkpoint(path: str) -> Qwen3ValueCheckpointManifest:
    """Validate and index a local HF-style safetensors value checkpoint."""

    from safetensors import safe_open

    path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Qwen3 value checkpoint must be a local directory, got {path!r}")
    shards = tuple(sorted(glob.glob(os.path.join(path, "*.safetensors"))))
    if not shards:
        raise FileNotFoundError(f"Qwen3 value checkpoint has no *.safetensors files under {path!r}")

    key_to_shard: Dict[str, str] = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in key_to_shard:
                    raise ValueError(
                        f"Qwen3 value checkpoint key {key!r} appears in both {key_to_shard[key]!r} and {shard!r}"
                    )
                key_to_shard[key] = shard
    keys = frozenset(key_to_shard)
    if not any(key.startswith("model.") for key in keys):
        raise ValueError("Qwen3 value checkpoint contains no decoder weights under the required 'model.' prefix")
    return Qwen3ValueCheckpointManifest(
        path=path,
        shards=shards,
        keys=keys,
        key_to_shard=key_to_shard,
    )


def _read_checkpoint_tensor(manifest: Qwen3ValueCheckpointManifest, key: str) -> torch.Tensor:
    from safetensors import safe_open

    shard = manifest.key_to_shard.get(key)
    if shard is None:
        raise KeyError(key)
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def _validate_checkpoint_keys(
    model: Qwen3TokenValueModel,
    manifest: Qwen3ValueCheckpointManifest,
    *,
    allow_random_value_init: bool,
    require_all_parameters: bool = False,
) -> None:
    required = {
        name
        for name, parameter in model.named_parameters()
        if (require_all_parameters or parameter.requires_grad)
        and not (allow_random_value_init and name == "value_head.weight")
    }
    missing = sorted(required - manifest.keys)
    if missing:
        scope = "model" if require_all_parameters else "trainable critic"
        raise ValueError(
            f"Qwen3 value checkpoint is missing {scope} weight(s): {missing[:16]}{' ...' if len(missing) > 16 else ''}"
        )
    if not allow_random_value_init and "value_head.weight" not in manifest.keys:
        raise ValueError(
            "Qwen3 value checkpoint is missing required 'value_head.weight'. "
            "Only mechanical smoke tests may set allow_random_value_init=True."
        )


class Qwen3ValueBundle(Bundle):
    """Independent Qwen3 decoder/value-head weights plus tokenizer."""

    def __init__(
        self,
        *,
        transformer: Qwen3TokenValueModel,
        tokenizer: Any,
        dtype: torch.dtype,
        device: torch.device,
        pretrained_path: str,
        trainable_parameter_names: Tuple[str, ...],
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.device = device
        self.pretrained_path = pretrained_path
        self.trainable_parameter_names = trainable_parameter_names

    @classmethod
    def from_config(cls, config: Qwen3ValueConfig) -> "Qwen3ValueBundle":
        """Load or meta-initialize a token-value artifact."""

        from transformers import AutoConfig, AutoModel, AutoTokenizer

        manifest = inspect_value_checkpoint(config.pretrained_value_ckpt_path)
        if "value_head.weight" not in manifest.keys and not config.allow_random_value_init:
            raise ValueError(
                "Qwen3 value checkpoint is missing required 'value_head.weight'. "
                "Only mechanical smoke tests may set allow_random_value_init=True."
            )
        path = manifest.path
        tokenizer_path = config.tokenizer_ckpt_path or path
        device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)
        dtype = parse_torch_dtype(config.model_precision, field_name="model_precision")
        hf_config = AutoConfig.from_pretrained(path, trust_remote_code=bool(config.trust_remote_code))

        if config.meta_init_transformer:
            from accelerate import init_empty_weights

            model_kwargs: Dict[str, Any] = {"trust_remote_code": bool(config.trust_remote_code)}
            if config.attn_implementation:
                # Match the eager ``from_pretrained`` path: HF must receive the
                # requested backend before it constructs the attention modules.
                model_kwargs["attn_implementation"] = str(config.attn_implementation)
            with init_empty_weights(include_buffers=False):
                decoder = AutoModel.from_config(hf_config, **model_kwargs)
                transformer = Qwen3TokenValueModel(model=decoder)
            init_state = capture_init_state(transformer)
            stamp_init_state_restore(transformer)
            trainable_names = configure_value_trainable_parameters(transformer)
            _validate_checkpoint_keys(
                transformer,
                manifest,
                allow_random_value_init=False,
                # The generic sharded loader materializes missing meta tensors
                # without initializing them.  Every decoder/head parameter must
                # therefore exist in a production meta-init artifact, including
                # frozen attention, embeddings, routers, and norms.
                require_all_parameters=True,
            )
            transformer = finalize_meta_init(transformer, dtype=dtype)
        else:
            load_kwargs: Dict[str, Any] = {}
            if config.attn_implementation:
                load_kwargs["attn_implementation"] = str(config.attn_implementation)
            decoder = AutoModel.from_pretrained(
                path,
                torch_dtype=dtype,
                trust_remote_code=bool(config.trust_remote_code),
                **load_kwargs,
            ).to(device)
            transformer = Qwen3TokenValueModel(model=decoder).to(device=device, dtype=dtype)
            if "value_head.weight" in manifest.keys:
                value_head = _read_checkpoint_tensor(manifest, "value_head.weight")
                if tuple(value_head.shape) != tuple(transformer.value_head.weight.shape):
                    raise ValueError(
                        "Qwen3 value checkpoint has incompatible value_head.weight shape: "
                        f"expected {tuple(transformer.value_head.weight.shape)}, "
                        f"got {tuple(value_head.shape)}"
                    )
                with torch.no_grad():
                    transformer.value_head.weight.copy_(
                        value_head.to(
                            device=transformer.value_head.weight.device,
                            dtype=transformer.value_head.weight.dtype,
                        )
                    )
            elif not config.allow_random_value_init:
                raise ValueError(
                    "Qwen3 value checkpoint is missing required 'value_head.weight'. "
                    "Only mechanical smoke tests may set allow_random_value_init=True."
                )
            trainable_names = configure_value_trainable_parameters(transformer)
            _validate_checkpoint_keys(
                transformer,
                manifest,
                allow_random_value_init=bool(config.allow_random_value_init),
                # ``from_pretrained`` otherwise initializes absent decoder
                # tensors silently.  The smoke escape hatch permits only the
                # scalar head to be random, never a partially random critic.
                require_all_parameters=True,
            )

        decoder = transformer.model
        if config.use_gradient_checkpointing:
            if not hasattr(decoder, "gradient_checkpointing_enable"):
                raise TypeError(
                    f"Qwen3 value decoder {type(decoder).__name__} does not expose gradient_checkpointing_enable"
                )
            decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=bool(config.trust_remote_code),
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        summary = trainable_value_parameter_summary(transformer)
        logger.info(
            "Qwen3 value critic trainable set: %d tensors / %d parameters: %s",
            summary["tensors"],
            summary["parameters"],
            summary["names"],
        )
        bundle = cls(
            transformer=transformer,
            tokenizer=tokenizer,
            dtype=dtype,
            device=device,
            pretrained_path=path,
            trainable_parameter_names=trainable_names,
        )
        if config.meta_init_transformer:
            # Value artifacts are HF-style safetensors at the checkpoint root;
            # the generic FSDP/VeOmni post-wrap loader sees wrapper keys
            # ``model.*`` and ``value_head.weight`` directly.
            bundle._transformer_weights_path = path
            bundle._meta_init_state = init_state
        return bundle


__all__ = [
    "Qwen3ValueBundle",
    "Qwen3ValueCheckpointManifest",
    "inspect_value_checkpoint",
]
