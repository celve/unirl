"""Training-only Qwen3 token-value model and replay stage for SAO.

The critic is deliberately independent from the actor: it owns a Qwen3 decoder
without an LM head and a scalar value head, has its own optimizer/checkpoint,
and is never sent to rollout workers.  Values are aligned to the causal state
immediately before each generated action token.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from unirl.types.segments import TextSegment
from unirl.utils.dtypes import parse_torch_dtype

from .ar import _packed_replay_supported
from .conditions import Qwen3ARConditions
from .replay_layout import (
    build_packed_replay_layout,
    build_padded_replay_layout,
    pack_padded_token_outputs,
)

if TYPE_CHECKING:
    from .value_bundle import Qwen3ValueBundle


_DENSE_MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_MOE_EXPERT_PROJECTIONS = ("gate_up_proj", "down_proj")


def _is_value_projection_parameter(name: str) -> bool:
    """Whether ``name`` is an explicitly trainable critic MLP projection.

    The matcher is intentionally structural and narrow.  In particular it does
    not match MoE router ``mlp.gate``, attention output projections, shared
    experts, embeddings, or norms.
    """

    parts = name.split(".")
    for index, part in enumerate(parts):
        if part == "mlp" and index + 1 < len(parts) and parts[index + 1] in _DENSE_MLP_PROJECTIONS:
            return True
        if (
            part == "mlp"
            and index + 2 < len(parts)
            and parts[index + 1] == "experts"
            and parts[index + 2] in _MOE_EXPERT_PROJECTIONS
        ):
            return True
    return False


def configure_value_trainable_parameters(model: "Qwen3TokenValueModel") -> Tuple[str, ...]:
    """Apply and validate SAO's frozen-attention critic policy.

    Everything is frozen first.  Only the scalar value head plus dense MLP or
    MoE expert projections are re-enabled.  A configuration that matches no
    projection fails loudly; silently training only the tiny head would not be
    the intended SAO critic.
    """

    model.requires_grad_(False)
    trainable = []
    projection_names = []
    head_names = []
    for name, parameter in model.named_parameters():
        is_head = name.startswith("value_head.")
        is_projection = _is_value_projection_parameter(name)
        if is_head or is_projection:
            parameter.requires_grad_(True)
            trainable.append(name)
            if is_head:
                head_names.append(name)
            else:
                projection_names.append(name)

    if head_names != ["value_head.weight"]:
        raise RuntimeError(
            f"Qwen3 token-value critic must expose exactly one trainable scalar value_head.weight; found {head_names!r}"
        )
    if not projection_names:
        raise RuntimeError(
            "Qwen3 token-value critic freeze policy matched no MLP/expert projections; "
            "expected dense mlp.{gate,up,down}_proj or MoE "
            "mlp.experts.{gate_up,down}_proj parameters"
        )

    actual = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    expected = tuple(trainable)
    if actual != expected:
        raise RuntimeError(
            f"Qwen3 token-value critic has an unexpected trainable-parameter set: expected {expected!r}, got {actual!r}"
        )
    return actual


class Qwen3TokenValueModel(nn.Module):
    """Qwen3 decoder plus a bias-free scalar head, with no language-model head."""

    def __init__(self, *, model: nn.Module, hidden_size: Optional[int] = None) -> None:
        super().__init__()
        self.model = model
        self.config = getattr(model, "config", None)
        if hidden_size is None:
            hidden_size = getattr(self.config, "hidden_size", None)
        if hidden_size is None or int(hidden_size) <= 0:
            raise ValueError("Qwen3TokenValueModel requires hidden_size or model.config.hidden_size")
        self.value_head = nn.Linear(int(hidden_size), 1, bias=False)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        response_tokens: torch.Tensor,
        prompt_len: int,
        packed_predict_index: Optional[torch.Tensor] = None,
        autocast_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Return padded ``[B,T]`` or packed ``[T_total]`` FP32 values."""

        autocast_context = (
            torch.autocast("cuda", autocast_dtype)
            if input_ids.device.type == "cuda" and autocast_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        with autocast_context:
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )
            hidden = output.last_hidden_state

        if packed_predict_index is not None:
            selected = hidden[0].index_select(0, packed_predict_index)
        else:
            max_response_len = int(response_tokens.shape[1])
            if max_response_len == 0:
                return hidden.new_zeros((int(response_tokens.shape[0]), 0), dtype=torch.float32)
            if int(prompt_len) < 1:
                raise ValueError("Qwen3TokenValueModel: prompt_len must be >= 1 for response values")
            start = int(prompt_len) - 1
            selected = hidden[:, start : start + max_response_len, :]
            if int(selected.shape[1]) != max_response_len:
                raise ValueError(
                    "Qwen3TokenValueModel: hidden-state sequence is too short for requested "
                    f"response values ({selected.shape[1]} < {max_response_len})"
                )

        # The scalar prediction is returned in FP32 for stable return/MSE math.
        # The projection itself follows the module's parameter dtype/autocast.
        selected = selected.to(dtype=self.value_head.weight.dtype)
        return self.value_head(selected).squeeze(-1).float()


class Qwen3ValueStage:
    """Teacher-forced, token-aligned value inference over generated segments."""

    def __init__(
        self,
        *,
        model: "Qwen3ValueBundle",
        autocast_precision: str = "bf16",
    ) -> None:
        self.model = model
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="Qwen3ValueStage.autocast_precision")

    def trainable_module(self) -> nn.Module:
        """Return the independent value model wrapped by the train backend."""

        return self.model.transformer

    def predict_values(
        self,
        conditions: Qwen3ARConditions,
        *,
        segment: TextSegment,
    ) -> torch.Tensor:
        """Return packed FP32 values aligned one-for-one with action tokens."""

        self._validate_inputs(conditions, segment)
        attn_impl = getattr(getattr(self.model.transformer, "config", None), "_attn_implementation", None)
        if _packed_replay_supported(attn_impl):
            packed = self.packed_predict_values(conditions, segment=segment)
            if packed is not None:
                return packed
        return self.padding_predict_values(conditions, segment=segment)

    def packed_predict_values(
        self,
        conditions: Qwen3ARConditions,
        *,
        segment: TextSegment,
    ) -> Optional[torch.Tensor]:
        """Packed-varlen prediction for sparse packed attention backends."""

        self._validate_inputs(conditions, segment)
        transformer = self.model.transformer
        device = next(transformer.parameters()).device
        pad_id = self.model.tokenizer.pad_token_id or 0
        attn_impl = getattr(getattr(transformer, "config", None), "_attn_implementation", None)
        layout = build_packed_replay_layout(
            prompt_ids=conditions.prompt.input_ids,
            prompt_mask=conditions.prompt.attention_mask,
            segment=segment,
            device=device,
            pad_id=pad_id,
            caller="Qwen3ValueStage.packed_predict_values",
            pad_to_multiple=1024 if attn_impl == "flex_attention" else None,
        )
        if layout is None:
            return None
        values = transformer(
            input_ids=layout.input_ids,
            attention_mask=None,
            position_ids=layout.position_ids,
            response_tokens=layout.response_tokens,
            prompt_len=0,
            packed_predict_index=layout.predict_index,
            autocast_dtype=(self.autocast_dtype if device.type == "cuda" else None),
        )
        return values.float()

    def padding_predict_values(
        self,
        conditions: Qwen3ARConditions,
        *,
        segment: TextSegment,
    ) -> torch.Tensor:
        """Dense padded prediction and repacking into segment token order."""

        self._validate_inputs(conditions, segment)
        transformer = self.model.transformer
        device = next(transformer.parameters()).device
        layout = build_padded_replay_layout(
            prompt_ids=conditions.prompt.input_ids,
            prompt_mask=conditions.prompt.attention_mask,
            segment=segment,
            device=device,
            pad_id=self.model.tokenizer.pad_token_id or 0,
            caller="Qwen3ValueStage.padding_predict_values",
        )
        padded = transformer(
            input_ids=layout.input_ids,
            attention_mask=layout.attention_mask,
            position_ids=layout.position_ids,
            response_tokens=layout.response_tokens,
            prompt_len=layout.prompt_len,
            packed_predict_index=None,
            autocast_dtype=(self.autocast_dtype if device.type == "cuda" else None),
        )
        return pack_padded_token_outputs(padded, layout.lengths).float()

    @staticmethod
    def _validate_inputs(conditions: Qwen3ARConditions, segment: TextSegment) -> None:
        if conditions.prompt is None or conditions.prompt.input_ids is None:
            raise ValueError("Qwen3ValueStage: conditions.prompt.input_ids is None")
        if conditions.prompt.attention_mask is None:
            raise ValueError("Qwen3ValueStage: conditions.prompt.attention_mask is None")
        if segment.tokens is None or segment.cu_seqlens is None or segment.lengths is None:
            raise ValueError(
                "Qwen3ValueStage: segment requires tokens with framework-managed "
                "cu_seqlens (construct via TextSegment.pack)"
            )


def trainable_value_parameter_summary(model: Qwen3TokenValueModel) -> Dict[str, Any]:
    """Small logging/validation payload for trainer startup."""

    names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    return {
        "names": names,
        "tensors": len(names),
        "parameters": sum(
            int(parameter.numel()) for _, parameter in model.named_parameters() if parameter.requires_grad
        ),
    }


__all__ = [
    "Qwen3TokenValueModel",
    "Qwen3ValueStage",
    "configure_value_trainable_parameters",
    "trainable_value_parameter_summary",
]
