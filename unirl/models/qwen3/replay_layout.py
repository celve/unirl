"""Shared teacher-forcing layouts for Qwen3 actor and value replay.

Both the actor and the SAO value critic must read the hidden state immediately
*before* each generated token.  Keeping the prompt/response packing and the
prediction indices in one module prevents a subtle one-token drift between the
policy log-probabilities and critic values.

The helpers here only build tensors.  They do not invoke a model and therefore
remain usable by the causal-LM actor and the training-only value model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from unirl.types.segments import TextSegment


@dataclass(frozen=True)
class Qwen3PaddedReplayLayout:
    """Dense ``prompt + response`` teacher-forcing inputs.

    ``response_tokens`` is padded to ``[B, T_max]``.  Model outputs at
    ``prompt_len - 1 + t`` predict response token ``t``.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    response_tokens: torch.Tensor
    lengths: Tuple[int, ...]
    prompt_len: int


@dataclass(frozen=True)
class Qwen3PackedReplayLayout:
    """One-row packed teacher-forcing inputs with restart position IDs."""

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    response_tokens: torch.Tensor
    predict_index: torch.Tensor
    lengths: Tuple[int, ...]


def _segment_lengths(segment: TextSegment, *, batch_size: int, caller: str) -> Tuple[int, ...]:
    if segment.tokens is None or segment.cu_seqlens is None or segment.lengths is None:
        raise ValueError(
            f"{caller}: segment requires tokens with framework-managed cu_seqlens (construct via TextSegment.pack)"
        )
    lengths = tuple(int(n) for n in segment.lengths.tolist())
    if len(lengths) != batch_size:
        raise ValueError(
            f"{caller}: prompt batch size ({batch_size}) does not match segment batch size ({len(lengths)})"
        )
    return lengths


def build_padded_replay_layout(
    *,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    segment: TextSegment,
    device: torch.device,
    pad_id: int,
    caller: str,
) -> Qwen3PaddedReplayLayout:
    """Build the dense replay layout used by actor and value forwards.

    Input prompts are emitted right-padded by the Qwen3 chat-template stage.
    They are re-padded to the left so every response begins immediately after
    the row's last real prompt token.  Leading columns that are padding for the
    whole microbatch are then removed.
    """

    prompt_ids = prompt_ids.to(device=device, dtype=torch.long)
    prompt_mask = prompt_mask.to(device=device)
    if prompt_ids.ndim != 2 or prompt_mask.shape != prompt_ids.shape:
        raise ValueError(
            f"{caller}: expected prompt ids/mask with matching [B, P] shapes, "
            f"got {tuple(prompt_ids.shape)} and {tuple(prompt_mask.shape)}"
        )

    batch_size = int(prompt_ids.shape[0])
    prompt_len = int(prompt_ids.shape[1])
    lengths = _segment_lengths(segment, batch_size=batch_size, caller=caller)
    max_response_len = max(lengths, default=0)

    response_tokens = torch.full(
        (batch_size, max_response_len),
        int(pad_id),
        dtype=torch.long,
        device=device,
    )
    response_mask = torch.zeros(
        (batch_size, max_response_len),
        dtype=prompt_mask.dtype,
        device=device,
    )
    cu = tuple(int(c) for c in segment.cu_seqlens.tolist())
    flat_tokens = segment.tokens.to(device=device, dtype=torch.long)
    for row, length in enumerate(lengths):
        if length == 0:
            continue
        response_tokens[row, :length] = flat_tokens[cu[row] : cu[row] + length]
        response_mask[row, :length] = 1

    real_prompt_lens = prompt_mask.long().sum(dim=-1)
    if real_prompt_lens.numel() and int(real_prompt_lens.min().item()) < prompt_len:
        left_padded_ids = torch.full_like(prompt_ids, int(pad_id))
        left_padded_mask = torch.zeros_like(prompt_mask)
        bool_mask = prompt_mask.bool()
        for row in range(batch_size):
            real_tokens = prompt_ids[row][bool_mask[row]]
            length = int(real_tokens.numel())
            if length == 0:
                continue
            left_padded_ids[row, prompt_len - length :] = real_tokens
            left_padded_mask[row, prompt_len - length :] = 1
        prompt_ids = left_padded_ids
        prompt_mask = left_padded_mask

    max_real_prompt = int(real_prompt_lens.max().item()) if real_prompt_lens.numel() else 0
    if max_real_prompt <= 0 and any(length > 0 for length in lengths):
        raise ValueError(f"{caller}: cannot value/replay a response with no real prompt token")
    if 0 < max_real_prompt < prompt_len:
        prompt_ids = prompt_ids[:, prompt_len - max_real_prompt :]
        prompt_mask = prompt_mask[:, prompt_len - max_real_prompt :]
        prompt_len = max_real_prompt

    if max_response_len > 0:
        input_ids = torch.cat([prompt_ids, response_tokens], dim=1)
        attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
    else:
        input_ids = prompt_ids
        attention_mask = prompt_mask

    # HF Qwen3 otherwise derives an arange that counts padding.  Cumulative
    # positions match rollout engines under left padding and keep response RoPE
    # positions invariant to batch composition.
    position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
    return Qwen3PaddedReplayLayout(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        response_tokens=response_tokens,
        lengths=lengths,
        prompt_len=prompt_len,
    )


def build_packed_replay_layout(
    *,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    segment: TextSegment,
    device: torch.device,
    pad_id: int,
    caller: str,
    pad_to_multiple: Optional[int] = None,
) -> Optional[Qwen3PackedReplayLayout]:
    """Build a restart-position, one-row packed replay layout.

    Returns ``None`` for a single-sample batch, where packing has no benefit.
    Callers are responsible for ensuring their attention backend recognizes
    restarting ``position_ids`` as independent block-causal sequences.
    """

    prompt_ids = prompt_ids.to(device=device, dtype=torch.long)
    prompt_mask = prompt_mask.to(device=device)
    if prompt_ids.ndim != 2 or prompt_mask.shape != prompt_ids.shape:
        raise ValueError(
            f"{caller}: expected prompt ids/mask with matching [B, P] shapes, "
            f"got {tuple(prompt_ids.shape)} and {tuple(prompt_mask.shape)}"
        )
    batch_size = int(prompt_ids.shape[0])
    if batch_size <= 1:
        return None

    lengths = _segment_lengths(segment, batch_size=batch_size, caller=caller)
    cu = tuple(int(c) for c in segment.cu_seqlens.tolist())
    flat_response = segment.tokens.to(device=device, dtype=torch.long)
    real_prompt_lens = prompt_mask.long().sum(dim=-1)

    streams: List[torch.Tensor] = []
    positions: List[torch.Tensor] = []
    prediction_indices: List[torch.Tensor] = []
    offset = 0
    bool_mask = prompt_mask.bool()
    for row, response_len in enumerate(lengths):
        # Gather by mask rather than assuming right padding.  This keeps packed
        # and padded replay equivalent for both left- and right-padded inputs.
        real_prompt = prompt_ids[row][bool_mask[row]]
        prompt_len = int(real_prompt_lens[row].item())
        if prompt_len < 1:
            raise ValueError(f"{caller}: packed stream {row} has no real prompt token")
        response = flat_response[cu[row] : cu[row] + response_len]
        stream = torch.cat([real_prompt, response])
        streams.append(stream)
        positions.append(torch.arange(stream.numel(), device=device, dtype=torch.long))
        if response_len > 0:
            prediction_indices.append(
                torch.arange(
                    offset + prompt_len - 1,
                    offset + prompt_len - 1 + response_len,
                    device=device,
                    dtype=torch.long,
                )
            )
        offset += int(stream.numel())

    packed_ids = torch.cat(streams).unsqueeze(0)
    packed_positions = torch.cat(positions).unsqueeze(0)
    predict_index = (
        torch.cat(prediction_indices) if prediction_indices else torch.zeros(0, dtype=torch.long, device=device)
    )

    multiple = int(pad_to_multiple or 0)
    if multiple > 0:
        length = int(packed_ids.shape[1])
        target = ((length + multiple - 1) // multiple) * multiple
        if target > length:
            filler_len = target - length
            filler_ids = torch.full((1, filler_len), int(pad_id), dtype=packed_ids.dtype, device=device)
            # Restart at zero: the sparse packed mask treats filler as an
            # isolated stream, and no output is gathered from it.
            filler_positions = torch.arange(filler_len, device=device, dtype=torch.long).unsqueeze(0)
            packed_ids = torch.cat([packed_ids, filler_ids], dim=1)
            packed_positions = torch.cat([packed_positions, filler_positions], dim=1)

    return Qwen3PackedReplayLayout(
        input_ids=packed_ids,
        position_ids=packed_positions,
        response_tokens=flat_response,
        predict_index=predict_index,
        lengths=lengths,
    )


def pack_padded_token_outputs(values: torch.Tensor, lengths: Tuple[int, ...]) -> torch.Tensor:
    """Remove response padding from ``[B, T_max]`` model outputs."""

    if values.ndim != 2 or int(values.shape[0]) != len(lengths):
        raise ValueError(
            "pack_padded_token_outputs: expected [B, T_max] values matching "
            f"{len(lengths)} lengths, got {tuple(values.shape)}"
        )
    parts = [values[row, :length] for row, length in enumerate(lengths) if length > 0]
    if not parts:
        return values.new_zeros((0,))
    return torch.cat(parts, dim=0)


__all__ = [
    "Qwen3PackedReplayLayout",
    "Qwen3PaddedReplayLayout",
    "build_packed_replay_layout",
    "build_padded_replay_layout",
    "pack_padded_token_outputs",
]
