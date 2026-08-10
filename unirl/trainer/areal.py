"""AReaL trajectory serialization for agentic text training."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from unirl.distributed.tensor import hydrate
from unirl.types.conditions import TextTokenCondition
from unirl.types.sample import Part, Sample
from unirl.types.segments import TextSegment

_PROTOCOL = "areal_deep_research/v1"


class ARealTrajectoryError(ValueError):
    """The rollout cannot be represented as one faithful AReaL training row."""


def areal_metadata(trajectory: Sample) -> Mapping[str, Any]:
    """Return the terminal AReaL harness metadata for one trajectory."""
    if not trajectory.parts or not trajectory.parts[-1].metadata:
        raise ARealTrajectoryError("trajectory has no terminal harness metadata")
    metadata = trajectory.parts[-1].metadata[0] or {}
    harness = metadata.get("harness")
    if not isinstance(harness, Mapping) or harness.get("protocol") != _PROTOCOL:
        raise ARealTrajectoryError("trajectory is not stamped with the AReaL protocol")
    return harness


def _prompt_ids(part: Part) -> torch.Tensor:
    prompt = part.conditions.get("prompt")
    if not isinstance(prompt, TextTokenCondition) or prompt.input_ids is None or prompt.attention_mask is None:
        raise ARealTrajectoryError("generated turn has no tokenized prompt condition")
    input_ids = hydrate(prompt.input_ids).to(dtype=torch.long, device="cpu")
    attention_mask = hydrate(prompt.attention_mask).to(dtype=torch.bool, device="cpu")
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape or input_ids.shape[0] != 1:
        raise ARealTrajectoryError("generated prompt condition must contain exactly one aligned row")
    return input_ids[0][attention_mask[0]].clone()


def _output(part: Part) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(part.segment, TextSegment) or part.segment.tokens is None or part.segment.log_probs is None:
        raise ARealTrajectoryError("generated turn has no text tokens and behavior log-probabilities")
    if part.batch_size != 1:
        raise ARealTrajectoryError("AReaL trajectory assembly requires single-row generated turns")
    tokens = hydrate(part.segment.tokens).to(dtype=torch.long, device="cpu").flatten()
    log_probs = hydrate(part.segment.log_probs).to(dtype=torch.float32, device="cpu").flatten()
    if tokens.shape != log_probs.shape:
        raise ARealTrajectoryError("generated tokens and behavior log-probabilities are not aligned")
    return tokens, log_probs


def build_areal_part(trajectory: Sample) -> Part:
    """Build one masked training row from every generated turn in a trajectory."""
    metadata = areal_metadata(trajectory)
    if not isinstance(metadata.get("prediction"), str):
        raise ARealTrajectoryError("completed trajectory has no string prediction")
    generated = trajectory.gen_parts()
    if not generated:
        raise ARealTrajectoryError("trajectory has no generated turns")

    token_chunks: list[torch.Tensor] = []
    log_prob_chunks: list[torch.Tensor] = []
    mask_chunks: list[torch.Tensor] = []
    output_versions: list[int] = []
    first_prompt: torch.Tensor | None = None
    previous_prompt: torch.Tensor | None = None
    previous_output: torch.Tensor | None = None

    for turn, part in enumerate(generated):
        prompt = _prompt_ids(part)
        output, log_probs = _output(part)
        if first_prompt is None:
            first_prompt = prompt
        else:
            assert previous_prompt is not None and previous_output is not None
            expected_prefix = torch.cat([previous_prompt, previous_output])
            if prompt.numel() <= expected_prefix.numel() or not torch.equal(
                prompt[: expected_prefix.numel()], expected_prefix
            ):
                raise ARealTrajectoryError(f"turn {turn} prompt is not a strict extension of its parent interaction")
            suffix = prompt[expected_prefix.numel() :]
            token_chunks.append(suffix)
            log_prob_chunks.append(torch.zeros(suffix.numel(), dtype=torch.float32))
            mask_chunks.append(torch.zeros(suffix.numel(), dtype=torch.bool))

        token_chunks.append(output)
        log_prob_chunks.append(log_probs)
        mask_chunks.append(torch.ones(output.numel(), dtype=torch.bool))
        output_versions.append(int(part.output_version) if part.output_version is not None else -1)
        previous_prompt = prompt
        previous_output = output

    assert first_prompt is not None
    if first_prompt.numel() == 0:
        raise ARealTrajectoryError("trajectory has an empty initial policy prompt")
    tokens = torch.cat(token_chunks) if token_chunks else torch.zeros(0, dtype=torch.long)
    log_probs = torch.cat(log_prob_chunks) if log_prob_chunks else torch.zeros(0, dtype=torch.float32)
    loss_mask = torch.cat(mask_chunks) if mask_chunks else torch.zeros(0, dtype=torch.bool)
    if not bool(loss_mask.any()):
        raise ARealTrajectoryError("trajectory has no trainable assistant tokens")
    if tokens.shape != log_probs.shape or tokens.shape != loss_mask.shape:
        raise ARealTrajectoryError("assembled tokens, log-probabilities, and loss mask are not aligned")

    prompt_condition = TextTokenCondition(
        input_ids=first_prompt.unsqueeze(0),
        attention_mask=torch.ones((1, first_prompt.numel()), dtype=torch.long),
    )
    segment = TextSegment.pack(tokens=[tokens], log_probs=[log_probs], loss_mask=[loss_mask])
    row_metadata = {
        "areal": {
            "termination_reason": metadata.get("termination_reason"),
            "policy_call_count": metadata.get("policy_call_count"),
            "output_versions": output_versions,
        }
    }
    return Part(
        sample_ids=[generated[-1].sample_ids[0]],
        segment=segment,
        conditions={"prompt": prompt_condition},
        metadata=[row_metadata],
        sampling_params=generated[0].sampling_params,
    )


__all__ = ["ARealTrajectoryError", "areal_metadata", "build_areal_part"]
