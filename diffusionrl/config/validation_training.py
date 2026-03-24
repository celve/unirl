"""Training-geometry and optimizer-facing validation helpers."""

from __future__ import annotations

from typing import Any, Optional

from diffusionrl.config.resolution import (
    derive_global_rollout_batch_size,
    normalize_lora_target_modules,
    derive_num_updates_per_local_batch,
    require_prompts_per_rollout,
    derive_training_topology,
)


def validate_training_batch_geometry(args: Any) -> None:
    """Validate batch-geometry invariants using resolved training geometry."""
    prompts_per_rollout = int(require_prompts_per_rollout(args))
    samples_per_prompt = int(args.algorithm.samples_per_prompt)
    if samples_per_prompt < 1:
        raise ValueError(f"algorithm.samples_per_prompt must be >= 1. Got {samples_per_prompt}.")

    num_updates_per_local_batch = derive_num_updates_per_local_batch(args)
    global_batch_size = derive_global_rollout_batch_size(args)
    topology = derive_training_topology(args)
    dp_size = int(topology.dp_size)
    dp_replicate_size = int(topology.dp_replicate_size)
    raw_micro_batch_size = args.training.local_micro_batch_size

    def _format_geometry(
        *,
        local_batch_size: Optional[int],
        update_batch_size: Optional[int],
        micro_batch_size: Optional[int],
    ) -> str:
        local_text = str(local_batch_size) if local_batch_size is not None else "<not divisible by dp_size>"
        update_text = (
            str(update_batch_size)
            if update_batch_size is not None
            else "<not divisible by num_updates_per_local_batch>"
        )
        if raw_micro_batch_size is None:
            micro_text = "auto"
            if micro_batch_size is not None:
                micro_text = f"auto (= {micro_batch_size})"
        else:
            micro_text = str(micro_batch_size if micro_batch_size is not None else raw_micro_batch_size)
        return "\n".join(
            [
                "Resolved training batch geometry:",
                f"  global_batch_size = prompts_per_rollout({prompts_per_rollout}) * "
                f"samples_per_prompt({samples_per_prompt}) = {global_batch_size}",
                f"  local_batch_size = global_batch_size / dp_size({dp_size}) = {local_text}",
                f"  local_update_batch_size = local_batch_size / "
                f"num_updates_per_local_batch({num_updates_per_local_batch}) = {update_text}",
                f"  local_micro_batch_size = {micro_text}",
                f"  dp_replicate_size = {dp_replicate_size}",
            ]
        )

    def _raise_geometry_error(
        *,
        reason: str,
        fix_hint: str,
        local_batch_size: Optional[int],
        update_batch_size: Optional[int],
        micro_batch_size: Optional[int],
    ) -> None:
        raise ValueError(
            "\n".join(
                [
                    f"Invalid training batch geometry: {reason}",
                    _format_geometry(
                        local_batch_size=local_batch_size,
                        update_batch_size=update_batch_size,
                        micro_batch_size=micro_batch_size,
                    ),
                    f"Fix: {fix_hint}",
                ]
            )
        )

    if global_batch_size % dp_size != 0:
        _raise_geometry_error(
            reason="global rollout batch cannot be split evenly across training DP ranks "
            "(global_batch_size % dp_size != 0).",
            fix_hint="Adjust algorithm.prompts_per_rollout, algorithm.samples_per_prompt, "
            "or the training backend dp_size.",
            local_batch_size=None,
            update_batch_size=None,
            micro_batch_size=None,
        )
    local_batch_size = int(global_batch_size // dp_size)

    if global_batch_size % dp_replicate_size != 0:
        _raise_geometry_error(
            reason="global rollout batch must also be divisible by dp_replicate_size "
            "(global_batch_size % dp_replicate_size != 0).",
            fix_hint="Adjust algorithm.prompts_per_rollout, algorithm.samples_per_prompt, "
            "or the backend replicate topology.",
            local_batch_size=local_batch_size,
            update_batch_size=None,
            micro_batch_size=None,
        )

    if local_batch_size % num_updates_per_local_batch != 0:
        _raise_geometry_error(
            reason="local batch cannot be split evenly into optimizer updates "
            "(local_batch_size % num_updates_per_local_batch != 0).",
            fix_hint="Choose a training.num_updates_per_local_batch that evenly divides "
            "the resolved local_batch_size.",
            local_batch_size=local_batch_size,
            update_batch_size=None,
            micro_batch_size=None,
        )
    update_batch_size = int(local_batch_size // num_updates_per_local_batch)

    if raw_micro_batch_size is None:
        micro_batch_size = int(update_batch_size)
    else:
        micro_batch_size = int(raw_micro_batch_size)
        if micro_batch_size < 1:
            _raise_geometry_error(
                reason="training.local_micro_batch_size must be >= 1.",
                fix_hint="Set training.local_micro_batch_size to a positive integer, "
                "or omit it to use the resolved local_update_batch_size.",
                local_batch_size=local_batch_size,
                update_batch_size=update_batch_size,
                micro_batch_size=micro_batch_size,
            )

    if update_batch_size % micro_batch_size != 0:
        _raise_geometry_error(
            reason="local update batch cannot be split evenly into micro-batches "
            "(local_update_batch_size % local_micro_batch_size != 0).",
            fix_hint="Choose a training.local_micro_batch_size that evenly divides "
            "the resolved local_update_batch_size.",
            local_batch_size=local_batch_size,
            update_batch_size=update_batch_size,
            micro_batch_size=micro_batch_size,
        )


def validate_training_misc(args: Any) -> None:
    """Validate non-batch training knobs that affect downstream components."""
    normalize_lora_target_modules(args.training.lora_target_modules)


__all__ = [
    "validate_training_batch_geometry",
    "validate_training_misc",
]
