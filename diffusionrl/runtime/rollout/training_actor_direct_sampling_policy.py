"""Training-actor direct-sampling request planning and sub-batch execution."""

from __future__ import annotations

import os
from functools import partial
import tqdm as tqdm_
tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)
from dataclasses import dataclass
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from diffusionrl.config.arguments import is_training_actor_direct_sampling_mode

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)

def slice_prompt_aligned_batch(
    batch: Dict[str, Any],
    *,
    start: int,
    end: int,
) -> Dict[str, Any]:
    """Slice prompt-aligned fields while preserving shared batch values."""
    prompts = batch.get("prompts", []) or []
    if not isinstance(prompts, list):
        raise ValueError("Prompt-aligned batch slicing requires batch['prompts'] to be a list.")

    prompt_count = len(prompts)
    sliced: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, list) and len(value) == prompt_count:
            sliced[key] = value[start:end]
            continue
        if isinstance(value, tuple) and len(value) == prompt_count:
            sliced[key] = value[start:end]
            continue
        try:
            if hasattr(value, "dim") and callable(value.dim) and value.dim() > 0 and int(value.shape[0]) == prompt_count:
                sliced[key] = value[start:end]
                continue
        except Exception:
            pass
        sliced[key] = value

    sliced["prompts"] = list(prompts[start:end])
    return sliced


@dataclass(frozen=True)
class SampledRolloutBatch:
    """Aggregated outputs for one rollout after request-level sub-batching."""

    sampler_outputs: List[Any]
    train_prompts: List[str]
    base_prompts: List[str]
    prompt_metadata: Optional[List[Optional[Dict[str, Any]]]]


class TrainingActorDirectSamplingPolicy:
    """Plan and execute training-actor direct-sampling request splits."""

    def __init__(
        self,
        *,
        is_direct_sampling_mode: bool,
        num_samples_per_prompt: int,
        direct_sampling_batch_size: Optional[int],
    ) -> None:
        self.is_direct_sampling_mode = bool(is_direct_sampling_mode)
        self.num_samples_per_prompt = max(1, int(num_samples_per_prompt))
        self.direct_sampling_batch_size = (
            None if direct_sampling_batch_size is None else max(1, int(direct_sampling_batch_size))
        )

    @classmethod
    def from_args(cls, args: Any) -> "TrainingActorDirectSamplingPolicy":
        return cls(
            is_direct_sampling_mode=is_training_actor_direct_sampling_mode(args),
            num_samples_per_prompt=getattr(args.algorithm, "num_samples_per_prompt", 1),
            direct_sampling_batch_size=getattr(args.sampling, "direct_sampling_batch_size", None),
        )

    def requests_per_rollout(self, *, rollout_total_samples: int) -> int:
        total_samples = max(1, int(rollout_total_samples))
        direct_sampling_batch_size = self.direct_sampling_batch_size
        if (
            not self.is_direct_sampling_mode
            or direct_sampling_batch_size is None
            or direct_sampling_batch_size >= total_samples
        ):
            return 1
        if total_samples % direct_sampling_batch_size != 0:
            raise ValueError(
                "sampling.direct_sampling_batch_size must evenly divide rollout_total_samples "
                "for training-actor direct sampling. "
                f"Got rollout_total_samples={total_samples}, "
                f"direct_sampling_batch_size={direct_sampling_batch_size}."
            )
        return total_samples // direct_sampling_batch_size

    def iter_request_batches(
        self,
        *,
        batch: Dict[str, Any],
        num_samples_per_prompt: Optional[int] = None,
    ) -> List[Tuple[int, Dict[str, Any]]]:
        prompts = batch.get("prompts", []) or []
        if not isinstance(prompts, list) or len(prompts) == 0:
            raise ValueError("Rollout sampling requires non-empty batch['prompts'].")

        effective_num_samples_per_prompt = max(
            1,
            int(
                num_samples_per_prompt
                if num_samples_per_prompt is not None
                else self.num_samples_per_prompt
            ),
        )
        direct_sampling_batch_size = self.direct_sampling_batch_size
        if not self.is_direct_sampling_mode or direct_sampling_batch_size is None:
            return [(0, batch)]

        total_samples = len(prompts) * effective_num_samples_per_prompt
        if direct_sampling_batch_size >= total_samples:
            return [(0, batch)]
        if direct_sampling_batch_size % effective_num_samples_per_prompt != 0:
            raise ValueError(
                "sampling.direct_sampling_batch_size must be divisible by algorithm.num_samples_per_prompt. "
                f"Got direct_sampling_batch_size={direct_sampling_batch_size}, "
                f"num_samples_per_prompt={effective_num_samples_per_prompt}."
            )

        prompts_per_request = direct_sampling_batch_size // effective_num_samples_per_prompt
        if prompts_per_request < 1 or len(prompts) % prompts_per_request != 0:
            raise ValueError(
                "Prompt batch size must be divisible by prompts_per_request derived from "
                "sampling.direct_sampling_batch_size. "
                f"Got prompt_count={len(prompts)}, prompts_per_request={prompts_per_request}."
            )

        requests: List[Tuple[int, Dict[str, Any]]] = []
        batch_index = 0
        for start in range(0, len(prompts), prompts_per_request):
            end = start + prompts_per_request
            requests.append(
                (
                    batch_index,
                    slice_prompt_aligned_batch(batch, start=start, end=end),
                )
            )
            batch_index += 1
        return requests

    def sample_rollout(
        self,
        *,
        batch: Dict[str, Any],
        sample_request: Callable[[Dict[str, Any]], Tuple[List[Any], List[str], List[str]]],
        attach_embeddings: Optional[Callable[[List[Any], List[str]], None]] = None,
        validate_sampler_outputs: Optional[Callable[[List[Any]], None]] = None,
        rollout_id: Optional[int] = None,
    ) -> SampledRolloutBatch:
        """Execute one rollout, optionally split into multiple sampling requests."""

        all_outputs: List[Any] = []
        all_train_prompts: List[str] = []
        all_base_prompts: List[str] = []
        all_prompt_metadata: List[Optional[Dict[str, Any]]] = []
        prompts = batch.get("prompts", []) or []
        if not isinstance(prompts, list) or len(prompts) == 0:
            raise ValueError("Rollout sampling requires non-empty batch['prompts'].")
        requests_per_rollout = self.requests_per_rollout(
            rollout_total_samples=len(prompts) * self.num_samples_per_prompt
        )

        rank = int(os.environ.get("RANK", 0))
        for batch_idx, current_batch in tqdm(
            self.iter_request_batches(batch=batch),
            desc=f"Rollout {rollout_id} on rank {rank}",
            unit="request",
            disable=(rank != 0),
        ):
            current_prompts = current_batch.get("prompts", []) or []
            if not isinstance(current_prompts, list) or len(current_prompts) == 0:
                raise ValueError(
                    "Rollout sampling requires non-empty text prompts in batch['prompts']."
                )

            if requests_per_rollout > 1:
                logger.debug(
                    "rollout=%s direct-sampling request %s/%s prompts=%s",
                    rollout_id,
                    batch_idx + 1,
                    requests_per_rollout,
                    len(current_prompts),
                )

            sampler_outputs, train_prompts, base_prompts = sample_request(current_batch)

            if attach_embeddings is not None:
                attach_embeddings(
                    sampler_outputs,
                    train_prompts if train_prompts else current_prompts,
                )

            if validate_sampler_outputs is not None:
                validate_sampler_outputs(sampler_outputs)

            all_outputs.extend(sampler_outputs)
            all_train_prompts.extend(train_prompts if train_prompts else current_prompts)
            all_base_prompts.extend(base_prompts if base_prompts else current_prompts)

            current_metadata = current_batch.get("metadata")
            if isinstance(current_metadata, list) and len(current_metadata) == len(current_prompts):
                all_prompt_metadata.extend(current_metadata)
            else:
                if current_metadata is not None and not isinstance(current_metadata, list):
                    logger.warning(
                        "Ignoring non-list prompt metadata in rollout batch %s: %s",
                        batch_idx,
                        type(current_metadata).__name__,
                    )
                elif isinstance(current_metadata, list) and len(current_metadata) != len(current_prompts):
                    logger.warning(
                        "Ignoring misaligned prompt metadata in rollout batch %s (metadata=%s, prompts=%s).",
                        batch_idx,
                        len(current_metadata),
                        len(current_prompts),
                    )
                all_prompt_metadata.extend([None] * len(current_prompts))

        return SampledRolloutBatch(
            sampler_outputs=all_outputs,
            train_prompts=all_train_prompts,
            base_prompts=all_base_prompts,
            prompt_metadata=all_prompt_metadata if any(item is not None for item in all_prompt_metadata) else None,
        )


__all__ = [
    "SampledRolloutBatch",
    "TrainingActorDirectSamplingPolicy",
]
