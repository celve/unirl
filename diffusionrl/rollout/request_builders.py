"""Driver-side helpers for loading prompt batches and building eval requests."""

from __future__ import annotations

from typing import Any, Dict


def load_prompt_batch_from_source(
    *,
    data_source: Any,
    prompt_batch_size: int,
) -> Dict[str, Any]:
    """Fetch one rollout prompt batch from the configured data source."""
    if data_source is None:
        raise RuntimeError("Rollout pipeline requires an initialized data source.")
    samples = data_source.get_samples(int(prompt_batch_size))
    if isinstance(samples, dict):
        return samples
    raise TypeError(
        "DataSource.get_samples() must return Dict[str, Any] with at least 'prompts'. "
        f"Got {type(samples).__name__}."
    )


def build_eval_request_batch(
    *,
    data_source: Any,
    prompt_batch_size: int,
    evaluation_settings: Any,
) -> Dict[str, Any]:
    """Build the canonical evaluation request payload before request expansion."""
    eval_batch_size = int(getattr(evaluation_settings, "eval_batch_size", 0) or 0)
    if data_source is not None and hasattr(data_source, "get_eval_samples"):
        eval_samples = data_source.get_eval_samples(eval_batch_size)
        if isinstance(eval_samples, dict):
            eval_batch = dict(eval_samples)
        elif isinstance(eval_samples, list):
            eval_batch = {"prompts": list(eval_samples)}
        else:
            raise TypeError(
                "DataSource.get_eval_samples() must return List[str] or Dict[str, Any] "
                f"with at least 'prompts'. Got {type(eval_samples).__name__}."
            )
    else:
        eval_batch = dict(
            load_prompt_batch_from_source(
                data_source=data_source,
                prompt_batch_size=prompt_batch_size,
            )
        )

    prompts = list(eval_batch.get("prompts", [])[:eval_batch_size])
    prompt_ids = eval_batch.get("prompt_ids")
    if isinstance(prompt_ids, list):
        prompt_ids = prompt_ids[: len(prompts)]
    else:
        prompt_ids = None
    prompt_metadata = eval_batch.get("metadata")
    if isinstance(prompt_metadata, list):
        prompt_metadata = prompt_metadata[: len(prompts)]
    else:
        prompt_metadata = None

    request_batch: Dict[str, Any] = {
        "prompts": prompts,
        "prompt_ids": prompt_ids,
        "metadata": prompt_metadata,
    }
    raw_request_kwargs = eval_batch.get("kwargs")
    if isinstance(raw_request_kwargs, dict):
        request_batch["kwargs"] = dict(raw_request_kwargs)

    eval_overrides: Dict[str, Any] = {}
    if getattr(evaluation_settings, "num_inference_steps", None) is not None:
        eval_overrides["num_inference_steps"] = int(evaluation_settings.num_inference_steps)

    sampling_adapter = getattr(evaluation_settings, "sampling_adapter", None)
    if sampling_adapter is not None and str(sampling_adapter).strip():
        eval_overrides["sampling_adapter"] = str(sampling_adapter).strip()

    sampler_overrides: Dict[str, Any] = {}
    raw_sde_type = getattr(evaluation_settings, "sde_type", None)
    if raw_sde_type is not None and str(raw_sde_type).strip():
        sampler_overrides["sde_type"] = str(raw_sde_type).strip()
    if getattr(evaluation_settings, "eta", None) is not None:
        sampler_overrides["eta"] = float(evaluation_settings.eta)
    if sampler_overrides:
        eval_overrides["kwargs"] = {"sampler_overrides": sampler_overrides}

    if eval_overrides:
        request_batch.update(
            {key: value for key, value in eval_overrides.items() if key != "kwargs"}
        )
        override_kwargs = eval_overrides.get("kwargs")
        if isinstance(override_kwargs, dict):
            merged_kwargs = dict(request_batch.get("kwargs") or {})
            merged_sampler_overrides = dict(merged_kwargs.get("sampler_overrides") or {})
            merged_sampler_overrides.update(
                dict(override_kwargs.get("sampler_overrides") or {})
            )
            if merged_sampler_overrides:
                merged_kwargs["sampler_overrides"] = merged_sampler_overrides
            request_batch["kwargs"] = merged_kwargs
    return request_batch


__all__ = [
    "build_eval_request_batch",
    "load_prompt_batch_from_source",
]
