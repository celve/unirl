"""
Patch FastVideo Executor with weight update and offload utilities.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _restore_grpo_payload(batch: Any) -> Any:
    if batch is None:
        return batch
    extra = getattr(batch, "extra", None)
    if not isinstance(extra, dict):
        extra = {}
        batch.extra = extra

    if getattr(batch, "trajectory_latents", None) is None:
        batch.trajectory_latents = extra.get("__grpo_trajectory_latents")
    if getattr(batch, "trajectory_timesteps", None) is None:
        batch.trajectory_timesteps = extra.get("__grpo_trajectory_timesteps")
    if getattr(batch, "trajectory_decoded", None) is None:
        batch.trajectory_decoded = extra.get("__grpo_trajectory_decoded")

    if not getattr(batch, "prompt_embeds", None):
        batch.prompt_embeds = extra.get("__grpo_prompt_embeds") or []
    if getattr(batch, "negative_prompt_embeds", None) in (None, []):
        batch.negative_prompt_embeds = extra.get("__grpo_negative_prompt_embeds") or []
    if getattr(batch, "prompt_attention_mask", None) in (None, []):
        batch.prompt_attention_mask = extra.get("__grpo_prompt_attention_mask")
    if getattr(batch, "negative_attention_mask", None) in (None, []):
        batch.negative_attention_mask = extra.get("__grpo_negative_attention_mask")

    return batch


def _update_weights_from_path(self, checkpoint_path: str, strict: bool = False) -> None:
    responses: List[Dict[str, Any]] = self.collective_rpc(
        "update_weights_from_path",
        kwargs={"checkpoint_path": checkpoint_path, "strict": strict},
    )
    for idx, resp in enumerate(responses):
        status = resp.get("status") if isinstance(resp, dict) else None
        if status != "updated":
            raise RuntimeError(f"FastVideo worker {idx} weight update failed: {resp}")


def _offload_model(self) -> None:
    responses: List[Dict[str, Any]] = self.collective_rpc("offload_model")
    for idx, resp in enumerate(responses):
        status = resp.get("status") if isinstance(resp, dict) else None
        if status != "offloaded":
            raise RuntimeError(f"FastVideo worker {idx} offload failed: {resp}")


def _onload_model(self) -> None:
    responses: List[Dict[str, Any]] = self.collective_rpc("onload_model")
    for idx, resp in enumerate(responses):
        status = resp.get("status") if isinstance(resp, dict) else None
        if status != "onloaded":
            raise RuntimeError(f"FastVideo worker {idx} onload failed: {resp}")


def _mp_execute_forward_with_grpo_payload(self, forward_batch, fastvideo_args):
    from fastvideo import envs
    from fastvideo.pipelines import ForwardBatch

    responses = self.collective_rpc(
        "execute_forward",
        kwargs={"forward_batch": forward_batch, "fastvideo_args": fastvideo_args},
    )
    output = responses[0]["output_batch"]
    logging_info = responses[0]["logging_info"] if envs.FASTVIDEO_STAGE_LOGGING else None
    extra = responses[0].get("extra", {})

    result_batch = ForwardBatch(
        data_type=forward_batch.data_type,
        output=output,
        logging_info=logging_info,
        extra=extra,
    )
    result_batch = _restore_grpo_payload(result_batch)
    self._last_output_batch = result_batch
    return result_batch


def _ray_execute_forward_with_grpo_payload(self, forward_batch, fastvideo_args):
    from fastvideo import envs
    from fastvideo.pipelines import ForwardBatch

    responses = self.collective_rpc(
        "execute_forward",
        kwargs={"forward_batch": forward_batch, "fastvideo_args": fastvideo_args},
    )
    source = responses[0]
    output = source.output.cpu()
    logging_info = source.logging_info if envs.FASTVIDEO_STAGE_LOGGING else None
    extra = dict(getattr(source, "extra", {}) or {})

    result_batch = ForwardBatch(
        data_type=forward_batch.data_type,
        output=output,
        logging_info=logging_info,
        extra=extra,
    )
    # Preserve fields that were previously discarded by Ray executor path.
    for attr_name in (
        "trajectory_latents",
        "trajectory_timesteps",
        "trajectory_decoded",
        "prompt_embeds",
        "negative_prompt_embeds",
        "prompt_attention_mask",
        "negative_attention_mask",
    ):
        setattr(result_batch, attr_name, getattr(source, attr_name, None))
    result_batch = _restore_grpo_payload(result_batch)
    self._last_output_batch = result_batch
    return result_batch


def apply() -> None:
    from fastvideo.worker.executor import Executor

    if not hasattr(Executor, "update_weights_from_path"):
        setattr(Executor, "update_weights_from_path", _update_weights_from_path)
    if not hasattr(Executor, "offload_model"):
        setattr(Executor, "offload_model", _offload_model)
    if not hasattr(Executor, "onload_model"):
        setattr(Executor, "onload_model", _onload_model)

    # Preserve GRPO contract payloads (trajectory/prompt embeddings) across IPC.
    try:
        from fastvideo.worker.multiproc_executor import MultiprocExecutor

        if getattr(MultiprocExecutor.execute_forward, "__name__", "") != "_mp_execute_forward_with_grpo_payload":
            setattr(MultiprocExecutor, "execute_forward", _mp_execute_forward_with_grpo_payload)
    except Exception:
        pass

    try:
        from fastvideo.worker.ray_distributed_executor import RayDistributedExecutor

        if getattr(RayDistributedExecutor.execute_forward, "__name__", "") != "_ray_execute_forward_with_grpo_payload":
            setattr(RayDistributedExecutor, "execute_forward", _ray_execute_forward_with_grpo_payload)
    except Exception:
        pass
