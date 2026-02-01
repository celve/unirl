"""
Patch FastVideo VideoGenerator with weight update and offload utilities.
"""

from __future__ import annotations


def _augment_result_with_grpo_payload(self, result):
    if not isinstance(result, dict):
        return result
    executor = getattr(self, "executor", None)
    batch = getattr(executor, "_last_output_batch", None)
    if batch is None:
        return result

    # Restore trajectory fields when executor path strips them.
    if result.get("trajectory") is None and getattr(batch, "trajectory_latents", None) is not None:
        result["trajectory"] = batch.trajectory_latents
    if result.get("trajectory_timesteps") is None and getattr(batch, "trajectory_timesteps", None) is not None:
        result["trajectory_timesteps"] = batch.trajectory_timesteps
    if result.get("trajectory_decoded") is None and getattr(batch, "trajectory_decoded", None) is not None:
        result["trajectory_decoded"] = batch.trajectory_decoded

    # Expose prompt embeddings for GRPO contract construction.
    prompt_embeds = getattr(batch, "prompt_embeds", None)
    if prompt_embeds is not None and (not isinstance(prompt_embeds, list) or len(prompt_embeds) > 0):
        result["prompt_embeds"] = prompt_embeds[0] if isinstance(prompt_embeds, list) else prompt_embeds
    negative_prompt_embeds = getattr(batch, "negative_prompt_embeds", None)
    if negative_prompt_embeds is not None and (
        not isinstance(negative_prompt_embeds, list) or len(negative_prompt_embeds) > 0
    ):
        result["negative_prompt_embeds"] = (
            negative_prompt_embeds[0]
            if isinstance(negative_prompt_embeds, list)
            else negative_prompt_embeds
        )

    prompt_attention_mask = getattr(batch, "prompt_attention_mask", None)
    if prompt_attention_mask is not None and (
        not isinstance(prompt_attention_mask, list) or len(prompt_attention_mask) > 0
    ):
        result["encoder_attention_mask"] = (
            prompt_attention_mask[0]
            if isinstance(prompt_attention_mask, list)
            else prompt_attention_mask
        )
    return result


def _generate_video_with_grpo_payload(self, *args, **kwargs):
    result = _ORIG_GENERATE_VIDEO(self, *args, **kwargs)
    if isinstance(result, list):
        return result
    return _augment_result_with_grpo_payload(self, result)


def _update_weights_from_path(self, checkpoint_path: str, strict: bool = False) -> None:
    if not hasattr(self, "executor") or self.executor is None:
        raise RuntimeError("FastVideo executor not initialized")
    self.executor.update_weights_from_path(checkpoint_path, strict=strict)


def _offload_model(self) -> None:
    if not hasattr(self, "executor") or self.executor is None:
        raise RuntimeError("FastVideo executor not initialized")
    self.executor.offload_model()


def _onload_model(self) -> None:
    if not hasattr(self, "executor") or self.executor is None:
        raise RuntimeError("FastVideo executor not initialized")
    self.executor.onload_model()


def _get_grpo_patch_status(self) -> dict:
    """Introspect GRPO monkey-patch capabilities for compatibility checks."""
    return {
        "generate_video_payload": hasattr(self, "generate_video")
        and getattr(self.generate_video, "__name__", "") == "_generate_video_with_grpo_payload",
        "update_weights_from_path": hasattr(self, "update_weights_from_path"),
        "offload_model": hasattr(self, "offload_model"),
        "onload_model": hasattr(self, "onload_model"),
    }


def apply() -> None:
    from fastvideo.entrypoints.video_generator import VideoGenerator

    global _ORIG_GENERATE_VIDEO
    if "_ORIG_GENERATE_VIDEO" not in globals():
        _ORIG_GENERATE_VIDEO = VideoGenerator.generate_video
    if getattr(VideoGenerator.generate_video, "__name__", "") != "_generate_video_with_grpo_payload":
        setattr(VideoGenerator, "generate_video", _generate_video_with_grpo_payload)

    if not hasattr(VideoGenerator, "update_weights_from_path"):
        setattr(VideoGenerator, "update_weights_from_path", _update_weights_from_path)
    if not hasattr(VideoGenerator, "offload_model"):
        setattr(VideoGenerator, "offload_model", _offload_model)
    if not hasattr(VideoGenerator, "onload_model"):
        setattr(VideoGenerator, "onload_model", _onload_model)
    if not hasattr(VideoGenerator, "get_grpo_patch_status"):
        setattr(VideoGenerator, "get_grpo_patch_status", _get_grpo_patch_status)
