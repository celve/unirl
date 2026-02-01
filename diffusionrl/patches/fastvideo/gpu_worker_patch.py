"""
Patch FastVideo GPU worker with weight update and offload utilities.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple


def _load_state_dict(checkpoint_path: str) -> Dict[str, Any]:
    # Prefer safetensors if available and file suffix matches.
    if checkpoint_path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file

            return load_file(checkpoint_path, device="cpu")
        except Exception:
            pass

    import torch

    return torch.load(checkpoint_path, map_location="cpu")


def _stash_grpo_payload(output_batch: Any) -> None:
    """
    Persist GRPO-relevant tensors in `extra` so executor IPC paths can reconstruct them.

    Multiproc/Ray executors often reduce ForwardBatch payloads before returning
    to the parent process. Stashing here keeps rollout contract fields available.
    """
    if output_batch is None:
        return
    extra = getattr(output_batch, "extra", None)
    if not isinstance(extra, dict):
        extra = {}
    extra.setdefault("__grpo_trajectory_latents", getattr(output_batch, "trajectory_latents", None))
    extra.setdefault("__grpo_trajectory_timesteps", getattr(output_batch, "trajectory_timesteps", None))
    extra.setdefault("__grpo_trajectory_decoded", getattr(output_batch, "trajectory_decoded", None))
    extra.setdefault("__grpo_prompt_embeds", getattr(output_batch, "prompt_embeds", None))
    extra.setdefault(
        "__grpo_negative_prompt_embeds",
        getattr(output_batch, "negative_prompt_embeds", None),
    )
    extra.setdefault(
        "__grpo_prompt_attention_mask",
        getattr(output_batch, "prompt_attention_mask", None),
    )
    extra.setdefault(
        "__grpo_negative_attention_mask",
        getattr(output_batch, "negative_attention_mask", None),
    )
    output_batch.extra = extra


def _execute_forward_with_grpo_payload(self, forward_batch, fastvideo_args):
    eta_override = getattr(fastvideo_args, "eta", None)
    if eta_override is not None:
        try:
            forward_batch.eta = float(eta_override)
        except Exception:
            pass
    output_batch = _ORIG_EXECUTE_FORWARD(self, forward_batch, fastvideo_args)
    _stash_grpo_payload(output_batch)
    return output_batch


def _get_transformer(worker) -> Any:
    pipeline = getattr(worker, "pipeline", None)
    if pipeline is None:
        return None
    modules = getattr(pipeline, "modules", None)
    if isinstance(modules, dict) and "transformer" in modules:
        return modules.get("transformer")
    if hasattr(pipeline, "get_module"):
        try:
            return pipeline.get_module("transformer")
        except Exception:
            return None
    return None


def _update_weights_from_path(self, checkpoint_path: str, strict: bool = False) -> Dict[str, Any]:
    transformer = _get_transformer(self)
    if transformer is None:
        return {"status": "failed: no transformer found"}

    state_dict = _load_state_dict(checkpoint_path)
    missing, unexpected = transformer.load_state_dict(state_dict, strict=strict)
    return {"status": "updated", "missing": len(missing), "unexpected": len(unexpected)}


def _iter_pipeline_modules(pipeline) -> Iterable[Tuple[str, Any]]:
    modules = getattr(pipeline, "modules", None)
    if isinstance(modules, dict):
        for name, module in modules.items():
            yield name, module
        return
    if hasattr(pipeline, "get_module"):
        for name in ("transformer", "transformer_2", "vae", "text_encoder", "image_encoder"):
            try:
                module = pipeline.get_module(name)
            except Exception:
                module = None
            if module is not None:
                yield name, module


def _offload_model(self) -> Dict[str, Any]:
    pipeline = getattr(self, "pipeline", None)
    if pipeline is None:
        return {"status": "failed: no pipeline"}

    moved = []
    released = []
    skipped = []
    for name, module in _iter_pipeline_modules(pipeline):
        if module is None or not hasattr(module, "to"):
            skipped.append(name)
            continue
        mgr = getattr(module, "_layerwise_offload_manager", None)
        mgr_enabled = mgr is not None and getattr(mgr, "enabled", False)
        if mgr_enabled:
            try:
                mgr.release_all()
                released.append(name)
            except Exception as e:
                return {"status": "failed: release_all", "module": name, "error": str(e)}
        try:
            module.to("cpu")
            moved.append(name)
        except Exception as e:
            return {"status": "failed: to_cpu", "module": name, "error": str(e)}

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return {"status": "offloaded", "moved": moved, "released": released, "skipped": skipped}


def _onload_model(self) -> Dict[str, Any]:
    pipeline = getattr(self, "pipeline", None)
    if pipeline is None:
        return {"status": "failed: no pipeline"}

    try:
        from fastvideo.distributed.parallel_state import get_local_torch_device
    except Exception as e:
        return {"status": "failed: get_device", "error": str(e)}

    fastvideo_args = getattr(self, "fastvideo_args", None)
    device = get_local_torch_device()
    moved = []
    prefetched = []
    skipped = []
    for name, module in _iter_pipeline_modules(pipeline):
        if module is None or not hasattr(module, "to"):
            skipped.append(name)
            continue
        mgr = getattr(module, "_layerwise_offload_manager", None)
        if mgr is not None and getattr(mgr, "enabled", False):
            try:
                # Keep non-layer params (norm/embeddings/head) in sync with full offload/onload.
                module.to(device)
                moved.append(name)
            except Exception as e:
                return {"status": "failed: to_device", "module": name, "error": str(e)}
            try:
                mgr.prefetch_layer(0, non_blocking=False)
                if getattr(mgr, "copy_stream", None) is not None:
                    import torch

                    torch.cuda.current_stream().wait_stream(mgr.copy_stream)
                prefetched.append(name)
                continue
            except Exception as e:
                return {"status": "failed: prefetch", "module": name, "error": str(e)}

        if fastvideo_args is not None:
            if name == "vae" and getattr(fastvideo_args, "vae_cpu_offload", False):
                skipped.append(name)
                continue
            if name == "text_encoder" and getattr(fastvideo_args, "text_encoder_cpu_offload", False):
                skipped.append(name)
                continue
            if name == "image_encoder" and getattr(fastvideo_args, "image_encoder_cpu_offload", False):
                skipped.append(name)
                continue
            if name in ("transformer", "transformer_2"):
                if getattr(fastvideo_args, "dit_layerwise_offload", False):
                    skipped.append(name)
                    continue
                if getattr(fastvideo_args, "dit_cpu_offload", False):
                    skipped.append(name)
                    continue

        try:
            module.to(device)
            moved.append(name)
        except Exception as e:
            return {"status": "failed: to_device", "module": name, "error": str(e)}

    return {"status": "onloaded", "moved": moved, "prefetched": prefetched, "skipped": skipped}


def apply() -> None:
    from fastvideo.worker.gpu_worker import Worker

    global _ORIG_EXECUTE_FORWARD
    if "_ORIG_EXECUTE_FORWARD" not in globals():
        _ORIG_EXECUTE_FORWARD = Worker.execute_forward
    if getattr(Worker.execute_forward, "__name__", "") != "_execute_forward_with_grpo_payload":
        setattr(Worker, "execute_forward", _execute_forward_with_grpo_payload)

    if not hasattr(Worker, "update_weights_from_path"):
        setattr(Worker, "update_weights_from_path", _update_weights_from_path)
    if not hasattr(Worker, "offload_model"):
        setattr(Worker, "offload_model", _offload_model)
    if not hasattr(Worker, "onload_model"):
        setattr(Worker, "onload_model", _onload_model)
