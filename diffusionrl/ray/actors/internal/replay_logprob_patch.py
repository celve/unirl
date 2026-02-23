"""Replay old log-prob patch for samplers that do not emit log_probs."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict

import torch

from diffusionrl.types.training_batch import BackwardTrainingBatch
from diffusionrl.utils import load_function

logger = logging.getLogger(__name__)


class ReplayLogProbPatch:
    """Recompute old log_probs on training actors for compatibility paths."""

    def __init__(self) -> None:
        self._replay_sampler = None
        self._replay_sampler_path: str | None = None

    def _resolve_replay_sampler_path(self, *, sampling_config: Dict[str, Any], model_bundle: Any) -> str:
        replay_path = sampling_config.get("replay_sampler_path")
        if replay_path:
            return replay_path

        sampler_path = sampling_config.get("sampler_path")
        if sampler_path and "sglang" not in sampler_path.lower():
            return sampler_path

        model_type = getattr(model_bundle, "model_type", None)
        fallback = {
            "flux": "diffusionrl.samplers.fsdp.flux_sampler.FluxSampler",
            "hunyuan": "diffusionrl.samplers.fsdp.hunyuan_sampler.FSDPHunyuanSampler",
            "sd3": "diffusionrl.samplers.fsdp.sd3_sampler.SD3Sampler",
            "mochi": "diffusionrl.samplers.fastvideo.fastvideo_sampler.FastVideoSampler",
        }
        resolved = fallback.get(model_type)
        if not resolved:
            raise RuntimeError(
                "replay_log_probs requires replay_sampler_path or a known model_type fallback. "
                f"model_type={model_type!r}"
            )
        return resolved

    def _build_replay_sampler(
        self,
        *,
        sampling_config: Dict[str, Any],
        model_bundle: Any,
        model: Any,
        text_encoder: Any,
        vae: Any,
        scheduler: Any,
    ) -> None:
        if self._replay_sampler is not None:
            return
        if model is None:
            raise RuntimeError("Model not initialized for replay sampler")

        sampler_path = self._resolve_replay_sampler_path(
            sampling_config=sampling_config,
            model_bundle=model_bundle,
        )
        sampler_cls = load_function(sampler_path)
        init_sig = inspect.signature(sampler_cls.__init__)
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in init_sig.parameters.values()
        )

        sampler_kwargs = dict(sampling_config.get("sampler_kwargs", {}) or {})
        base_kwargs: Dict[str, Any] = {
            "model": model,
            "text_encoder": text_encoder,
            "vae": vae,
            "scheduler": scheduler,
            "eta": sampling_config.get("eta", 1.0),
            "sde_type": sampling_config.get("sde_type", "sde"),
            "shift": sampling_config.get("shift", 3.0),
            **sampler_kwargs,
        }
        filtered_kwargs: Dict[str, Any] = {}
        for key, value in base_kwargs.items():
            if key in init_sig.parameters or accepts_kwargs:
                filtered_kwargs[key] = value

        self._replay_sampler = sampler_cls(**filtered_kwargs)
        self._replay_sampler_path = sampler_path
        logger.warning("Enabled experimental replay sampler for old log_probs: %s", sampler_path)

    def maybe_replay_old_log_probs(
        self,
        *,
        batch: BackwardTrainingBatch,
        enabled: bool,
        loss_type: str,
        sampling_config: Dict[str, Any],
        model_bundle: Any,
        model: Any,
        text_encoder: Any,
        vae: Any,
        scheduler: Any,
        guidance_scale: float,
    ) -> BackwardTrainingBatch:
        if not enabled or loss_type != "grpo" or len(batch.log_probs) > 0:
            return batch

        self._build_replay_sampler(
            sampling_config=sampling_config,
            model_bundle=model_bundle,
            model=model,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
        )

        replay_sampler = self._replay_sampler
        replay_fn = getattr(replay_sampler, "compute_log_prob_for_training", None)
        if replay_fn is None:
            raise RuntimeError(
                "Replay sampler does not implement compute_log_prob_for_training; "
                f"sampler_path={self._replay_sampler_path}"
            )

        allowed_steps = set(int(v) for v in batch.resolved_step_indices[:-1].tolist())
        target_steps = sorted(int(i) for i in batch.sde_indices if int(i) in allowed_steps)
        if not target_steps:
            raise RuntimeError(
                "replay_log_probs enabled but no target SDE steps were provided in batch."
            )

        replay_sig = inspect.signature(replay_fn)
        replayed: Dict[int, torch.Tensor] = {}
        for step_idx in target_steps:
            pos = batch.get_position_for_step(step_idx)
            arg_map: Dict[str, Any] = {
                "latents": batch.trajectories[:, pos],
                "prev_latents": batch.trajectories[:, pos + 1],
                "prompt_embeds": batch.embeddings.prompt_embeds,
                "pooled_prompt_embeds": batch.embeddings.pooled_prompt_embeds,
                "encoder_attention_mask": batch.embeddings.encoder_attention_mask,
                "negative_prompt_embeds": batch.embeddings.negative_prompt_embeds,
                "negative_pooled_prompt_embeds": batch.embeddings.negative_pooled_prompt_embeds,
                "text_ids": batch.embeddings.text_ids,
                "image_ids": batch.embeddings.image_ids,
                "timestep_index": int(step_idx),
                "sigma_schedule": batch.timesteps,
                "guidance_scale": guidance_scale,
            }
            call_kwargs = {
                name: value
                for name, value in arg_map.items()
                if name in replay_sig.parameters
            }
            missing_required = [
                name
                for name, param in replay_sig.parameters.items()
                if (
                    param.default is inspect.Parameter.empty
                    and param.kind in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
                    and name not in call_kwargs
                )
            ]
            if missing_required:
                raise RuntimeError(
                    f"Replay sampler missing required args {missing_required} for step={step_idx}. "
                    f"sampler_path={self._replay_sampler_path}"
                )
            critical_non_null = {"text_ids", "image_ids"}
            missing_non_null = [
                name
                for name, param in replay_sig.parameters.items()
                if (
                    name in critical_non_null
                    and param.default is inspect.Parameter.empty
                    and param.kind
                    in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
                    and name in call_kwargs
                    and call_kwargs[name] is None
                )
            ]
            if missing_non_null:
                raise RuntimeError(
                    "Replay sampler requires non-null args "
                    f"{missing_non_null} for step={step_idx}. "
                    f"sampler_path={self._replay_sampler_path}"
                )
            with torch.no_grad():
                old_log_prob = replay_fn(**call_kwargs)
            if not torch.is_tensor(old_log_prob):
                raise RuntimeError(
                    "Replay sampler must return torch.Tensor for old_log_prob, "
                    f"got {type(old_log_prob).__name__}"
                )
            replayed[int(step_idx)] = old_log_prob.detach()

        batch.log_probs = type(batch.log_probs).from_dict(replayed)
        return batch
