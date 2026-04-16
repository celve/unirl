"""Replay old log-prob patch for samplers that do not emit log_probs."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict

import torch

from diffusionrl.types.training_batch import TrainingBatch
from diffusionrl.utils import load_function

logger = logging.getLogger(__name__)


class ReplayLogProbPatch:
    """Recompute log_probs on training actors when rollout side omits them."""

    def __init__(self) -> None:
        self._replay_sampler = None
        self._replay_sampler_dotpath: str | None = None

    @staticmethod
    def _resolve_model_default_replay_sampler_dotpath(model_bundle: Any) -> str | None:
        for attr in (
            "default_replay_sampler_dotpath",
            "default_sampler_dotpath",
            "default_sampler_path",
        ):
            fn = getattr(model_bundle, attr, None)
            if callable(fn):
                try:
                    resolved = fn()
                except Exception as exc:
                    logger.debug("Model bundle %s() failed: %s", attr, exc)
                    continue
                if isinstance(resolved, str) and resolved.strip():
                    return resolved.strip()
        return None

    def _resolve_replay_sampler_dotpath(self, *, sampling_config: Dict[str, Any], model_bundle: Any) -> str:
        replay_path = sampling_config.get("replay_sampler_dotpath")
        if replay_path:
            return replay_path

        sampler_engine_type = str(
            sampling_config.get("sampler_engine_type", "") or ""
        ).strip().lower()
        sampler_dotpath = sampling_config.get("sampler_dotpath")
        model_default_path = self._resolve_model_default_replay_sampler_dotpath(model_bundle)

        # For sglang rollout engines, sampler_dotpath may still point to a legacy/default
        # FSDP sampler. Prefer model-declared replay sampler in this case.
        if sampler_engine_type == "sglang":
            if model_default_path and "sglang" not in model_default_path.lower():
                return model_default_path
        else:
            if sampler_dotpath and "sglang" not in sampler_dotpath.lower():
                return sampler_dotpath
            if model_default_path and "sglang" not in model_default_path.lower():
                return model_default_path

        model_type = str(getattr(model_bundle, "model_type", "") or "").lower()
        raise RuntimeError(
            "logprob_source='replay' requires a non-sglang replay sampler dotpath. Provide "
            "--replay-sampler-dotpath explicitly, or implement default_replay_sampler_dotpath() "
            f"in model bundle '{type(model_bundle).__name__}' (model_type={model_type!r}, "
            f"sampler_engine_type={sampler_engine_type!r})."
        )

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

        sampler_dotpath = self._resolve_replay_sampler_dotpath(
            sampling_config=sampling_config,
            model_bundle=model_bundle,
        )
        sampler_cls = load_function(sampler_dotpath)
        init_sig = inspect.signature(sampler_cls.__init__)
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in init_sig.parameters.values()
        )

        sampler_kwargs = dict(sampling_config.get("sampler_kwargs", {}) or {})
        for _reserved in ("autocast_precision", "trajectory_precision", "logprob_precision"):
            sampler_kwargs.pop(_reserved, None)
        base_kwargs: Dict[str, Any] = {
            "model": model,
            "text_encoder": text_encoder,
            "vae": vae,
            "scheduler": scheduler,
            "eta": sampling_config.get("eta", 1.0),
            "sde_type": sampling_config.get("sde_type", "flow"),
            "shift": sampling_config.get("shift", 3.0),
            "autocast_precision": sampling_config.get("autocast_precision", "bf16"),
            "trajectory_precision": sampling_config.get("trajectory_precision", "fp16"),
            "logprob_precision": sampling_config.get("logprob_precision", "fp32"),
            **sampler_kwargs,
        }
        filtered_kwargs: Dict[str, Any] = {}
        for key, value in base_kwargs.items():
            if key in init_sig.parameters or accepts_kwargs:
                filtered_kwargs[key] = value

        self._replay_sampler = sampler_cls(**filtered_kwargs)
        self._replay_sampler_dotpath = sampler_dotpath
        logger.warning(
            "Enabled experimental replay sampler for old log_probs: %s", sampler_dotpath
        )

    def maybe_replay_old_log_probs(
        self,
        *,
        batch: TrainingBatch,
        enabled: bool,
        algorithm_type: str,
        sampling_config: Dict[str, Any],
        model_bundle: Any,
        model: Any,
        text_encoder: Any,
        vae: Any,
        scheduler: Any,
    ) -> TrainingBatch:
        if not enabled or algorithm_type != "grpo" or (batch.log_probs is not None and len(batch.log_probs) > 0):
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
                f"sampler_dotpath={self._replay_sampler_dotpath}"
            )

        allowed_steps = set(int(v) for v in batch.resolved_step_indices[:-1].tolist())
        target_steps = sorted(int(i) for i in batch.sde_indices if int(i) in allowed_steps)
        if not target_steps:
            raise RuntimeError(
                "logprob_source='replay' requested but no target SDE steps were provided in batch."
            )

        replay_sig = inspect.signature(replay_fn)
        replayed: Dict[int, torch.Tensor] = {}
        for step_idx in target_steps:
            pos = batch.get_position_for_step(step_idx)
            # Replay samplers index into the provided sigma_schedule by position.
            # When rollout trajectories miss initial x_T (for example some sglang paths),
            # resolved step ids can be shifted (1..T) while sigma_schedule remains local (0..T-1).
            # Use local trajectory position to keep replay indexing in-bounds.
            local_step_index = int(pos)
            ctx_dict = batch.forward_context.to_dict()
            arg_map: Dict[str, Any] = {
                "latents": batch.trajectory_store.get_position(pos),
                "prev_latents": batch.trajectory_store.get_position(pos + 1),
                "timestep_index": local_step_index,
                "sigma_schedule": batch.timesteps,
                **ctx_dict,
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
                    f"sampler_dotpath={self._replay_sampler_dotpath}"
                )
            with torch.no_grad():
                old_log_prob = replay_fn(**call_kwargs)
            if not torch.is_tensor(old_log_prob):
                raise RuntimeError(
                    "Replay sampler must return torch.Tensor for old_log_prob, "
                    f"got {type(old_log_prob).__name__}"
                )
            replayed[int(step_idx)] = old_log_prob.detach()

        from diffusionrl.types.sampling import LogProbData
        batch.log_probs = LogProbData.from_dict(replayed)
        return batch
