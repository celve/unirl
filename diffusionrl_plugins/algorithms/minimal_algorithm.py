"""Minimal algorithm plugin example using the current algorithm API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Set, Tuple

import torch
import torch.nn as nn

from diffusionrl.algorithms.base import (
    BaseAlgorithm,
    BaseAlgorithmConfig,
    EMAConfig,
    SamplingRequirements,
)
from diffusionrl.algorithms.registry import register_algorithm
from diffusionrl.cmdline.algorithms import validate_algorithm_kwargs
from diffusionrl.cmdline.registry import register_cmdline_config_parser
from diffusionrl.types import TimestepData
from diffusionrl.types.sampling import SamplingParams


@dataclass(frozen=True)
class MinimalAlgorithmConfig(BaseAlgorithmConfig):
    skip_last_timestep: bool = False
    skip_initial_timesteps: int = 0


@register_cmdline_config_parser(MinimalAlgorithmConfig)
def build_minimal_algorithm_config_from_args(
    args: Any,
    *,
    sampling_spec: SamplingParams | None = None,
    **_: Any,
) -> MinimalAlgorithmConfig:
    """
    If one wants to use the framework's cmdline interface (train.py),
    define a function that build algorithm config from the framework's
    cmdline arguments. The built config will be passed to instantiate algorithm.
    """

    if not isinstance(sampling_spec, SamplingParams):
        from diffusionrl.cmdline.resolution import derive_sampling_spec

        sampling_spec = derive_sampling_spec(args).to_params(args.precision)
    extra = dict(getattr(args.algorithm, "algorithm_kwargs", {}) or {})
    validate_algorithm_kwargs(
        config_class=MinimalAlgorithmConfig,
        algorithm_kwargs=extra,
    )

    if extra.get("skip_last_timestep") is not None:
        extra["skip_last_timestep"] = bool(extra["skip_last_timestep"])
    if extra.get("skip_initial_timesteps") is not None:
        extra["skip_initial_timesteps"] = int(extra["skip_initial_timesteps"])

    return MinimalAlgorithmConfig(
        component_mix_stage=str(args.algorithm.component_mix_stage),
        adv_normalization_scope=str(args.algorithm.adv_normalization_scope),
        samples_per_prompt=int(args.algorithm.samples_per_prompt),
        sampling=sampling_spec,
        ema=EMAConfig(
            eval_ema_decay=float(args.algorithm.eval_ema_decay),
            eval_ema_update_interval=int(args.algorithm.eval_ema_update_interval),
        ),
        epsilon=float(args.algorithm.adv_norm_eps),
        clip_max=args.algorithm.clip_max,
        use_global_std=bool(args.algorithm.use_global_std),
        trim_outliers_ratio=float(args.algorithm.trim_outliers_ratio),
        **extra,
    )


@register_algorithm(component_name="minimal", component_cfg=MinimalAlgorithmConfig)
class MinimalAlgorithm(BaseAlgorithm):
    """
    Small GRPO-style plugin example with a placeholder loss.

    You can use a custom algorithm in two ways.

    1. Write your own driver script, or modify ``train.py``.
       In that case you can instantiate your algorithm config, init payload yourself in the driver,
       and do not need to integrate with the framework's cmdline parser registry.

    2. Reuse the framework's cmdline interface.
       In that case the framework needs to turn its cmdline args into your
       algorithm config object, so you should register a parser for your config:

       ```python
       @register_cmdline_config_parser(MinimalAlgorithmConfig)
       def build_minimal_algorithm_config_from_args(args, **_) -> MinimalAlgorithmConfig:
           ...
       ```

       Inside that parser you can interpret the framework cmdline args however
       you want and build your own config object.

    To bind your algorithm type name to both the algorithm class and its config
    class, register the algorithm with ``component_cfg``:

    ```python
    @register_algorithm(
        component_name="minimal",
        component_cfg=MinimalAlgorithmConfig,
    )
    class MinimalAlgorithm(BaseAlgorithm):
        ...
    ```

    Then users can select your algorithm through the framework's algorithm
    registry while still getting your custom cmdline-to-config adaptation.
    """

    @classmethod
    def _parse_config_from_dict(cls, config: dict) -> MinimalAlgorithmConfig:
        extra = cls.resolve_config_kwargs(config)
        return MinimalAlgorithmConfig(
            skip_last_timestep=bool(extra.get("skip_last_timestep", False)),
            skip_initial_timesteps=int(extra.get("skip_initial_timesteps", 0)),
            component_mix_stage=str(config.get("component_mix_stage", "reward")),
            adv_normalization_scope=str(config.get("adv_normalization_scope", "group")),
            samples_per_prompt=int(config.get("samples_per_prompt", 1)),
            sampling=SamplingParams(num_inference_steps=int(config.get("num_inference_steps", 0))),
            ema=EMAConfig(
                eval_ema_decay=float(config.get("eval_ema_decay", 0.9)),
                eval_ema_update_interval=int(config.get("eval_ema_update_interval", 1)),
            ),
            epsilon=float(config.get("adv_norm_eps", 1e-8)),
            clip_max=config.get("clip_max", 5.0),
            use_global_std=bool(config.get("use_global_std", False)),
            trim_outliers_ratio=float(config.get("trim_outliers_ratio", 0.0)),
        )

    @classmethod
    def from_config(
        cls,
        config: dict | MinimalAlgorithmConfig,
    ) -> "MinimalAlgorithm":
        if isinstance(config, dict):
            config = cls._parse_config_from_dict(config)
        if not isinstance(config, MinimalAlgorithmConfig):
            raise TypeError(
                f"{cls.__name__}.from_config expects dict or MinimalAlgorithmConfig, got {type(config).__name__}."
            )
        return cls(config=config)

    def __init__(
        self,
        *,
        config: MinimalAlgorithmConfig,
        **kwargs: Any,
    ) -> None:
        if not isinstance(config, MinimalAlgorithmConfig):
            raise TypeError(f"{type(self).__name__} expects MinimalAlgorithmConfig, got {type(config).__name__}.")
        super().__init__(
            component_mix_stage=config.component_mix_stage,
            adv_normalization_scope=config.adv_normalization_scope,
            samples_per_prompt=config.samples_per_prompt,
            sampling=config.sampling,
            ema=config.ema,
            epsilon=config.epsilon,
            clip_max=config.clip_max,
            use_global_std=config.use_global_std,
            trim_outliers_ratio=config.trim_outliers_ratio,
            **kwargs,
        )
        self.config = config
        self.skip_last_timestep = bool(config.skip_last_timestep)
        self.skip_initial_timesteps = int(config.skip_initial_timesteps)

    def get_sampling_requirements(self) -> SamplingRequirements:
        return SamplingRequirements(
            requires_trajectory=True,
            requires_log_prob=True,
            requires_embeddings=True,
        )

    def get_ema_spec(self) -> EMAConfig:
        return EMAConfig(enable_eval_ema=False)

    def resolve_rollout_sde_indices(
        self,
        *,
        current_step: int,
    ) -> Set[int]:
        del current_step
        if self.sampling.num_inference_steps < 1:
            raise ValueError(
                f"{type(self).__name__}.resolve_rollout_sde_indices requires "
                f"sampling.num_inference_steps >= 1, got {self.sampling.num_inference_steps}."
            )
        return set(range(self.sampling.num_inference_steps))

    def get_sampler_validation_config(self, *, allow_replay: bool) -> Dict[str, Any]:
        return {
            "allow_replay": bool(allow_replay),
            "assert_step_alignment": True,
            "mode_label": "trajectory",
        }

    def compute_loss(
        self,
        model: nn.Module,
        timestep_data: TimestepData,
        advantages: torch.Tensor,
        forward_context: Any = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Single loss entrypoint for debugging and training."""
        del model, advantages, forward_context, kwargs
        loss = timestep_data.latents.float().sum() * 0.0
        return loss, {"placeholder": True}

    def compute_loss_and_backward(
        self,
        *,
        model: nn.Module,
        batch: Any,
        timesteps: Any = None,
        loss_scale: float = 1.0,
        **kwargs: Any,
    ) -> tuple:
        del kwargs
        from diffusionrl.types.training_batch import TrainingBatch

        if not isinstance(batch, TrainingBatch):
            raise TypeError(f"{type(self).__name__} expects TrainingBatch, got {type(batch).__name__}")

        valid_step_indices = sorted(batch.sde_indices)
        if not valid_step_indices:
            return 0.0, {}, 0, False

        total_loss = 0.0
        has_backward = False
        num_timesteps = len(valid_step_indices)

        for t_idx in valid_step_indices:
            timestep_data = batch.get_timestep_data_by_step(t_idx)
            loss, _ = self.compute_loss(
                model=model,
                timestep_data=timestep_data,
                advantages=batch.advantages,
                forward_context=batch.forward_context,
            )
            scaled_loss = loss * loss_scale / num_timesteps
            scaled_loss.backward()
            total_loss += scaled_loss.detach().item()
            has_backward = True

        return total_loss, {"placeholder": 1.0}, num_timesteps, has_backward

    def get_filtered_training_indices(
        self,
        sde_indices: Set[int],
        num_steps: int,
    ) -> Set[int]:
        del num_steps
        result = set(sde_indices)
        if self.skip_last_timestep and result:
            result.discard(max(result))
        if self.skip_initial_timesteps > 0:
            result = {i for i in result if i >= self.skip_initial_timesteps}
        return result

    def resolve_training_timesteps(
        self,
        *,
        batch: Any,
        current_step: int,
        **kwargs: Any,
    ) -> Any:
        from diffusionrl.types.training_batch import TrainingBatch

        del current_step, kwargs
        if not isinstance(batch, TrainingBatch):
            raise TypeError(f"{type(self).__name__} expects TrainingBatch, got {type(batch).__name__}")

        step_indices = batch.resolved_step_indices[:-1]
        step_labels = set(int(v) for v in step_indices.tolist())
        if not step_labels:
            return tuple()

        filtered_steps = self.get_filtered_training_indices(
            set(int(i) for i in batch.sde_indices),
            len(step_labels),
        )
        missing_steps = sorted(int(i) for i in filtered_steps if int(i) not in step_labels)
        if missing_steps:
            raise ValueError(
                f"{type(self).__name__}.resolve_training_timesteps selected steps "
                f"not present in batch: missing={missing_steps}, "
                f"available={sorted(step_labels)}"
            )
        if not filtered_steps:
            return tuple()

        # With step_idx == position, step labels are just range(T),
        # so we can directly index into batch.timesteps by step label.
        selected_positions = sorted(filtered_steps)
        return batch.timesteps[selected_positions]
